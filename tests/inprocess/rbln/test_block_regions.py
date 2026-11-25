# SPDX-License-Identifier: Apache-2.0
"""A GLM-5.3 chunk restores every stored cache byte-exact into other blocks.

``RBLNConnector`` moves the DSA latent as a token region on the grouped path
and every other stored cache as a block region. Caches are CPU tensors of the
per-block shapes vLLM registered for GLM-5.3 on an NPU. The KDA states and the
DSA latent share one allocation, as vLLM allocates them, so a copy that
addressed the wrong cache or block would show up as a byte mismatch.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.rbln.integration.vllm.cache_layout import CacheLayout
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# Local
from . import kv_spec_fixtures as fx

BS = fx.GLM_BLOCK_SIZE
NUM_BLOCKS = 5
FIXTURE = fx.glm53(4)
LAYOUT = CacheLayout.from_kv_cache_config(FIXTURE.kv_cache_config, FIXTURE.aux_specs)


def name(layer: int, slot: int) -> str:
    return f"model.layers.{layer}.self_attn.attn.{slot}"


LATENT = name(3, 0)
TAIL = name(3, 3)


def _caches(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rand(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        raw = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
        if dtype == torch.uint8:
            return raw
        # Small integers are exact in every float dtype here.
        return (raw.to(torch.float32) / 16).to(dtype)

    # The KDA states and the DSA latent are views of one allocation.
    shared = rand((NUM_BLOCKS, BS, 512), torch.uint8)
    return {
        cache_name: shared
        if dtype == torch.uint8
        else rand((NUM_BLOCKS, *block_shape), dtype)
        for cache_name, (block_shape, dtype) in FIXTURE.registered.items()
    }


def _zeros() -> dict[str, torch.Tensor]:
    dst = {k: torch.zeros_like(v) for k, v in _caches(seed=2).items()}
    shared = dst[LATENT]
    return {k: shared if v.dtype == torch.uint8 else v for k, v in dst.items()}


def _connector(caches: dict[str, torch.Tensor]) -> RBLNConnector:
    (latent,) = LAYOUT.kv_regions
    connector = RBLNConnector(
        num_layers=1,
        num_kv_heads=1,
        head_dim=latent.head_size or 0,
        block_size=BS,
        dtype=latent.dtype,
        use_mla=True,
        block_groups=LAYOUT.block_groups,
    )
    connector.register_kv_caches(
        [caches[n] for n in latent.layer_names],
        layer_group_indices=[0],
        block_kv_caches=[
            [(caches[c.name], c.group) for c in r.caches] for r in LAYOUT.block_regions
        ],
    )
    return connector


def _memory_obj(connector: RBLNConnector, tokens: int = BS) -> TensorMemoryObj:
    # As RBLNLMCacheMetadata.get_shapes sizes it: tokens at dim 2 of region 0.
    shapes = [torch.Size([1, 1, tokens, 512])] + connector.get_shapes(tokens)[1:]
    dtypes = [LAYOUT.kv_regions[0].dtype] + [r.dtype for r in LAYOUT.block_regions]
    nbytes = sum(s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=True))
    meta = MemoryObjMetadata(
        shape=shapes[0],
        dtype=dtypes[0],
        address=0,
        phy_size=nbytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_MLA_FMT,
        shapes=shapes,
        dtypes=dtypes,
    )
    return TensorMemoryObj(torch.zeros(nbytes, dtype=torch.uint8), meta, None)


def _slots(blocks: list[int]) -> torch.Tensor:
    """``[T, groups]`` slot mapping of one chunk; -1 for a null block."""
    columns = [
        torch.full((BS,), -1) if b < 0 else b * BS + torch.arange(BS) for b in blocks
    ]
    return torch.stack(columns, dim=1)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _group_of(cache_name: str) -> int:
    if cache_name == LATENT:
        return 0
    return next(
        c.group for r in LAYOUT.block_regions for c in r.caches if c.name == cache_name
    )


# Group order: g0 = DSA latent/index key, g1..g3 = KDA layers 0..2.
SRC_BLOCKS = [1, 2, 3, 4]
DST_BLOCKS = [0, 4, 2, 3]


def test_every_stored_cache_round_trips_into_other_blocks() -> None:
    src = _caches(seed=1)
    store = _connector(src)
    obj = _memory_obj(store)
    assert obj.get_num_tokens() == BS
    store.from_gpu(obj, 0, BS, slot_mapping=_slots(SRC_BLOCKS))

    dst = _zeros()
    _connector(dst).to_gpu(obj, 0, BS, slot_mapping=_slots(DST_BLOCKS))

    stored = [LATENT] + [c.name for r in LAYOUT.block_regions for c in r.caches]
    assert len(stored) == len(FIXTURE.registered) - 1
    for cache_name in stored:
        group = _group_of(cache_name)
        got = _bytes(dst[cache_name][DST_BLOCKS[group]])
        want = _bytes(src[cache_name][SRC_BLOCKS[group]])
        assert torch.equal(got, want), f"{cache_name} not restored"
    assert not _bytes(dst[TAIL]).any(), "the index tail is not part of the chunk"
    # Nothing outside the restored blocks was written.
    untouched = sorted(set(range(NUM_BLOCKS)) - set(DST_BLOCKS))
    assert not _bytes(dst[LATENT][untouched]).any()


def test_load_skips_checkpoint_regions_whose_block_is_null() -> None:
    src = _caches(seed=1)
    store = _connector(src)
    obj = _memory_obj(store)
    store.from_gpu(obj, 0, BS, slot_mapping=_slots(SRC_BLOCKS))

    dst = _zeros()
    # An earlier chunk of a longer hit: only the MLA group has a block.
    _connector(dst).to_gpu(obj, 0, BS, slot_mapping=_slots([0, -1, -1, -1]))

    for slot in (0, 1, 2):
        assert torch.equal(_bytes(dst[name(3, slot)][0]), _bytes(src[name(3, slot)][1]))
    assert not _bytes(dst[LATENT][1:]).any(), "a KDA state was written"
    assert not any(_bytes(dst[name(i, 1)]).any() for i in range(3))


def test_store_refuses_a_chunk_without_a_checkpoint_block() -> None:
    connector = _connector(_caches(seed=1))
    with pytest.raises(ValueError, match="has no block"):
        connector.from_gpu(
            _memory_obj(connector), 0, BS, slot_mapping=_slots([1, 2, -1, 4])
        )


def test_a_chunk_with_block_regions_is_exactly_one_block() -> None:
    connector = _connector(_caches(seed=1))
    obj = _memory_obj(connector, BS // 2)
    with pytest.raises(ValueError, match="one 4096-token block, got 2048"):
        connector.from_gpu(obj, 0, BS // 2, slot_mapping=_slots(SRC_BLOCKS))


def test_block_regions_copy_one_batch_per_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = _connector(_caches(seed=1))
    obj = _memory_obj(connector)
    batches: list[set[torch.dtype]] = []
    original = torch._foreach_copy_

    def record(dsts: list[torch.Tensor], srcs: list[torch.Tensor]) -> None:
        batches.append({t.dtype for t in dsts})
        original(dsts, srcs)

    monkeypatch.setattr(torch, "_foreach_copy_", record)
    connector.from_gpu(obj, 0, BS, slot_mapping=_slots(SRC_BLOCKS))
    assert all(len(b) == 1 for b in batches), batches
    block_dtypes = {r.dtype for r in LAYOUT.block_regions}
    assert block_dtypes <= set().union(*batches)


def test_registration_refuses_block_caches_the_layout_does_not_describe() -> None:
    caches = _caches(seed=1)
    (latent,) = LAYOUT.kv_regions
    connector = RBLNConnector(
        1, 1, 512, BS, torch.uint8, use_mla=True, block_groups=LAYOUT.block_groups
    )
    blocks = [
        [(caches[c.name], c.group) for c in r.caches] for r in LAYOUT.block_regions
    ]
    with pytest.raises(ValueError, match="block cache lists"):
        connector.register_kv_caches(
            [caches[LATENT]], layer_group_indices=[0], block_kv_caches=blocks[:-1]
        )
    scale = blocks[1][0]
    blocks[1][0] = (scale[0].to(torch.bfloat16), scale[1])
    with pytest.raises(ValueError, match=r"is not \[NB, \*\(1024,\)\] torch.float16"):
        connector.register_kv_caches(
            [caches[LATENT]], layer_group_indices=[0], block_kv_caches=blocks
        )
    with pytest.raises(ValueError, match="layer_group_indices"):
        connector.register_kv_caches(
            [caches[n] for n in latent.layer_names],
            block_kv_caches=[
                [(caches[c.name], c.group) for c in r.caches]
                for r in LAYOUT.block_regions
            ],
        )


def test_a_kda_only_chunk_round_trips_without_a_token_region() -> None:
    # GLM-5.3 cut to its first 3 layers: KDA states and conv rings only.
    fixture = fx.glm53(3)
    layout = CacheLayout.from_kv_cache_config(
        fixture.kv_cache_config, fixture.aux_specs
    )
    assert not layout.kv_regions

    def caches(seed: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        return {
            n: torch.randint(0, 256, (NUM_BLOCKS, *shape), generator=g)
            .to(torch.uint8)
            .view(dtype)
            if dtype.itemsize == 1
            else torch.randint(0, 16, (NUM_BLOCKS, *shape), generator=g).to(dtype)
            for n, (shape, dtype) in fixture.registered.items()
        }

    def connector(c: dict[str, torch.Tensor]) -> RBLNConnector:
        conn = RBLNConnector(
            0, 1, 0, BS, torch.uint8, use_mla=True, block_groups=layout.block_groups
        )
        conn.register_kv_caches(
            [],
            layer_group_indices=[],
            block_kv_caches=[
                [(c[x.name], x.group) for x in r.caches] for r in layout.block_regions
            ],
        )
        return conn

    src, dst = caches(1), {k: torch.zeros_like(v) for k, v in caches(2).items()}
    store = connector(src)
    shapes = store.get_shapes(BS)
    assert shapes[0][2] == BS, "LMCache reads tokens at dim 2 of region 0"
    dtypes = [r.dtype for r in layout.block_regions]
    nbytes = sum(s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=True))
    meta = MemoryObjMetadata(
        shape=shapes[0],
        dtype=dtypes[0],
        address=0,
        phy_size=nbytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_MLA_FMT,
        shapes=shapes,
        dtypes=dtypes,
    )
    obj = TensorMemoryObj(torch.zeros(nbytes, dtype=torch.uint8), meta, None)
    assert obj.get_num_tokens() == BS
    store.from_gpu(obj, 0, BS, slot_mapping=_slots([1, 2, 3]))
    connector(dst).to_gpu(obj, 0, BS, slot_mapping=_slots([4, 0, 2]))
    for r in layout.block_regions:
        for c in r.caches:
            got = _bytes(dst[c.name][[4, 0, 2][c.group]])
            assert torch.equal(got, _bytes(src[c.name][[1, 2, 3][c.group]])), c.name


def test_kda_states_and_conv_rings_use_their_own_groups() -> None:
    assert [_group_of(name(i, 0)) for i in range(4)] == [1, 2, 3, 0]
    assert [_group_of(name(i, 1)) for i in range(3)] == [1, 2, 3]
    assert _group_of(name(3, 2)) == 0

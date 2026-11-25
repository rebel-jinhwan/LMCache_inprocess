# SPDX-License-Identifier: Apache-2.0
"""DSA indexer caches travel with the MLA latent in one chunk.

DeepSeek-V3.2 (DSA) keeps, per layer, the packed MLA latent plus a lightning
indexer key cache (fp8, ``index_head_dim`` wide) and an fp16 scale cache under
an fp8 KV cache. A connector that moved only the latent restored a block whose
indexer state stayed as the fresh block held it (zeros), shifting the sparse
top-k and the output of every request that hit it. These tests pin that all
three caches round-trip byte-exact through one multi-group chunk, on the
whole-block fast path and the partial-slot path.
"""

# Third Party
from vllm.v1.kv_cache_interface import KVCacheSpec
import pytest
import torch

# First Party
from lmcache.inprocess.rbln.integration.vllm.cache_layout import CacheLayout
from lmcache.inprocess.rbln.integration.vllm.connector import RBLNLMCacheConnectorV1Impl
from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import RBLNLMCacheMetadata
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# Local
from . import kv_spec_fixtures as fx

NUM_LAYERS = 3
NUM_BLOCKS = 6
BLOCK_SIZE = 16
LATENT = 24  # stands in for the 768-byte packed latent row
INDEX = 8  # stands in for index_head_dim
AUX = [(NUM_LAYERS, INDEX, torch.float8_e4m3fn), (NUM_LAYERS, 1, torch.float16)]


def _caches(seed: int) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    g = torch.Generator().manual_seed(seed)

    def rand(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        raw = torch.randint(-100, 100, shape, generator=g, dtype=torch.int8)
        return raw if dtype == torch.int8 else raw.to(torch.float32).div(8).to(dtype)

    mla = [
        rand((NUM_BLOCKS, BLOCK_SIZE, LATENT), torch.int8) for _ in range(NUM_LAYERS)
    ]
    keys = [
        rand((NUM_BLOCKS, BLOCK_SIZE, INDEX), torch.float8_e4m3fn)
        for _ in range(NUM_LAYERS)
    ]
    scales = [rand((NUM_BLOCKS, BLOCK_SIZE), torch.float16) for _ in range(NUM_LAYERS)]
    return mla, [keys, scales]


def _connector(mla: list[torch.Tensor], aux: list[list[torch.Tensor]]) -> RBLNConnector:
    conn = RBLNConnector(
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_dim=LATENT,
        block_size=BLOCK_SIZE,
        dtype=torch.int8,
        use_mla=True,
        aux_groups=AUX,
    )
    conn.register_kv_caches(mla, aux_kv_caches=aux)
    return conn


def _memory_obj(conn: RBLNConnector, num_tokens: int) -> TensorMemoryObj:
    """A multi-group ``TensorMemoryObj`` laid out as the RDS allocator does."""
    shapes = conn.get_shapes(num_tokens)
    dtypes = [torch.int8] + [d for _, _, d in AUX]
    nbytes = sum(s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=True))
    meta = MemoryObjMetadata(
        shape=shapes[0],
        dtype=dtypes[0],
        address=0,
        phy_size=nbytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
        shapes=shapes,
        dtypes=dtypes,
    )
    return TensorMemoryObj(torch.zeros(nbytes, dtype=torch.uint8), meta, None)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _slots(block: int, first: int, n: int) -> torch.Tensor:
    return torch.arange(block * BLOCK_SIZE + first, block * BLOCK_SIZE + first + n)


@pytest.mark.parametrize(
    "first, n",
    [(0, BLOCK_SIZE), (3, 5)],
    ids=["whole-block fast path", "partial-slot path"],
)
def test_latent_and_indexer_state_round_trip_byte_exact(first: int, n: int) -> None:
    src_mla, src_aux = _caches(seed=1)
    src = _connector(src_mla, src_aux)
    obj = _memory_obj(src, n)
    src.from_gpu(obj, 0, n, slot_mapping=_slots(2, first, n))

    # Restore into a different, zeroed engine and a different block, as a
    # later request that got a fresh block would.
    dst_mla, dst_aux = _caches(seed=2)
    for t in dst_mla + dst_aux[0] + dst_aux[1]:
        t.zero_()
    dst = _connector(dst_mla, dst_aux)
    dst.to_gpu(obj, 0, n, slot_mapping=_slots(4, first, n))

    sel = slice(first, first + n)
    for a, b in zip(src_mla, dst_mla, strict=True):
        assert torch.equal(_bytes(a[2, sel]), _bytes(b[4, sel]))
    for kind in range(2):
        for a, b in zip(src_aux[kind], dst_aux[kind], strict=True):
            assert torch.equal(_bytes(a[2, sel]), _bytes(b[4, sel])), (
                f"aux group {kind} not restored"
            )
            # and nothing else in the fresh block was written
            assert int(_bytes(b[4, :first]).count_nonzero()) == 0


@pytest.mark.parametrize("num_blocks", [1, 2], ids=["one block", "two blocks"])
def test_whole_block_chunk_moves_every_group_in_one_copy(
    monkeypatch: pytest.MonkeyPatch, num_blocks: int
) -> None:
    calls: list[int] = []
    real = torch._foreach_copy_

    def counting(dsts: list[torch.Tensor], srcs: list[torch.Tensor]) -> None:
        calls.append(len(dsts))
        real(dsts, srcs)

    monkeypatch.setattr(torch, "_foreach_copy_", counting)
    n = num_blocks * BLOCK_SIZE
    src_mla, src_aux = _caches(seed=1)
    src = _connector(src_mla, src_aux)
    obj = _memory_obj(src, n)
    src.from_gpu(obj, 0, n, slot_mapping=_slots(1, 0, n))
    assert calls == [3 * NUM_LAYERS * num_blocks]

    calls.clear()
    dst_mla, dst_aux = _caches(seed=2)
    dst = _connector(dst_mla, dst_aux)
    dst.to_gpu(obj, 0, n, slot_mapping=_slots(3, 0, n))
    assert calls == [3 * NUM_LAYERS * num_blocks]
    blocks = slice(1, 1 + num_blocks)
    restored = slice(3, 3 + num_blocks)
    for a, b in zip(
        src_mla + src_aux[0] + src_aux[1],
        dst_mla + dst_aux[0] + dst_aux[1],
        strict=True,
    ):
        assert torch.equal(_bytes(a[blocks]), _bytes(b[restored]))


def test_register_without_the_indexer_caches_is_refused() -> None:
    mla, _ = _caches(seed=1)
    conn = RBLNConnector(
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_dim=LATENT,
        block_size=BLOCK_SIZE,
        dtype=torch.int8,
        use_mla=True,
        aux_groups=AUX,
    )
    with pytest.raises(ValueError, match="aux groups"):
        conn.register_kv_caches(mla)


def test_aux_groups_with_hybrid_block_tables_are_refused() -> None:
    # The grouped (HMA) transfer walks attention layers only, so it would
    # leave the indexer caches unrestored; the pair must not start at all.
    mla, aux = _caches(seed=1)
    conn = RBLNConnector(
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_dim=LATENT,
        block_size=BLOCK_SIZE,
        dtype=torch.int8,
        use_mla=True,
        aux_groups=AUX,
    )
    with pytest.raises(ValueError, match="per-layer cache group"):
        conn.register_kv_caches(
            mla, layer_group_indices=[0] * NUM_LAYERS, aux_kv_caches=aux
        )


def test_metadata_sizes_the_chunk_for_every_group() -> None:
    md = RBLNLMCacheMetadata(
        model_name="dsa",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.int8,
        kv_shape=(NUM_LAYERS, 1, 32, 1, LATENT),
        use_mla=True,
        kv_head_groups=[(NUM_LAYERS, 1)],
        aux_groups=AUX,
    )
    assert md.get_num_groups() == 3
    assert md.get_dtypes() == [torch.int8, torch.float8_e4m3fn, torch.float16]
    assert md.get_shapes(32)[1:] == [
        torch.Size([1, NUM_LAYERS, 32, INDEX]),
        torch.Size([1, NUM_LAYERS, 32, 1]),
    ]


def _layout(specs: dict[str, KVCacheSpec]) -> CacheLayout:
    return CacheLayout.from_kv_cache_config(fx.worker_kv_cache_config(specs))


def test_dsa_indexer_groups_follow_the_cache_dtype() -> None:
    assert _layout(fx.deepseek_v32("fp8", range(8)).specs).aux_groups == [
        (8, 128, torch.float8_e4m3fn),
        (8, 1, torch.bfloat16),
    ]
    assert _layout(fx.deepseek_v32("bf16", range(8)).specs).aux_groups == [
        (8, 128, torch.bfloat16)
    ]
    assert _layout(fx.qwen3().specs).aux_groups == []


def test_indexer_caches_are_split_from_the_attention_cache() -> None:
    layout = _layout(fx.deepseek_v32("fp8", range(14, 16)).specs)
    assert [r.layer_names for r in layout.kv_regions] == [
        ("model.layers.14.self_attn.attn", "model.layers.15.self_attn.attn")
    ]
    assert [r.layer_names for r in layout.aux_regions] == [
        (
            "model.layers.14.self_attn.indexer.k_cache",
            "model.layers.15.self_attn.indexer.k_cache",
        ),
        (
            "model.layers.14.self_attn.indexer.k_cache.scale_cache",
            "model.layers.15.self_attn.indexer.k_cache.scale_cache",
        ),
    ]


def test_dsa_without_scale_keeps_the_key_in_the_first_aux_group() -> None:
    layout = _layout(fx.deepseek_v32("bf16", range(3, 5)).specs)
    assert [r.layer_names for r in layout.aux_regions] == [
        tuple(f"model.layers.{i}.self_attn.indexer.k_cache" for i in (3, 4))
    ]


def test_a_lone_cache_is_the_main_kv_whatever_its_name() -> None:
    (_, spec), *_ = fx.qwen3().specs.items()
    layout = _layout({"model.layers.0.self_attn.indexer": spec})
    assert [r.layer_names for r in layout.kv_regions] == [
        ("model.layers.0.self_attn.indexer",)
    ]
    assert layout.aux_regions == ()


def test_a_cache_the_layout_does_not_place_is_refused_at_registration() -> None:
    # A side cache vLLM registers but the worker's specs do not list must not
    # be dropped from the chunk while every other cache is restored.
    model = fx.deepseek_v32("bf16", range(2))
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._cache_layout = _layout(model.specs)
    caches = {name: torch.empty(0) for name in model.specs}
    caches["model.layers.1.self_attn.side_cache"] = torch.empty(0)
    with pytest.raises(
        ValueError, match=r"\['model\.layers\.1\.self_attn\.side_cache'\]"
    ):
        impl.register_kv_caches(caches)


def test_an_aligned_negative_range_is_not_stored_as_a_whole_block() -> None:
    # Negative slots belong to reclaimed blocks, and a store must reject them.
    # Read as a block id, -1 would address the last cache block, so the
    # whole-block fast path must not take an aligned negative range.
    mla, aux = _caches(seed=1)
    conn = _connector(mla, aux)
    with pytest.raises(ValueError, match="reclaimed"):
        conn.from_gpu(
            _memory_obj(conn, BLOCK_SIZE),
            0,
            BLOCK_SIZE,
            slot_mapping=torch.arange(-BLOCK_SIZE, 0),
        )

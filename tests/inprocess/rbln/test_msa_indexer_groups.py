# SPDX-License-Identifier: Apache-2.0
"""MSA indexer key caches travel with the GQA K/V in one chunk (#256).

MiniMax-M3 keeps, next to the GQA K/V of every layer whose
``sparse_attention_freq`` is nonzero, a lightning-indexer key cache
``[NB, BS, sparse_index_dim]`` named ``<attn>.indexer``: bf16, or the KV cache's
uint8 container under fp8, with no scale cache. A connector that moved only
the K/V restored a block whose indexer keys stayed as the fresh block held
them, so the indexer picked other sparse blocks than a cold prefill would.
"""

# Standard
from dataclasses import replace

# Third Party
from vllm.distributed.utils import get_pp_indices
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MLAAttentionSpec
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
from .vllm_deployment import build_metadata, deployment_of

M3_LAYERS = 60
INDEX_DIM = 128


def _layout(model: fx.KVSpecFixture) -> CacheLayout:
    return CacheLayout.from_kv_cache_config(model.kv_cache_config())


def _local_sparse(pp_size: int, pp_rank: int, freq: list[int] = fx.M3_FREQ) -> int:
    start, end = get_pp_indices(len(freq), pp_rank, pp_size)
    return sum(1 for f in freq[start:end] if f != 0)


# --- layout -------------------------------------------------------------------


@pytest.mark.parametrize("kv, dtype", [("bf16", torch.bfloat16), ("fp8", torch.uint8)])
def test_msa_indexer_group_follows_the_cache_dtype(kv: str, dtype: torch.dtype) -> None:
    # One key group over the 57 sparse layers; never a scale group.
    assert _layout(fx.minimax_m3(kv)).aux_groups == [(57, INDEX_DIM, dtype)]


def test_msa_non_fp8_indexer_remains_bf16_with_a_different_kv_dtype() -> None:
    model = fx.minimax_m3("bf16")
    specs = {
        name: spec if name.endswith(".indexer") else replace(spec, dtype=torch.float16)
        for name, spec in model.specs.items()
    }
    layout = CacheLayout.from_kv_cache_config(model.kv_cache_config(specs))
    assert layout.aux_groups == [(57, INDEX_DIM, torch.bfloat16)]
    assert [r.dtype for r in layout.kv_regions] == [torch.float16]


@pytest.mark.parametrize("pp_size", [2, 4, 8])
def test_msa_indexer_group_counts_this_stage_sparse_layers(pp_size: int) -> None:
    counts = []
    for pp_rank in range(pp_size):
        stage = range(*get_pp_indices(M3_LAYERS, pp_rank, pp_size))
        groups = _layout(fx.minimax_m3("bf16", stage)).aux_groups
        counts.append(groups[0][0])
        assert groups == [(_local_sparse(pp_size, pp_rank), INDEX_DIM, torch.bfloat16)]
    # Every MSA layer is counted on exactly one stage; dense 0-2 on none.
    assert sum(counts) == 57


def test_stage_with_only_dense_layers_has_no_indexer_group() -> None:
    # The reduced feature-test config at PP2: stage 0 owns only dense layer 0.
    assert _layout(fx.minimax_m3("bf16", range(0, 1), freq=[0, 1])).aux_groups == []
    assert _layout(fx.minimax_m3("bf16", range(1, 2), freq=[0, 1])).aux_groups == [
        (1, INDEX_DIM, torch.bfloat16)
    ]


def test_model_without_indexer_caches_has_no_msa_group() -> None:
    assert _layout(fx.dense(4, 4, torch.bfloat16)).aux_groups == []


@pytest.mark.parametrize("indexer_first", [False, True])
def test_msa_indexer_cache_ending_in_indexer_is_split_out(indexer_first: bool) -> None:
    # Either registration order must give the same split (#256).
    model = fx.minimax_m3("bf16", range(5))
    items = list(model.specs.items())
    if indexer_first:
        items = sorted(items, key=lambda kv: not kv[0].endswith(".indexer"))
    layout = CacheLayout.from_kv_cache_config(model.kv_cache_config(dict(items)))
    assert [r.layer_names for r in layout.kv_regions] == [
        tuple(f"model.layers.{i}.self_attn.attn" for i in range(5))
    ]
    assert [r.layer_names for r in layout.aux_regions] == [
        tuple(f"model.layers.{i}.self_attn.attn.indexer" for i in (3, 4))
    ]


# --- metadata ----------------------------------------------------------------


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
@pytest.mark.parametrize("pp_size, pp_rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
def test_m3_metadata_puts_the_indexer_keys_in_the_chunk(
    kv: str, pp_size: int, pp_rank: int
) -> None:
    start, end = get_pp_indices(M3_LAYERS, pp_rank, pp_size)
    model = fx.minimax_m3(kv, range(start, end))
    md = build_metadata(
        deployment_of(model, num_layers=M3_LAYERS, pp_size=pp_size, pp_rank=pp_rank)
    )

    dtype = torch.uint8 if kv == "fp8" else torch.bfloat16
    n = _local_sparse(pp_size, pp_rank)
    assert md.kv_shape[0] == end - start
    assert md.kv_dtype == dtype
    assert md.aux_groups == [(n, INDEX_DIM, dtype)]
    assert md.get_shapes(16)[1:] == [torch.Size([1, n, 16, INDEX_DIM])]
    assert md.get_dtypes()[1:] == [dtype]


def test_m3_eagle3_draft_adds_attention_kv_but_no_indexer() -> None:
    target = fx.minimax_m3("bf16")
    name, draft = fx._qwen3_layer(M3_LAYERS, 32)
    model = replace(
        target,
        specs={**target.specs, name: draft},
        draft=fx.DraftFacts(num_layers=1, num_kv_heads=32),
    )
    md = build_metadata(deployment_of(model))

    assert md.kv_head_groups == [(M3_LAYERS, 4), (1, 32)]
    assert md.aux_groups == [(57, INDEX_DIM, torch.bfloat16)]


# --- caches -> layout -> connector -> store / restore --------------------------

NUM_BLOCKS = 6
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_DIM = 8
REDUCED_INDEX_DIM = 4  # below the 2 x 2 x 8 K/V row, as 128 is below M3's
REDUCED_FREQ = [0, 1, 1]  # one dense layer, two MSA layers


def _m3_specs(dtype: torch.dtype, indexer_first: bool) -> dict[str, KVCacheSpec]:
    specs: dict[str, KVCacheSpec] = {}
    for i, f in enumerate(REDUCED_FREQ):
        attn = f"language_model.model.layers.{i}.self_attn.attn"
        layer: dict[str, KVCacheSpec] = {
            attn: FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=NUM_KV_HEADS,
                head_size=HEAD_DIM,
                dtype=dtype,
            )
        }
        if f:
            layer[f"{attn}.indexer"] = MLAAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=REDUCED_INDEX_DIM,
                dtype=dtype,
            )
        for name in reversed(layer) if indexer_first else layer:
            specs[name] = layer[name]
    return specs


def _m3_kv_caches(
    seed: int, dtype: torch.dtype, indexer_first: bool
) -> dict[str, torch.Tensor]:
    """vLLM's ``kv_caches`` dict for the reduced M3: blocks-first GQA K/V
    ``[NB, 2, NH, 1, BS, D]`` per layer, ``[NB, BS, D_idx]`` indexer keys."""
    g = torch.Generator().manual_seed(seed)

    def rand(shape: tuple[int, ...]) -> torch.Tensor:
        raw = torch.randint(0, 120, shape, generator=g, dtype=torch.uint8)
        return raw if dtype == torch.uint8 else raw.to(torch.float32).div(8).to(dtype)

    caches: dict[str, torch.Tensor] = {}
    for name in _m3_specs(dtype, indexer_first):
        if name.endswith(".indexer"):
            caches[name] = rand((NUM_BLOCKS, BLOCK_SIZE, REDUCED_INDEX_DIM))
        else:
            caches[name] = rand((NUM_BLOCKS, 2, NUM_KV_HEADS, 1, BLOCK_SIZE, HEAD_DIM))
    return caches


def _connector(
    caches: dict[str, torch.Tensor], dtype: torch.dtype, indexer_first: bool
) -> RBLNConnector:
    layout = CacheLayout.from_kv_cache_config(
        fx.worker_kv_cache_config(_m3_specs(dtype, indexer_first))
    )
    conn = RBLNConnector(
        num_layers=len(REDUCED_FREQ),
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        dtype=dtype,
        kv_head_groups=layout.kv_head_groups,
        aux_groups=layout.aux_groups,
    )
    conn.register_kv_caches(
        [caches[name] for r in layout.kv_regions for name in r.layer_names],
        aux_kv_caches=[
            [caches[name] for name in r.layer_names] for r in layout.aux_regions
        ],
    )
    return conn


def _memory_obj(conn: RBLNConnector, num_tokens: int) -> TensorMemoryObj:
    shapes = conn.get_shapes(num_tokens)
    dtypes = [conn.dtype] + [d for _, _, d in conn.aux_groups]
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


@pytest.mark.parametrize("declare_indexer", [False, True])
def test_msa_indexer_group_mismatch_is_refused(declare_indexer: bool) -> None:
    caches = _m3_kv_caches(1, torch.bfloat16, indexer_first=False)
    attn = [n for n in caches if not n.endswith(".indexer")]
    indexer = [n for n in caches if n.endswith(".indexer")]
    conn = RBLNConnector(
        num_layers=len(REDUCED_FREQ),
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        dtype=torch.bfloat16,
        aux_groups=[(2, REDUCED_INDEX_DIM, torch.bfloat16)] if declare_indexer else [],
    )
    registered_aux = [] if declare_indexer else [[caches[name] for name in indexer]]
    with pytest.raises(ValueError, match="aux groups.*aux cache lists"):
        conn.register_kv_caches(
            [caches[name] for name in attn], aux_kv_caches=registered_aux
        )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _slots(block: int, first: int, n: int) -> torch.Tensor:
    return torch.arange(block * BLOCK_SIZE + first, block * BLOCK_SIZE + first + n)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8], ids=["KV16", "KV8"])
@pytest.mark.parametrize("indexer_first", [False, True], ids=["attn>idx", "idx>attn"])
@pytest.mark.parametrize(
    "first, n",
    [(0, BLOCK_SIZE), (3, 5)],
    ids=["whole-block fast path", "partial-slot path"],
)
@pytest.mark.parametrize("split_kv_copy", [False, True], ids=["", "split-kv-copy"])
def test_gqa_and_indexer_keys_round_trip_byte_exact(
    dtype: torch.dtype,
    indexer_first: bool,
    first: int,
    n: int,
    split_kv_copy: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LMCACHE_RBLN_SPLIT_KV_COPY copies K and V as two halves; the key-only
    # indexer group has one and must still be copied whole.
    if split_kv_copy:
        monkeypatch.setenv("LMCACHE_RBLN_SPLIT_KV_COPY", "1")
    else:
        monkeypatch.delenv("LMCACHE_RBLN_SPLIT_KV_COPY", raising=False)
    src_caches = _m3_kv_caches(1, dtype, indexer_first)
    src = _connector(src_caches, dtype, indexer_first)
    obj = _memory_obj(src, n)
    src.from_gpu(obj, 0, n, slot_mapping=_slots(2, first, n))

    # Restore into a different, zeroed engine and a different block, as a
    # later request that got a fresh block would.
    dst_caches = _m3_kv_caches(2, dtype, indexer_first)
    for t in dst_caches.values():
        t.zero_()
    dst = _connector(dst_caches, dtype, indexer_first)
    dst.to_gpu(obj, 0, n, slot_mapping=_slots(4, first, n))

    sel = slice(first, first + n)
    for name, a in src_caches.items():
        b = dst_caches[name]
        if name.endswith(".indexer"):
            got, want = b[4, sel], a[2, sel]
        else:
            got, want = b[4, :, :, :, sel], a[2, :, :, :, sel]
        assert torch.equal(_bytes(got), _bytes(want)), f"{name} not restored"

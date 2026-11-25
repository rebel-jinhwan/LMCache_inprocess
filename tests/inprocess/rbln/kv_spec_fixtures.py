# SPDX-License-Identifier: Apache-2.0
"""Per-layer vLLM KV cache specs of real models, as vllm-rbln registers them.

Each builder returns one model at one KV cache dtype: the spec of every cache
(``get_kv_cache_spec`` of each attention module) and the config facts
the RBLN connector's ``RBLNServiceFactory`` reads for the same model. A worker's
``KVCacheConfig`` comes from vLLM's own ``get_kv_cache_groups``, so a fixture
sees the same grouping a worker does (one ``UniformTypeKVCacheSpecs`` for DSA,
two groups for gpt-oss).

A new model that offloads through ``CacheLayout`` gets a builder here.
"""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

# Third Party
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SparseCacheRole,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm_rbln.v1.kv_cache import RBLNMambaSpec
import torch

BLOCK_SIZE = 2048

register_all_kvcache_specs(None)


@dataclass(frozen=True)
class DraftFacts:
    num_layers: int
    num_kv_heads: int
    method: str = "eagle3"
    hf_config: SimpleNamespace = field(default_factory=SimpleNamespace)


@dataclass(frozen=True)
class KVSpecFixture:
    """One model's caches, and the config facts main derives its layout from."""

    specs: dict[str, KVCacheSpec]
    cache_dtype: str
    num_layers: int
    num_kv_heads: int
    # ``model_config.get_head_size()``: the MLA latent before fp8 packing.
    head_size: int
    hf_config: SimpleNamespace
    draft: DraftFacts | None = None
    start_layer: int = 0
    # ``model_config.dtype`` and ``model_config.use_mla``.
    model_dtype: torch.dtype = torch.bfloat16
    use_mla: bool = False

    @property
    def block_size(self) -> int:
        """The block size every cache shares, as vLLM's ``cache_config`` has it."""
        (block_size,) = {spec.block_size for spec in self.specs.values()}
        return block_size

    def kv_cache_config(
        self, specs: dict[str, KVCacheSpec] | None = None
    ) -> KVCacheConfig:
        """The worker's ``KVCacheConfig``; ``specs`` reorders or edits the caches."""
        return worker_kv_cache_config(self.specs if specs is None else specs)


def worker_kv_cache_config(specs: dict[str, KVCacheSpec]) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=get_kv_cache_groups(_grouping_config(), specs),
    )


def _grouping_config(
    block_size: int = BLOCK_SIZE, mamba_cache_mode: str = "none"
) -> Any:
    """The fields of ``VllmConfig`` that ``get_kv_cache_groups`` reads."""
    hf = SimpleNamespace(model_type="fixture", architectures=["Fixture"])
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=block_size,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode=mamba_cache_mode,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
        ),
        model_config=SimpleNamespace(
            hf_config=hf, hf_text_config=hf, max_model_len=8 * block_size
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        max_in_flight_tokens=block_size,
        speculative_config=None,
    )


def _dsa_layer(i: int, fp8: bool, indexer: bool = True) -> dict[str, KVCacheSpec]:
    p = f"model.layers.{i}.self_attn"
    specs: dict[str, KVCacheSpec] = {}
    if fp8:
        specs[f"{p}.attn"] = MLAAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=512 + 128 + 64 * 2,
            dtype=torch.uint8,
            cache_dtype_str="auto",
        )
    else:
        specs[f"{p}.attn"] = MLAAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
            is_index_group_leader=True,
        )
    if not indexer:
        return specs
    specs[f"{p}.indexer.k_cache"] = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        cache_role=SparseCacheRole.INDEXER,
    )
    if fp8:
        specs[f"{p}.indexer.k_cache.scale_cache"] = MLAAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        )
    return specs


def deepseek_v32(
    kv: str, layers: range = range(61), draft: str | None = None
) -> KVSpecFixture:
    """DeepSeek-V3.2, 61 MLA layers with a DSA indexer each.

    fp8: packed latent ``uint8 [768]`` (vllm-rbln ``patches/mla.py``), fp8
    indexer key and a model-dtype scale cache (``patches/deepseek_v2.py``).
    bf16: upstream latent ``[576]`` and a bf16 indexer key, no scale cache.

    ``layers`` is one pipeline stage's slice. ``draft`` adds layer 61: an
    ``"mtp"`` layer with its own indexer, or an ``"eagle3"`` layer without one.
    """
    fp8 = kv == "fp8"
    specs: dict[str, KVCacheSpec] = {}
    for i in layers:
        specs.update(_dsa_layer(i, fp8))
    hf_config = SimpleNamespace(index_topk=2048, index_head_dim=128)
    draft_facts = None
    if draft is not None:
        specs.update(_dsa_layer(61, fp8, indexer=draft == "mtp"))
        draft_facts = DraftFacts(
            num_layers=1,
            num_kv_heads=1,
            method=draft,
            hf_config=hf_config if draft == "mtp" else SimpleNamespace(),
        )
    return KVSpecFixture(
        specs=specs,
        cache_dtype="fp8" if fp8 else "auto",
        num_layers=len(layers),
        num_kv_heads=1,
        head_size=576,
        hf_config=hf_config,
        draft=draft_facts,
        start_layer=layers.start,
        use_mla=True,
    )


M3_FREQ = [0, 0, 0] + [1] * 57


def minimax_m3(
    kv: str, layers: range = range(60), freq: list[int] = M3_FREQ
) -> KVSpecFixture:
    """MiniMax-M3, 60 GQA layers; layers 3-59 add an MSA indexer key cache.

    vllm-rbln ``models/minimax_m3.py``: K/V and indexer key in the KV cache's
    uint8 container under fp8, bf16 otherwise. ``layers`` is one pipeline
    stage's slice of the ``freq`` layers.
    """
    dtype = torch.uint8 if kv == "fp8" else torch.bfloat16
    specs: dict[str, KVCacheSpec] = {}
    for i in layers:
        p = f"model.layers.{i}.self_attn.attn"
        specs[p] = FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=4,
            head_size=128,
            head_size_v=128,
            dtype=dtype,
        )
        if freq[i]:
            specs[f"{p}.indexer"] = MLAAttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=128, dtype=dtype
            )
    return KVSpecFixture(
        specs=specs,
        cache_dtype="fp8" if kv == "fp8" else "auto",
        num_layers=len(layers),
        num_kv_heads=4,
        head_size=128,
        hf_config=SimpleNamespace(
            sparse_attention_config={
                "sparse_attention_freq": freq,
                "sparse_index_dim": 128,
            }
        ),
        start_layer=layers.start,
    )


def _qwen3_layer(
    i: int, num_kv_heads: int, dtype: torch.dtype = torch.bfloat16
) -> tuple[str, FullAttentionSpec]:
    return f"model.layers.{i}.self_attn.attn", FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=128,
        head_size_v=128,
        dtype=dtype,
    )


def dense(num_layers: int, num_kv_heads: int, dtype: torch.dtype) -> KVSpecFixture:
    """A GQA model of ``num_layers`` layers of 128-wide heads in ``dtype``."""
    return KVSpecFixture(
        specs=dict(_qwen3_layer(i, num_kv_heads, dtype) for i in range(num_layers)),
        cache_dtype="auto",
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=128,
        hf_config=SimpleNamespace(),
        model_dtype=dtype,
    )


def qwen3() -> KVSpecFixture:
    """Qwen3-8B, 36 layers of 8 KV heads."""
    return dense(36, 8, torch.bfloat16)


def qwen3_eagle3(draft_kv_heads: int = 4) -> KVSpecFixture:
    """Qwen3-8B with a one-layer Eagle3 draft whose head count differs.

    The draft layer is registered after the target's, as layer 36.
    """
    target = qwen3()
    name, spec = _qwen3_layer(36, draft_kv_heads)
    return KVSpecFixture(
        specs={**target.specs, name: spec},
        cache_dtype=target.cache_dtype,
        num_layers=target.num_layers,
        num_kv_heads=target.num_kv_heads,
        head_size=target.head_size,
        hf_config=target.hf_config,
        draft=DraftFacts(num_layers=1, num_kv_heads=draft_kv_heads),
        model_dtype=target.model_dtype,
    )


def gpt_oss() -> KVSpecFixture:
    """gpt-oss-20b, 24 layers alternating sliding window (128) and full.

    On a worker these come as upstream ``SlidingWindowSpec`` and
    ``FullAttentionSpec`` in two groups (seen on an NPU, 11-02).
    """
    specs: dict[str, KVCacheSpec] = {}
    for i in range(24):
        name = f"model.layers.{i}.attn.attn"
        if i % 2 == 0:
            specs[name] = SlidingWindowSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=8,
                head_size=64,
                dtype=torch.bfloat16,
                sliding_window=128,
            )
        else:
            specs[name] = FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=8,
                head_size=64,
                head_size_v=64,
                dtype=torch.bfloat16,
            )
    return KVSpecFixture(
        specs=specs,
        cache_dtype="auto",
        num_layers=24,
        num_kv_heads=8,
        head_size=64,
        hf_config=SimpleNamespace(),
    )


def gemma4() -> KVSpecFixture:
    """Gemma4-31B's attention, cut to 12 layers: five sliding (1024 window,
    16 heads of 256) to one full (4 heads of 512).

    The full layer's page is half the sliding one, so vLLM gives its group a
    block twice as long (seen on an NPU, 11-25).
    """
    specs: dict[str, KVCacheSpec] = {}
    for i in range(12):
        name = f"model.language_model.layers.{i}.self_attn.attn"
        if i % 6 == 5:
            specs[name] = FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=4,
                head_size=512,
                head_size_v=512,
                dtype=torch.bfloat16,
            )
        else:
            specs[name] = SlidingWindowSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=16,
                head_size=256,
                dtype=torch.bfloat16,
                sliding_window=1024,
            )
    return KVSpecFixture(
        specs=specs,
        cache_dtype="auto",
        num_layers=12,
        num_kv_heads=16,
        # ``get_head_size()`` reads the full layers' ``global_head_dim``.
        head_size=512,
        hf_config=SimpleNamespace(),
    )


GLM_BLOCK_SIZE = 4096
GLM_MAX_MODEL_LEN = 4 * GLM_BLOCK_SIZE


@dataclass(frozen=True)
class AuxSpec:
    """vllm-rbln's ``RBLNBlockAuxSpec`` with A1's ``prefix_cacheable``."""

    owner: str
    block_shape: tuple[int, ...]
    dtype: torch.dtype
    prefix_cacheable: bool = True
    verify_only: bool = False


@dataclass(frozen=True)
class GlmFixture:
    """A GLM-5.3 worker: its ``KVCacheConfig``, block-aux specs, and the
    per-block shape of every cache as ``register_kv_caches`` receives it."""

    kv_cache_config: KVCacheConfig
    aux_specs: dict[str, AuxSpec]
    registered: dict[str, tuple[tuple[int, ...], torch.dtype]]


def glm53(
    num_layers: int = 4,
    drafts: int = 0,
    kda_groups: int = 1,
    mamba_cache_mode: str = "align",
) -> GlmFixture:
    """GLM-5.3-Flash, fp8 KV, prefix caching ("align"), block 4096.

    ``mamba_cache_mode="none"`` is prefix caching off: one KDA state a request,
    in a block of ``GLM_MAX_MODEL_LEN`` tokens.

    Every fourth layer is DSA: an MLA latent ``attn.0`` (``uint8 [4096, 512]``)
    and an indexer key ``attn.1`` (fp8, one row per 4 tokens) in one group,
    the indexer scale ``attn.2`` and the tail ring ``attn.3`` as block-aux. A
    KDA layer has ``kda_groups`` states ``attn.<j>``, each an ``RBLNMambaSpec``
    group of a latent block's bytes, and a conv ring after them; MTP adds a
    draft cache per state. Specs follow vllm-rbln ``glm5_next.py``; the 4-layer
    grouping and registered shapes match the 10-30 NPU log
    (``results/r4_probe_names_pc_L4.log``), the aux specs A1's dump
    (``results/a1/glm-aux-specs-cpu.log``).
    """
    bs = GLM_BLOCK_SIZE
    specs: dict[str, KVCacheSpec] = {}
    aux: dict[str, AuxSpec] = {}
    registered: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for i in range(num_layers):
        p = f"model.layers.{i}.self_attn.attn"
        if i % 4 == 3:
            specs[f"{p}.0"] = MLAAttentionSpec(
                block_size=bs, num_kv_heads=1, head_size=512, dtype=torch.uint8
            )
            specs[f"{p}.1"] = MLAAttentionSpec(
                block_size=bs,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.float8_e4m3fn,
                tokens_per_state=4,
            )
            tail = 4 * -(-(3 + max(drafts, 1)) // 4)
            aux[f"{p}.2"] = AuxSpec(f"{p}.1", (bs // 4,), torch.float16)
            aux[f"{p}.3"] = AuxSpec(f"{p}.0", (tail, 256), torch.bfloat16, False)
            registered |= {
                f"{p}.0": ((bs, 512), torch.uint8),
                f"{p}.1": ((bs // 4, 128), torch.float8_e4m3fn),
                f"{p}.2": ((bs // 4, 1), torch.float16),
                f"{p}.3": ((tail, 256), torch.bfloat16),
            }
            continue
        g = kda_groups
        for j in range(g):
            specs[f"{p}.{j}"] = RBLNMambaSpec(
                block_size=bs if mamba_cache_mode == "align" else GLM_MAX_MODEL_LEN,
                shapes=((bs, 512),),
                dtypes=(torch.uint8,),
                mamba_cache_mode=mamba_cache_mode,
            )
            registered[f"{p}.{j}"] = ((bs, 512), torch.uint8)
        conv = (max(4, 3 + drafts), 3 * g, 128 // g, 64)
        aux[f"{p}.{g}"] = AuxSpec(f"{p}.0", conv, torch.bfloat16)
        registered[f"{p}.{g}"] = (conv, torch.bfloat16)
        for j in range(g if drafts else 0):
            draft = (64 // g, drafts, 4, 128)
            aux[f"{p}.{g + 1 + j}"] = AuxSpec(
                f"{p}.{j}", draft, torch.bfloat16, False, verify_only=True
            )
            registered[f"{p}.{g + 1 + j}"] = (draft, torch.bfloat16)
    grouping = _grouping_config(bs, mamba_cache_mode)
    if mamba_cache_mode == "none":
        grouping.cache_config.enable_prefix_caching = False
        grouping.model_config.max_model_len = GLM_MAX_MODEL_LEN
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=get_kv_cache_groups(grouping, specs),
    )
    return GlmFixture(config, aux, registered)

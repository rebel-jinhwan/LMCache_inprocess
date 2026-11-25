# SPDX-License-Identifier: Apache-2.0
"""What ``RBLNServiceFactory`` builds for a vLLM deployment.

Each test describes a deployment (see :mod:`.vllm_deployment`) and checks the
LMCache metadata and RBLN connector the factory derives from it.
"""

# Standard
from dataclasses import replace
from types import SimpleNamespace
from typing import Callable, Optional
from unittest.mock import MagicMock, patch
import math

# Third Party
from vllm.distributed.utils import get_pp_indices
import pytest
import torch

# First Party
from lmcache.inprocess.integration.vllm.utils import ENGINE_NAME
from lmcache.inprocess.integration.vllm.vllm_service_factory import (
    VllmServiceFactory,
)
from lmcache.inprocess.rbln.integration.vllm import vllm_v1_adapter
from lmcache.inprocess.rbln.integration.vllm.cache_layout import CacheLayout
from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import (
    RBLNLMCacheMetadata,
    RBLNServiceFactory,
    _packed_mla_kv,
    _packed_mla_kv_from_hf_config,
)
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector
from lmcache.inprocess.v1.cache_engine import LMCacheEngineBuilder
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.lookup_client import LMCacheBypassLookupClient

# Local
from . import kv_spec_fixtures as fx
from .vllm_deployment import (
    CHUNK_SIZE,
    Deployment,
    Draft,
    build_engine,
    build_metadata,
    deployment_of,
    make_factory,
)


def _layout(model: fx.KVSpecFixture) -> CacheLayout:
    return CacheLayout.from_kv_cache_config(model.kv_cache_config())


def test_is_a_vllm_service_factory():
    """vllm-rbln hands it to upstream's ``LMCacheManager`` in its place."""
    assert issubclass(RBLNServiceFactory, VllmServiceFactory)


class TestEngine:
    def test_reuses_the_running_engine(self):
        engine = MagicMock()
        factory = make_factory(Deployment())
        with (
            patch.object(vllm_v1_adapter, "_packed_mla_kv", return_value=None),
            patch.object(
                vllm_v1_adapter.LMCacheEngineBuilder, "get", return_value=engine
            ),
        ):
            assert factory.get_or_create_lmcache_engine() is engine

    def test_scheduler_gets_no_connector(self):
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=CHUNK_SIZE, enable_scheduler_bypass_lookup=True
        )
        _, connector = build_engine(Deployment(), "scheduler", config=config)
        assert connector is None

    @pytest.mark.parametrize(
        "deployment, num_layers, kv_head_groups",
        [
            pytest.param(Deployment(), 32, [(32, 8)], id="gqa"),
            pytest.param(
                Deployment(block_size=1024), 32, [(32, 8)], id="gqa-1024-blocks"
            ),
            pytest.param(
                Deployment(use_mla=True, num_kv_heads=1, head_size=576),
                32,
                [(32, 1)],
                id="mla",
            ),
            pytest.param(
                Deployment(draft=Draft("eagle3", num_kv_heads=48)),
                33,
                [(32, 8), (1, 48)],
                id="eagle3-draft-with-own-heads",
            ),
        ],
    )
    def test_worker_connector_follows_the_metadata(
        self,
        deployment: Deployment,
        num_layers: int,
        kv_head_groups: list[tuple[int, int]],
    ):
        factory, connector = build_engine(deployment)
        md = factory.metadata

        assert connector is not None and md is not None
        assert connector.num_layers == md.kv_shape[0] == num_layers
        assert connector.kv_head_groups == md.kv_head_groups == kv_head_groups
        assert connector.use_mla is md.use_mla is deployment.use_mla
        # Slot mappings index blocks of the caches vLLM allocated.
        assert connector.block_size == deployment.block_size
        # MLA stores one latent per token, so the chunk's K/V axis is 1.
        assert md.kv_shape[1] == (1 if deployment.use_mla else 2)


class TestPackedMLAKVLayout:
    """vllm-rbln allocates the sparse-MLA fp8 latent cache as a packed row that
    `cache_dtype` and `get_head_size()` do not describe (#184). The metadata has
    to carry what was allocated, or every CPU/RDS buffer is sized wrong."""

    DEPLOYMENT = deployment_of(fx.deepseek_v32("fp8"))

    def test_metadata_describes_the_packed_layout(self):
        md = build_metadata(self.DEPLOYMENT, packed_mla_kv=(768, torch.int8))

        assert md.kv_shape[4] == 768
        assert md.kv_dtype is torch.int8

    def test_metadata_keeps_model_geometry_when_not_packed(self):
        md = build_metadata(self.DEPLOYMENT)

        assert md.kv_shape[4] == 576
        assert md.kv_dtype is torch.uint8  # vLLM's fp8 byte container

    def test_packed_layout_is_read_from_vllm_rbln(self, monkeypatch):
        # Third Party
        import vllm_rbln.v1.worker.utils as vllm_rbln_worker_utils

        seen = []

        def _published(model_config, cache_dtype):
            seen.append((model_config, cache_dtype))
            return 768, torch.int8

        monkeypatch.setattr(
            vllm_rbln_worker_utils,
            "packed_sparse_mla_kv",
            _published,
            raising=False,
        )
        vllm_config = self.DEPLOYMENT.vllm_config()

        assert _packed_mla_kv(vllm_config) == (768, torch.int8)
        assert seen == [(vllm_config.model_config, "fp8")]


class TestPackedMLAKVFromHfConfig:
    """The bridge for a vllm-rbln that packs the row without publishing it."""

    def _vllm_config(self, cache_dtype="fp8", sparse=True):
        hf_config = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
        if sparse:
            hf_config.index_topk = 2048
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config),
            cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        )

    def test_recomputes_the_packed_row(self):
        assert _packed_mla_kv_from_hf_config(self._vllm_config()) == (768, torch.int8)

    def test_dense_mla_is_not_packed(self):
        assert _packed_mla_kv_from_hf_config(self._vllm_config(sparse=False)) is None

    def test_bf16_kv_is_not_packed(self):
        config = self._vllm_config(cache_dtype="auto")
        assert _packed_mla_kv_from_hf_config(config) is None

    def test_packed_without_vllm_mla_flag(self):
        """An out-of-tree MLA model is absent from vLLM's model table, so
        `model_config.use_mla` is False while its layers still pack the row."""
        config = self._vllm_config()
        assert not hasattr(config.model_config, "use_mla")

        assert _packed_mla_kv_from_hf_config(config) == (768, torch.int8)


class TestKVHeadGroups:
    """A speculative draft joins the target's KV cache with its own layers;
    when its KV head count differs (eagle3 MHA heads over a GQA target), the
    chunk needs one region per head count, sized here before any allocation."""

    def test_draft_with_its_own_head_count_gets_its_own_group(self):
        md = build_metadata(Deployment(draft=Draft("eagle3", num_kv_heads=48)))

        assert md.kv_head_groups == [(32, 8), (1, 48)]
        assert md.kv_shape[0] == 33
        assert md.get_num_groups() == 2
        assert md.get_shapes(256) == [
            torch.Size([2, 32, 256, 8 * 128]),
            torch.Size([2, 1, 256, 48 * 128]),
        ]
        assert md.get_dtypes() == [torch.float16, torch.float16]

    def test_draft_matching_the_target_folds_into_one_group(self):
        md = build_metadata(Deployment(draft=Draft("eagle3", num_kv_heads=8)))

        assert md.kv_head_groups == [(33, 8)]
        assert md.get_num_groups() == 1
        assert md.get_shapes(256) == [torch.Size([2, 33, 256, 8 * 128])]

    def test_no_draft_is_one_group(self):
        md = build_metadata(Deployment())

        assert md.kv_head_groups == [(32, 8)]
        assert md.get_shapes() == [torch.Size([2, 32, 256, 8 * 128])]


class TestDraftLayersOnLastPPStage:
    """vllm-rbln builds every EAGLE-family draft, MTP included, only on the
    last pipeline stage, so its layers must be counted there -- not where
    upstream's helper lands them by splitting the draft over the target's
    pipeline size. DeepSeek-V3.2 at pp=8 is the reported case: 61 target
    layers split 7,7,8,8,8,8,8,7 and one MTP layer that upstream puts on
    stage 6."""

    TARGET_LAYERS = 61
    PP_SIZE = 8

    def _num_layer(self, method: str, pp_rank: int) -> int:
        deployment = Deployment(
            num_layers=self.TARGET_LAYERS,
            pp_size=self.PP_SIZE,
            pp_rank=pp_rank,
            draft=Draft(method, num_layers=1, num_kv_heads=8),
        )
        return build_metadata(deployment).kv_shape[0]

    @pytest.mark.parametrize("method", ["eagle", "eagle3", "mtp", "dflash"])
    def test_draft_counted_on_last_stage_only(self, method: str) -> None:
        for pp_rank in range(self.PP_SIZE):
            start, end = get_pp_indices(self.TARGET_LAYERS, pp_rank, self.PP_SIZE)
            draft = 1 if pp_rank == self.PP_SIZE - 1 else 0
            assert self._num_layer(method, pp_rank) == end - start + draft, (
                f"pp_rank={pp_rank}"
            )

    def test_deepseek_v32_pp8_mtp_stages_6_and_7(self) -> None:
        """The two stages the report names: both register 8 KV caches."""
        assert self._num_layer("mtp", 6) == 8
        assert self._num_layer("mtp", 7) == 8


# --- the worker's KV cache specs -----------------------------------------------


def _packed(model: fx.KVSpecFixture) -> Optional[tuple[int, torch.dtype]]:
    """What the hf_config fallback describes for a packed fp8 DSA latent."""
    if model.cache_dtype == "fp8" and hasattr(model.hf_config, "index_topk"):
        return (768, torch.int8)
    return None


KEY_FIXTURES: dict[str, Callable[[], fx.KVSpecFixture]] = {
    "DSA fp8": lambda: fx.deepseek_v32("fp8"),
    "DSA bf16": lambda: fx.deepseek_v32("bf16"),
    "DSA fp8 + MTP": lambda: fx.deepseek_v32("fp8", draft="mtp"),
    "MSA bf16": lambda: fx.minimax_m3("bf16"),
    "MSA fp8": lambda: fx.minimax_m3("fp8"),
    "Qwen3": fx.qwen3,
    "Qwen3 + Eagle3 draft of 4 heads": fx.qwen3_eagle3,
    "gpt-oss": fx.gpt_oss,
    "Gemma4": fx.gemma4,
}


@pytest.mark.parametrize("name", KEY_FIXTURES)
def test_scheduler_and_worker_build_the_same_cache_key(name: str) -> None:
    """A chunk's key carries these fields, and the scheduler's lookup client
    hashes with its own copy, so a field that differs from a worker's turns
    every lookup into a miss. The worker's layout must not feed any of them."""
    model = KEY_FIXTURES[name]()
    deployment = deployment_of(model)
    worker = build_metadata(deployment, "worker", packed_mla_kv=_packed(model))
    scheduler = build_metadata(deployment, "scheduler", packed_mla_kv=_packed(model))
    fields = ("kv_dtype", "use_mla", "model_name", "world_size", "worker_id")
    assert {f: getattr(scheduler, f) for f in fields} == {
        f: getattr(worker, f) for f in fields
    }
    assert scheduler.kv_head_groups == scheduler.aux_groups == []
    assert scheduler.kv_buffer_dtype is None
    assert worker.kv_head_groups == _layout(model).kv_head_groups
    assert worker.aux_groups == _layout(model).aux_groups


@pytest.mark.parametrize("name", KEY_FIXTURES)
def test_config_geometry_agrees_with_the_worker_specs(name: str) -> None:
    """``kv_shape`` and ``kv_dtype`` stay config-derived because buffers and
    keys are sized from them on both roles; ``CacheLayout`` carries no main-KV
    head size. They must still describe the caches vLLM allocated."""
    model = KEY_FIXTURES[name]()
    md = build_metadata(deployment_of(model), packed_mla_kv=_packed(model))
    for region in _layout(model).kv_regions:
        assert md.buffer_kv_dtype == region.dtype
        if md.sized_head_groups:
            # Each group carries its own head size, read from these specs:
            # test_head_groups_of_several_head_sizes_size_their_own_regions.
            continue
        for layer in region.layer_names:
            assert model.specs[layer].head_size == md.kv_shape[4], layer


def test_head_groups_of_several_head_sizes_size_their_own_regions() -> None:
    """Gemma4's sliding (16 x 256) and full (4 x 512) layers each fill a region
    of their own head size, and the CPU budget counts what a chunk holds, not
    every layer at the config's 512."""
    factory, connector = build_engine(deployment_of(fx.gemma4()))
    md = factory.metadata
    assert md is not None and connector is not None
    assert md.kv_head_groups == [(10, 16, 256), (2, 4, 512)]
    tokens = md.chunk_size
    assert md.get_shapes() == [
        torch.Size([2, 10, tokens, 16 * 256]),
        torch.Size([2, 2, tokens, 4 * 512]),
    ]
    sizes = [math.prod(s) for s in md.get_shapes()]
    assert math.prod(md.kv_shape) == sum(sizes)
    assert [math.prod(s) for s in connector.get_shapes(tokens)] == sizes


@pytest.mark.parametrize("bypass", [False, True])
def test_scheduler_path_never_sizes_a_chunk(
    monkeypatch: pytest.MonkeyPatch, bypass: bool
) -> None:
    """The scheduler cannot read a ``CacheLayout`` (its config collapses each
    group to one layer's spec), so its metadata has no groups. That is sound
    only while nothing it runs sizes a chunk: with
    ``enable_scheduler_bypass_lookup`` it even builds an engine, without a
    connector, and keys lookups with that engine's token database."""

    def sized(*args: object, **kwargs: object) -> None:
        raise AssertionError("the scheduler sized a chunk")

    for name in ("get_shapes", "get_dtypes", "get_num_groups"):
        monkeypatch.setattr(RBLNLMCacheMetadata, name, sized)
    monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")

    model = fx.deepseek_v32("fp8", draft="mtp")
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, enable_scheduler_bypass_lookup=bypass, max_local_cpu_size=0.01
    )
    factory = make_factory(deployment_of(model), "scheduler", config=config)
    try:
        with patch.object(
            vllm_v1_adapter, "_packed_mla_kv", return_value=_packed(model)
        ):
            engine = factory.get_or_create_lmcache_engine()
        if not bypass:
            assert engine is None
            return
        assert engine is not None and engine.gpu_connector is None
        client = LMCacheBypassLookupClient(
            factory.lmcache_config, factory.metadata, engine
        )
        keys = list(client.token_database.process_tokens(list(range(1024))))
        assert len(keys) == 4
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


_DSA_HF = SimpleNamespace(index_topk=2048, index_head_dim=128)


@pytest.mark.parametrize("method", ["eagle3", "mtp"])
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("pp_size, pp_rank", [(1, 0), (8, 6), (8, 7)])
def test_dsa_draft_metadata_registers_only_existing_indexer_caches(
    method: str, cache_dtype: str, pp_size: int, pp_rank: int
) -> None:
    """ICR-65: Eagle3 adds attention KV, but only DSA MTP adds indexer KV."""
    last = pp_rank == pp_size - 1
    # Every rank is configured with the draft; vLLM allocates it on the last.
    model = fx.deepseek_v32(
        "fp8" if cache_dtype == "fp8" else "bf16",
        layers=range(*get_pp_indices(61, pp_rank, pp_size)),
        draft=method if last else None,
    )
    md = build_metadata(
        deployment_of(
            model,
            num_layers=61,
            pp_size=pp_size,
            pp_rank=pp_rank,
            draft=Draft(
                method,
                num_layers=1,
                num_kv_heads=1,
                hf_config=_DSA_HF if method == "mtp" else SimpleNamespace(),
            ),
        )
    )

    target_layers = model.num_layers
    attention_layers = target_layers + int(last)
    indexer_layers = target_layers + int(last and method == "mtp")
    expected_aux = (
        [
            (indexer_layers, 128, torch.float8_e4m3fn),
            (indexer_layers, 1, torch.bfloat16),
        ]
        if cache_dtype == "fp8"
        else [(indexer_layers, 128, torch.bfloat16)]
    )
    assert md.kv_shape[0] == attention_layers
    assert md.kv_head_groups == [(attention_layers, 1)]
    assert md.aux_groups == expected_aux
    assert md.get_shapes(16)[1:] == [
        torch.Size([1, n, 16, w]) for n, w, _ in expected_aux
    ]

    # Register CPU stand-ins for the tensors the target and draft actually
    # allocate. Before #225 Eagle3 raised "got 61 caches" / "got 7 caches".
    conn = RBLNConnector(
        num_layers=attention_layers,
        num_kv_heads=1,
        head_dim=8,
        block_size=16,
        dtype=md.buffer_kv_dtype,
        use_mla=True,
        kv_head_groups=md.kv_head_groups,
        aux_groups=md.aux_groups,
    )
    conn.register_kv_caches(
        [
            torch.zeros(2, 16, 8, dtype=md.buffer_kv_dtype)
            for _ in range(attention_layers)
        ],
        aux_kv_caches=[
            [torch.zeros(2, 16, w, dtype=d) for _ in range(n)]
            for n, w, d in expected_aux
        ],
    )


def test_worker_takes_every_cache_dtype_from_its_kv_cache_specs() -> None:
    """vllm-rbln#1337 allocates the DSA scale cache in the model dtype and
    vllm-rbln#1338 the packed fp8 MLA latent as uint8, while the hf_config
    fallback describes the latent as int8. Chunk buffers and the connector
    follow the worker's specs; cache keys keep the int8 the scheduler, whose
    config has no per-layer spec, computes too."""
    model = fx.deepseek_v32("fp8", layers=range(1))
    md = build_metadata(deployment_of(model), packed_mla_kv=(768, torch.int8))
    assert md.kv_dtype == torch.int8
    assert md.kv_buffer_dtype == torch.uint8
    assert md.get_dtypes() == [torch.uint8, torch.float8_e4m3fn, torch.bfloat16]
    conn = RBLNConnector(
        num_layers=1,
        num_kv_heads=1,
        head_dim=768,
        block_size=16,
        dtype=md.buffer_kv_dtype,
        use_mla=True,
        aux_groups=md.aux_groups,
    )
    conn.register_kv_caches(
        [torch.zeros(2, 16, 768, dtype=torch.uint8)],
        aux_kv_caches=[
            [torch.zeros(2, 16, 128, dtype=torch.float8_e4m3fn)],
            [torch.zeros(2, 16, dtype=torch.bfloat16)],
        ],
    )


def test_main_kv_of_two_dtypes_keeps_the_key_dtype_for_buffers() -> None:
    """``RBLNConnector`` moves the main KV in one dtype; when the regions
    disagree no single allocated dtype describes it, and buffers keep
    ``kv_dtype`` as before the layout existed."""
    specs = dict(fx.dense(2, 8, torch.float16).specs)
    name, layer = fx._qwen3_layer(2, 4, torch.bfloat16)
    specs[name] = layer
    model = replace(fx.dense(2, 8, torch.float16), specs=specs)
    md = build_metadata(deployment_of(model))
    assert md.kv_dtype == torch.float16  # the fp16 model's key dtype
    assert md.kv_buffer_dtype is None


def test_worker_without_a_kv_cache_config_is_refused() -> None:
    factory = make_factory(Deployment())
    factory.cache_layout = None
    with (
        patch.object(vllm_v1_adapter, "_packed_mla_kv", return_value=None),
        pytest.raises(ValueError, match="KVCacheConfig"),
    ):
        factory.get_or_create_metadata()


# GLM-5.3's config facts as the 10-30 NPU log printed them for a worker.
GLM = Deployment(
    num_layers=4,
    num_kv_heads=1,
    head_size=576,
    dtype=torch.bfloat16,
    use_mla=True,
    hf_config=SimpleNamespace(index_topk=2048, index_head_dim=128),
    cache_dtype="fp8",
    block_size=fx.GLM_BLOCK_SIZE,
)
GLM_PACKED = (640, torch.int8)


def _glm_factory(role: str, layout: Optional[CacheLayout]) -> RBLNServiceFactory:
    config = LMCacheEngineConfig.from_defaults(chunk_size=CHUNK_SIZE)
    return RBLNServiceFactory(config, GLM.vllm_config(), role, layout)


def _glm_metadata(role: str, layout: Optional[CacheLayout]) -> RBLNLMCacheMetadata:
    factory = _glm_factory(role, layout)
    with patch.object(vllm_v1_adapter, "_packed_mla_kv", return_value=GLM_PACKED):
        metadata = factory.get_or_create_metadata()
    assert isinstance(metadata, RBLNLMCacheMetadata)
    return metadata


def _glm_connector(layout: CacheLayout) -> RBLNConnector:
    factory = _glm_factory("worker", layout)
    builder = vllm_v1_adapter.LMCacheEngineBuilder
    with (
        patch.object(vllm_v1_adapter, "_packed_mla_kv", return_value=GLM_PACKED),
        patch.object(builder, "get", return_value=None),
        patch.object(builder, "get_or_create", return_value=MagicMock()) as create,
        patch("vllm.distributed.parallel_state.get_tp_group"),
    ):
        factory.get_or_create_lmcache_engine()
    return create.call_args.args[3]


def _glm_layout(num_layers: int = 4) -> CacheLayout:
    glm = fx.glm53(num_layers)
    return CacheLayout.from_kv_cache_config(glm.kv_cache_config, glm.aux_specs)


def test_glm_worker_keys_its_block_chunks_like_the_scheduler() -> None:
    layout = _glm_layout()
    worker = _glm_metadata("worker", layout)
    scheduler = _glm_metadata("scheduler", None)
    fields = ("kv_dtype", "use_mla", "model_name", "world_size", "worker_id")
    assert {f: getattr(scheduler, f) for f in fields} == {
        f: getattr(worker, f) for f in fields
    }
    assert worker.kv_dtype == torch.int8
    # The latent at the size vLLM allocated it (512, not the config's 640),
    # then one block of every block region.
    assert worker.get_shapes() == [torch.Size([1, 1, worker.chunk_size, 512])] + [
        torch.Size([1, n, *shape]) for n, shape, _, _ in layout.block_groups
    ]
    assert worker.get_dtypes() == [torch.uint8] + [
        d for _, _, d, _ in layout.block_groups
    ]
    assert worker.get_num_groups() == 5
    # The CPU budget covers a whole chunk.
    chunk_bytes = sum(
        math.prod(shape) * dtype.itemsize
        for shape, dtype in zip(worker.get_shapes(), worker.get_dtypes(), strict=True)
    )
    kv_bytes = math.prod(worker.kv_shape) * worker.kv_dtype.itemsize
    assert chunk_bytes <= kv_bytes < chunk_bytes + 4096


def test_glm_worker_engine_moves_chunks_with_the_rbln_connector() -> None:
    layout = _glm_layout()
    connector = _glm_connector(layout)
    assert isinstance(connector, RBLNConnector)
    assert (connector.num_layers, connector.head_dim, connector.dtype) == (
        1,
        512,
        torch.uint8,
    )
    assert connector.block_groups == layout.block_groups
    assert connector.get_shapes(4096)[1:] == [
        torch.Size([1, n, *shape]) for n, shape, _, _ in layout.block_groups
    ]


def test_a_kda_only_glm_worker_builds_a_block_only_connector() -> None:
    layout = _glm_layout(3)
    md = _glm_metadata("worker", layout)
    assert md.get_shapes() == [
        torch.Size([1, n, *shape]) for n, shape, _, _ in layout.block_groups
    ]
    assert md.get_num_groups() == len(layout.block_groups) == 2
    connector = _glm_connector(layout)
    assert connector.num_layers == 0 and connector.kv_head_groups == []
    assert connector.get_shapes(4096) == md.get_shapes()

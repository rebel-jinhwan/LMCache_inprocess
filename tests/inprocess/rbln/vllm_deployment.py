# SPDX-License-Identifier: Apache-2.0
"""A vLLM deployment as ``RBLNServiceFactory`` reads it.

Factory tests describe a deployment -- model geometry, pipeline layout, KV
cache dtype, speculative draft, and on a worker the KV caches vLLM allocated
-- and check the metadata and connector the factory builds from it. The
configs carry real values instead of mocks, so the LMCache and vLLM helpers
the factory calls (``mla_enabled``, ``validate_mla_config``,
``calculate_draft_layers``, ``get_kv_cache_torch_dtype``,
``SpeculativeConfig.use_eagle``) run unpatched. An upstream refactor that
keeps their behaviour, such as moving them to another module, cannot break
these tests; one that changes it does.

A worker reads its caches from vLLM's per-layer KV cache specs. A deployment
derives plain attention specs from its geometry, or carries a real model's
from :mod:`.kv_spec_fixtures` (``deployment_of``).

Only the RBLN connector's own seams are patched: ``_packed_mla_kv``, whose answer
depends on the installed vllm-rbln and has its own tests, and the engine
builder, which would start LMCache's services.
"""

# Standard
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

# Third Party
from vllm.config import SpeculativeConfig
from vllm.distributed.utils import get_pp_indices
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MLAAttentionSpec
import torch

# First Party
from lmcache.inprocess.rbln.integration.vllm import vllm_v1_adapter
from lmcache.inprocess.rbln.integration.vllm.cache_layout import CacheLayout
from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import (
    RBLNLMCacheMetadata,
    RBLNServiceFactory,
)
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector
from lmcache.inprocess.v1.config import LMCacheEngineConfig

# Local
from . import kv_spec_fixtures as fx

CHUNK_SIZE = 256


@dataclass
class Draft:
    """A speculative draft model that shares the target's KV cache."""

    method: str = "eagle3"
    num_layers: int = 1
    num_kv_heads: int = 8
    hf_config: Any = field(default_factory=SimpleNamespace)


@dataclass
class Deployment:
    """One rank of a vLLM deployment. ``num_layers`` is the whole model; each
    pipeline stage holds the share vLLM's ``get_pp_indices`` gives it."""

    num_layers: int = 32
    num_kv_heads: int = 8
    head_size: int = 128
    dtype: torch.dtype = torch.float16
    use_mla: bool = False
    hf_config: Any = field(default_factory=SimpleNamespace)
    # Defaults to ``hf_config``; multimodal models nest the text config.
    hf_text_config: Any = None
    cache_dtype: str = "auto"
    # ``cache_config.block_size``; every cache a worker allocates has it.
    block_size: int = fx.BLOCK_SIZE
    pp_size: int = 1
    pp_rank: int = 0
    draft: Optional[Draft] = None
    model: str = "test-model"
    # The specs of the caches a worker of this rank allocates. Unset, one
    # attention cache per layer, derived from the fields above.
    kv_specs: Optional[dict[str, KVCacheSpec]] = None

    def vllm_config(self) -> SimpleNamespace:
        parallel_config = SimpleNamespace(
            rank=self.pp_rank,
            world_size=self.pp_size,
            tensor_parallel_size=1,
            pipeline_parallel_size=self.pp_size,
            data_parallel_rank_local=0,
        )
        return SimpleNamespace(
            model_config=_model_config(
                total_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                hf_config=self.hf_config,
                hf_text_config=self.hf_text_config,
                model=self.model,
                dtype=self.dtype,
                use_mla=self.use_mla,
            ),
            parallel_config=parallel_config,
            cache_config=SimpleNamespace(
                cache_dtype=self.cache_dtype, block_size=self.block_size
            ),
            speculative_config=_speculative_config(self),
            kv_transfer_config=None,
        )

    def stage(self) -> range:
        """The target layers this rank's pipeline stage holds."""
        return range(*get_pp_indices(self.num_layers, self.pp_rank, self.pp_size))

    def cache_layout(self) -> CacheLayout:
        """The layout of the caches a worker of this rank allocates."""
        specs = self.kv_specs if self.kv_specs is not None else self._layer_specs()
        # vLLM sizes every cache from ``cache_config.block_size``; a test that
        # sets one but not the other describes a deployment vLLM cannot make.
        assert {spec.block_size for spec in specs.values()} == {self.block_size}, (
            f"cache_config.block_size {self.block_size} does not match the "
            "caches' block size"
        )
        return CacheLayout.from_kv_cache_config(fx.worker_kv_cache_config(specs))

    def _layer_specs(self) -> dict[str, KVCacheSpec]:
        """One attention cache per layer of this stage, then the draft's on the
        last stage: vllm-rbln builds an EAGLE-family draft there only."""
        dtype = get_kv_cache_torch_dtype(self.cache_dtype, self.dtype)
        layers = [(i, self.num_kv_heads) for i in self.stage()]
        if self.draft is not None and self.pp_rank == self.pp_size - 1:
            layers += [
                (self.num_layers + i, self.draft.num_kv_heads)
                for i in range(self.draft.num_layers)
            ]
        specs: dict[str, KVCacheSpec] = {}
        for i, num_kv_heads in layers:
            name = f"model.layers.{i}.self_attn.attn"
            if self.use_mla:
                specs[name] = MLAAttentionSpec(
                    block_size=self.block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=self.head_size,
                    dtype=dtype,
                )
            else:
                specs[name] = FullAttentionSpec(
                    block_size=self.block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=self.head_size,
                    head_size_v=self.head_size,
                    dtype=dtype,
                )
        return specs


def deployment_of(model: fx.KVSpecFixture, **overrides: Any) -> Deployment:
    """One rank serving ``model``, whose workers allocate ``model``'s caches.

    ``model`` holds one pipeline stage's caches; with ``pp_size`` > 1, pass the
    whole model's depth as ``num_layers``.
    """
    draft = model.draft
    deployment = Deployment(
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        head_size=model.head_size,
        dtype=model.model_dtype,
        use_mla=model.use_mla,
        hf_config=model.hf_config,
        cache_dtype=model.cache_dtype,
        block_size=model.block_size,
        draft=(
            None
            if draft is None
            else Draft(
                draft.method, draft.num_layers, draft.num_kv_heads, draft.hf_config
            )
        ),
        kv_specs=model.specs,
    )
    deployment = replace(deployment, **overrides)
    assert len(deployment.stage()) == model.num_layers, (
        f"stage {deployment.pp_rank}/{deployment.pp_size} of a "
        f"{deployment.num_layers}-layer model holds {len(deployment.stage())} "
        f"layers, the fixture {model.num_layers}"
    )
    return deployment


def _layer_range(total: int, parallel_config: SimpleNamespace) -> tuple[int, int]:
    pp_size = parallel_config.pipeline_parallel_size
    stage = (parallel_config.rank // parallel_config.tensor_parallel_size) % pp_size
    return get_pp_indices(total, stage, pp_size)


def _model_config(
    *,
    total_layers: int,
    num_kv_heads: int,
    head_size: int,
    hf_config: Any,
    hf_text_config: Any = None,
    model: str = "test-model",
    dtype: torch.dtype = torch.float16,
    use_mla: bool = False,
) -> SimpleNamespace:
    def num_layers(parallel_config: SimpleNamespace) -> int:
        start, end = _layer_range(total_layers, parallel_config)
        return end - start

    return SimpleNamespace(
        model=model,
        served_model_name=model,
        dtype=dtype,
        use_mla=use_mla,
        hf_config=hf_config,
        hf_text_config=hf_config if hf_text_config is None else hf_text_config,
        get_num_layers=num_layers,
        get_layers_start_end_indices=lambda pc: _layer_range(total_layers, pc),
        get_total_num_hidden_layers=lambda: total_layers,
        get_num_kv_heads=lambda pc: num_kv_heads,
        get_head_size=lambda: head_size,
    )


def _speculative_config(deployment: Deployment) -> Optional[SimpleNamespace]:
    draft = deployment.draft
    if draft is None:
        return None
    spec = SimpleNamespace(
        method=draft.method,
        draft_model_config=_model_config(
            total_layers=draft.num_layers,
            num_kv_heads=draft.num_kv_heads,
            head_size=deployment.head_size,
            hf_config=draft.hf_config,
            dtype=deployment.dtype,
        ),
    )
    # vLLM's own predicate, so a method it adds or drops reaches the factory.
    spec.use_eagle = lambda: SpeculativeConfig.use_eagle(spec)
    return spec


def make_factory(
    deployment: Deployment,
    role: str = "worker",
    *,
    config: Optional[LMCacheEngineConfig] = None,
    cache_layout: Optional[CacheLayout] = None,
) -> RBLNServiceFactory:
    """A worker gets ``deployment``'s cache layout unless given one; the
    scheduler gets none, as vllm-rbln hands it none."""
    if config is None:
        config = LMCacheEngineConfig.from_defaults(chunk_size=CHUNK_SIZE)
    if cache_layout is None and role != "scheduler":
        cache_layout = deployment.cache_layout()
    return RBLNServiceFactory(config, deployment.vllm_config(), role, cache_layout)


def build_metadata(
    deployment: Deployment,
    role: str = "worker",
    *,
    config: Optional[LMCacheEngineConfig] = None,
    cache_layout: Optional[CacheLayout] = None,
    packed_mla_kv: Optional[tuple[int, torch.dtype]] = None,
) -> RBLNLMCacheMetadata:
    factory = make_factory(deployment, role, config=config, cache_layout=cache_layout)
    with patch.object(vllm_v1_adapter, "_packed_mla_kv", return_value=packed_mla_kv):
        metadata = factory.get_or_create_metadata()
    assert isinstance(metadata, RBLNLMCacheMetadata)
    return metadata


def build_engine(
    deployment: Deployment,
    role: str = "worker",
    *,
    config: Optional[LMCacheEngineConfig] = None,
) -> tuple[RBLNServiceFactory, Optional[RBLNConnector]]:
    """Run ``get_or_create_lmcache_engine`` and return the factory and the
    connector it hands LMCache's engine builder."""
    factory = make_factory(deployment, role, config=config)
    builder = vllm_v1_adapter.LMCacheEngineBuilder
    with (
        patch.object(vllm_v1_adapter, "_packed_mla_kv", return_value=None),
        patch.object(builder, "get", return_value=None),
        patch.object(builder, "get_or_create", return_value=MagicMock()) as create,
        patch("vllm.distributed.parallel_state.get_tp_group"),
    ):
        factory.get_or_create_lmcache_engine()
    connector = create.call_args.args[3]
    assert connector is None or isinstance(connector, RBLNConnector)
    return factory, connector

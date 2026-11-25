# SPDX-License-Identifier: Apache-2.0
"""RBLN-specific vLLM v1 adapter for LMCache.

Provides RBLNServiceFactory that injects the RBLN connector instead of
the CUDA-based GPU connector.

Usage:
    In vllm-rbln, use RBLNServiceFactory when constructing LMCacheManager:

        factory = RBLNServiceFactory(config, vllm_config, role)
        manager = LMCacheManager(config, factory, connector=impl)
"""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional
import math

# Third Party
from torch_rbln.device import physical_device_count
import torch

# First Party
from lmcache.inprocess.integration.vllm.utils import (
    ENGINE_NAME,
    calculate_draft_layers,
    mla_enabled,
    validate_mla_config,
)
from lmcache.inprocess.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.inprocess.observability import PrometheusLogger
from lmcache.inprocess.rbln._offload_server import InterruptibleZMQOffloadServer
from lmcache.inprocess.rbln.integration.vllm.cache_layout import CacheLayout
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector, sized_head_groups
from lmcache.inprocess.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.inprocess.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.config import ParallelConfig, VllmConfig

logger = init_logger(__name__)


def _rbln_calculate_local_rank_and_world_size(
    vllm_config: "VllmConfig",
) -> tuple[int, int]:
    """Calculate local rank and world size for RBLN NPU.

    Unlike the upstream version which calls torch.cuda.device_count(),
    this counts RBLN NPUs.
    """
    parallel_config = vllm_config.parallel_config
    global_world_size = parallel_config.world_size

    try:
        # ``device_count`` collapses to 1 under RSD; this wants the host's
        # actual NPU count.
        num_devices = physical_device_count()
    except Exception:
        logger.warning("RBLN NPU count unavailable; assuming 1", exc_info=True)
        num_devices = 1

    if global_world_size <= num_devices:
        return parallel_config.rank, parallel_config.world_size
    else:
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        local_world_size = global_world_size // pp_size
        assert local_world_size == tp_size, (
            "LMCache assumes TP is intra-node and PP is inter-node. "
            f"Got local_world_size={local_world_size}, tp_size={tp_size}"
        )
        local_rank = parallel_config.rank % local_world_size
        return local_rank, local_world_size


def _align_mla_rank_dedup_with_pp(
    config: LMCacheEngineConfig,
    parallel_config: "ParallelConfig",
    use_mla: bool,
) -> None:
    """Turn off MLA's cross-rank cache dedup when pipeline parallel is on.

    LMCache derives ``save_only_first_rank``, ``lookup_server_worker_ids``
    and ``remote_enable_mla_worker_id_as0`` from ``use_mla`` alone, on the
    premise that every rank holds byte-identical MLA KV -- true across TP,
    false across PP, where each stage caches only its own layers.

    Left on, ``save_only_first_rank`` makes every rank but global rank 0
    *passive*: it stores nothing and waits for rank 0 to broadcast the
    chunks over the **TP** group, which under ``pp>1, tp=1`` has one
    member -- so the passive rank retrieves 0 tokens while the scheduler's
    lookup (answered by rank 0, which did store) reported a prefix hit,
    and vLLM fails the request. See rebellions-sw/lmcache-rbln#189.
    """
    pp_size = parallel_config.pipeline_parallel_size
    if not use_mla or pp_size <= 1:
        return

    if config.extra_config is None:
        config.extra_config = {}
    config.extra_config["save_only_first_rank"] = False
    config.extra_config["remote_enable_mla_worker_id_as0"] = False
    # Stages store as each stage's forward retires, not in lockstep like TP
    # ranks, so a hit must be confirmed by every rank, not just rank 0.
    config.lookup_server_worker_ids = list(range(parallel_config.world_size))

    logger.warning(
        "MLA + pipeline_parallel_size=%d: each PP stage caches its own "
        "layers, so LMCache's MLA rank dedup is disabled "
        "(save_only_first_rank / remote_enable_mla_worker_id_as0 off, "
        "lookup on all %d ranks). See rebellions-sw/lmcache-rbln#189.",
        pp_size,
        parallel_config.world_size,
    )


_PACKED_MLA_SCALE_BYTES = 128


def _packed_mla_kv_from_hf_config(
    vllm_config: "VllmConfig",
) -> Optional[tuple[int, torch.dtype]]:
    """What a vllm-rbln that does not publish the packed layout allocates.

    Those builds pack the row all the same, so the choice is between a second
    copy of the geometry and a KV cache the engine cannot register. The dtype
    is a placeholder of the right size: builds since vllm-rbln#1338 allocate
    uint8, and a worker's buffers take the allocated dtype from its KV cache
    spec. Delete this once the build that publishes the layout is the floor.
    """
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config
    text_config = getattr(model_config, "hf_text_config", hf_config)
    is_sparse = hasattr(text_config, "index_topk") or hasattr(hf_config, "index_topk")
    cache_dtype = vllm_config.cache_config.cache_dtype
    # Not gated on `model_config.use_mla`: vLLM reads that from its own model
    # table, which an out-of-tree MLA model (`patches/axk2`) is not in, while
    # its layers pack the row like any other DSA model.
    if not (cache_dtype.startswith("fp8") and is_sparse):
        return None

    logger.warning(
        "vllm-rbln does not publish its packed sparse-MLA fp8 KV layout; "
        "describing it from a local copy of the geometry. Upgrading vllm-rbln "
        "to a build that carries `packed_sparse_mla_kv` retires this path."
    )
    return (
        text_config.kv_lora_rank
        + _PACKED_MLA_SCALE_BYTES
        + text_config.qk_rope_head_dim * 2,
        torch.int8,
    )


def _packed_mla_kv(vllm_config: "VllmConfig") -> Optional[tuple[int, torch.dtype]]:
    """`(head_size, dtype)` of vllm-rbln's packed sparse-MLA fp8 latent cache.

    This is the one layout that `cache_config.cache_dtype` and
    `model_config.get_head_size()` do not describe: the fp8 DSA kernel packs
    the latent in-kernel into a wider row, and the cache is allocated at that
    width. vllm-rbln publishes what it allocates, so take the geometry from
    there instead of deriving a second copy of it here (#184).
    """
    try:
        # Third Party
        from vllm_rbln.v1.worker.utils import packed_sparse_mla_kv
    except ImportError:
        return _packed_mla_kv_from_hf_config(vllm_config)

    return packed_sparse_mla_kv(
        vllm_config.model_config, vllm_config.cache_config.cache_dtype
    )


@dataclass
class RBLNLMCacheMetadata(LMCacheMetadata):
    """``LMCacheMetadata`` whose chunk may span several KV groups.

    ``kv_head_groups`` is ``(num_layers, num_kv_heads)`` per run of layers
    sharing one head count, in layer order. Upstream sizes a chunk from the
    single ``kv_shape`` unless a ``KVLayerGroupsManager`` is attached, and
    only its CUDA V3 / mp connectors build one, from their own layout
    discovery. The RBLN connector transfers per group itself, so the
    per-group shapes come from here; with one group every answer is the
    ``kv_shape`` fallback upstream would give. Where groups differ in head
    size (Gemma4: 256 sliding, 512 full) each entry carries its own as a
    third element, and ``kv_shape`` only sizes the CPU budget.

    ``aux_groups`` is ``(num_layers, width, dtype)`` per per-token cache that
    rides along with the attention KV in the same chunk -- the DSA indexer key
    and its scale, or the MSA indexer key (:class:`CacheLayout` aux regions).
    Each is one ``[1, L, T, width]`` region after the head groups, so the
    latent and the indexer state of a position are stored, found and evicted
    as one key. Only a worker knows them; the scheduler leaves both group
    lists empty, since nothing it runs sizes a chunk.

    ``kv_buffer_dtype`` is the dtype vLLM allocated the KV cache in, when it is
    a different dtype of the same size than ``kv_dtype`` (the packed fp8 MLA
    latent is int8 or uint8 depending on the vllm-rbln build). Chunk buffers
    and the connector use it. ``kv_dtype`` is part of every cache key, and the
    scheduler, whose KV cache config has no per-layer spec, can only compute
    the placeholder, so keys keep it on both sides.

    ``block_groups`` is ``(num_caches, block_shape, dtype, checkpoint)`` per
    cache with no row per token (:class:`CacheLayout` block regions), each one
    ``[1, N, *block_shape]`` region after the aux groups. With them, a chunk is
    one block, ``kv_head_size`` is the KV regions' head size, and ``kv_shape``
    only sizes the CPU budget: upstream reads a chunk's bytes from it.
    """

    kv_head_groups: list[tuple[int, ...]] = field(default_factory=list)
    aux_groups: list[tuple[int, int, torch.dtype]] = field(default_factory=list)
    kv_buffer_dtype: Optional[torch.dtype] = None
    block_groups: list[tuple[int, tuple[int, ...], torch.dtype, bool]] = field(
        default_factory=list
    )
    kv_head_size: Optional[int] = None

    @property
    def sized_head_groups(self) -> bool:
        """Whether the head groups carry their own head sizes: all or none."""
        return sized_head_groups(self.kv_head_groups)

    @property
    def buffer_kv_dtype(self) -> torch.dtype:
        return self.kv_buffer_dtype or self.kv_dtype

    def _single_kv_shape(self) -> bool:
        """Whether upstream's ``kv_shape`` alone describes the head groups."""
        return len(self.kv_head_groups) <= 1 and self.kv_head_size is None

    def _head_shapes(self, num_tokens: Optional[int]) -> list[torch.Size]:
        if self._single_kv_shape():
            return super().get_shapes(num_tokens)
        if num_tokens is None:
            num_tokens = self.chunk_size
        kv_size, head_size = self.kv_shape[1], self.kv_shape[4]
        sized = self.sized_head_groups
        if self.kv_head_size is not None or sized:
            # kv_shape only sizes the CPU budget.
            kv_size = 1 if self.use_mla else 2
            head_size = self.kv_head_size or 0
        return [
            torch.Size(
                [kv_size, g[0], num_tokens, g[1] * (g[2] if sized else head_size)]
            )
            for g in self.kv_head_groups
        ]

    def get_shapes(self, num_tokens: Optional[int] = None) -> list[torch.Size]:
        shapes = self._head_shapes(num_tokens)
        if num_tokens is None:
            num_tokens = self.chunk_size
        return (
            shapes
            + [torch.Size([1, n, num_tokens, w]) for n, w, _ in self.aux_groups]
            + [torch.Size([1, n, *shape]) for n, shape, _, _ in self.block_groups]
        )

    def get_dtypes(self) -> list[torch.dtype]:
        if self._single_kv_shape():
            dtypes = super().get_dtypes()
        else:
            dtypes = [self.kv_dtype] * len(self.kv_head_groups)
        if self.kv_buffer_dtype is not None:
            dtypes = [self.kv_buffer_dtype] * len(dtypes)
        return (
            dtypes
            + [d for _, _, d in self.aux_groups]
            + [d for _, _, d, _ in self.block_groups]
        )

    def get_num_groups(self) -> int:
        if self._single_kv_shape() and not self.aux_groups:
            return super().get_num_groups()
        heads = len(self.kv_head_groups)
        if self.kv_head_size is None:
            heads = max(heads, 1)
        return heads + len(self.aux_groups) + len(self.block_groups)


class RBLNServiceFactory(VllmServiceFactory):
    """RBLN-specific service factory for LMCache.

    Overrides the upstream VllmServiceFactory to:
    - Use RBLN NPU rank calculation instead of CUDA device counting
    - Create an :class:`RBLNConnector` instead of the CUDA-based GPU
      connector
    - Key the offload server's IPC socket by global rank instead of TP rank
      and let it stop while idle before CPU pools unregister
    """

    # The parent class assigns this in __init__ but lacks an explicit type
    # annotation, so mypy can't infer it here. Redeclare for type checking.
    metadata: Optional[RBLNLMCacheMetadata]

    def __init__(
        self,
        lmcache_config: LMCacheEngineConfig,
        vllm_config: "VllmConfig",
        role: str,
        cache_layout: Optional[CacheLayout] = None,
    ):
        super().__init__(lmcache_config, vllm_config, role)
        self.cache_layout = cache_layout

    def _local_rank_and_world_size(self) -> tuple[int, int]:
        # Use RBLN-specific rank calculation instead of CUDA-based one
        if self.role == "scheduler":
            parallel_config = self.vllm_config.parallel_config
            return parallel_config.rank, parallel_config.world_size
        return _rbln_calculate_local_rank_and_world_size(self.vllm_config)

    def _transfer_ids(self) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        kv_transfer_config = getattr(self.vllm_config, "kv_transfer_config", None)
        if kv_transfer_config is None:
            return None, None
        return (
            getattr(kv_transfer_config, "engine_id", None),
            getattr(kv_transfer_config, "kv_connector_extra_config", None),
        )

    def get_or_create_metadata(self) -> Optional[LMCacheMetadata]:
        if self.metadata is not None:
            return self.metadata

        # Third Party
        from vllm.utils.torch_utils import get_kv_cache_torch_dtype

        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config
        cache_config = self.vllm_config.cache_config

        kv_dtype = get_kv_cache_torch_dtype(
            cache_config.cache_dtype, model_config.dtype
        )

        use_mla = mla_enabled(model_config)
        validate_mla_config(self.lmcache_config, use_mla)
        _align_mla_rank_dedup_with_pp(self.lmcache_config, parallel_config, use_mla)

        num_layer = model_config.get_num_layers(parallel_config)
        # The EAGLE-family draft head implements no `SupportsPP` forward, so
        # vllm-rbln gives the draft `pipeline_parallel_size=1` and builds it only
        # under `get_pp_group().is_last_rank`. Upstream's helper splits the draft
        # over the *target*'s pipeline size instead, which for a one-layer draft
        # over four stages lands it on stage 2 -- a stage that holds none of it.
        #
        # "mtp" takes the same proposer, so its layers sit on the last stage
        # too: DeepSeek-V3.2 at pp=8 (61 layers, split 7,7,8,8,8,8,8,7) is
        # counted 9 on stage 6 and 7 on stage 7, where each registers 8. The
        # depth below is the one upstream reads at pp=1, where it already
        # matches what vLLM allocates -- vLLM's MTP arch-config converters report
        # `num_nextn_predict_layers` as the draft's layer count -- so moving
        # it to the last stage changes where it is counted, not how much.
        num_draft_layers = calculate_draft_layers(self.vllm_config)
        speculative_config = self.vllm_config.speculative_config
        if (
            speculative_config is not None
            and speculative_config.method in ("eagle", "eagle3", "mtp", "dflash")
            and parallel_config.pipeline_parallel_size > 1
        ):
            # `get_pp_group()` is not available here -- the metadata is built in
            # EngineCore, before the distributed groups exist. Derive the stage
            # the same way `get_layers_start_end_indices` does, so this term and
            # the target's layer count above agree on which rank they describe.
            pp_rank = (
                parallel_config.rank // parallel_config.tensor_parallel_size
            ) % parallel_config.pipeline_parallel_size
            num_draft_layers = (
                speculative_config.draft_model_config.get_total_num_hidden_layers()
                if pp_rank == parallel_config.pipeline_parallel_size - 1
                else 0
            )
        chunk_size = self.lmcache_config.chunk_size
        num_kv_head = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        num_layer += num_draft_layers
        packed_mla_kv = _packed_mla_kv(self.vllm_config)
        if packed_mla_kv is not None:
            # Every CPU/RDS buffer is sized from these two values, and the
            # connector registers against the tensors vLLM allocated. Describing
            # the upstream latent here would disagree with both.
            head_size, kv_dtype = packed_mla_kv
        engine_id, kv_connector_extra_config = self._transfer_ids()
        local_worker_id, local_world_size = self._local_rank_and_world_size()
        kv_head_groups: list[tuple[int, ...]] = []
        aux_groups: list[tuple[int, int, torch.dtype]] = []
        block_groups: list[tuple[int, tuple[int, ...], torch.dtype, bool]] = []
        kv_head_size = None
        kv_buffer_dtype = None
        if self.role != "scheduler":
            if self.cache_layout is None:
                raise ValueError(
                    "an RBLN LMCache worker sizes its chunk from vLLM's "
                    "KVCacheConfig, and the connector was given none"
                )
            kv_head_groups = self.cache_layout.kv_head_groups
            aux_groups = self.cache_layout.aux_groups
            block_groups = self.cache_layout.block_groups
            allocated = {r.dtype for r in self.cache_layout.kv_regions}
            if len(allocated) == 1:
                (allocated_kv,) = allocated
                if (
                    allocated_kv != kv_dtype
                    and allocated_kv.itemsize == kv_dtype.itemsize
                ):
                    kv_buffer_dtype = allocated_kv
        kv_shape: tuple[int, int, int, int, int] = (
            num_layer,
            1 if use_mla else 2,
            chunk_size,
            num_kv_head,
            head_size,
        )
        sized_groups = sized_head_groups(kv_head_groups)
        if block_groups or sized_groups:
            assert self.cache_layout is not None
            # A model with block regions allocates its KV regions at their own
            # head size (GLM-5.3's 512-byte latent, not the config's 640), and
            # its chunk is mostly block regions; head groups of several head
            # sizes (Gemma4) carry their own. kv_shape then only sizes the CPU
            # budget, which upstream reads as prod(kv_shape) * itemsize.
            # A KDA-only model has no KV region, so no head size to read.
            if not sized_groups:
                head_sizes = {r.head_size or 0 for r in self.cache_layout.kv_regions}
                kv_head_size = head_sizes.pop() if head_sizes else 0
            probe = RBLNLMCacheMetadata(
                model_name=model_config.model,
                world_size=parallel_config.world_size,
                local_world_size=local_world_size,
                worker_id=parallel_config.rank,
                local_worker_id=local_worker_id,
                kv_dtype=kv_dtype,
                kv_shape=kv_shape,
                use_mla=use_mla,
                chunk_size=chunk_size,
                kv_head_groups=kv_head_groups,
                aux_groups=aux_groups,
                kv_buffer_dtype=kv_buffer_dtype,
                block_groups=block_groups,
                kv_head_size=kv_head_size,
            )
            chunk_bytes = sum(
                math.prod(shape) * dtype.itemsize
                for shape, dtype in zip(
                    probe.get_shapes(), probe.get_dtypes(), strict=True
                )
            )
            width = math.ceil(chunk_bytes / (chunk_size * kv_dtype.itemsize))
            kv_shape = (1, 1, chunk_size, 1, width)

        self.metadata = RBLNLMCacheMetadata(
            model_name=model_config.model,
            world_size=parallel_config.world_size,
            local_world_size=local_world_size,
            worker_id=parallel_config.rank,
            local_worker_id=local_worker_id,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=use_mla,
            role=self.role,
            served_model_name=model_config.served_model_name,
            chunk_size=self.lmcache_config.chunk_size,
            engine_id=engine_id,
            kv_connector_extra_config=kv_connector_extra_config,
            kv_head_groups=kv_head_groups,
            aux_groups=aux_groups,
            kv_buffer_dtype=kv_buffer_dtype,
            block_groups=block_groups,
            kv_head_size=kv_head_size,
        )
        logger.info(
            "RBLN LMCache metadata: num_layer=%d, chunk_size=%d, "
            "num_kv_head=%d, head_size=%d, hidden_dim=%d, use_mla=%s, "
            "kv_shape=%s, kv_head_groups=%s, aux_groups=%s, block_groups=%s, "
            "kv_head_size=%s, num_draft_layers=%d, "
            "worker_id=%d, local_worker_id=%d, world=%d, role=%s",
            num_layer,
            chunk_size,
            num_kv_head,
            head_size,
            num_kv_head * head_size,
            use_mla,
            kv_shape,
            kv_head_groups,
            aux_groups,
            block_groups,
            kv_head_size,
            num_draft_layers,
            parallel_config.rank,
            local_worker_id,
            parallel_config.world_size,
            self.role,
        )
        return self.metadata

    def get_or_create_lmcache_engine(self) -> Optional[LMCacheEngine]:
        self._ensure_metadata()
        assert self.metadata is not None

        # Scheduler without bypass lookup does not need an engine
        if (
            self.role == "scheduler"
            and not self.lmcache_config.enable_scheduler_bypass_lookup
        ):
            PrometheusLogger.GetOrCreate(
                self.metadata,
                config=self.lmcache_config,
            )
            return None

        if curr_engine := LMCacheEngineBuilder.get(ENGINE_NAME):
            self.lmcache_engine = curr_engine
            return curr_engine

        rbln_connector: Optional[RBLNConnector] = None
        if self.role == "scheduler":
            tpg = SimpleNamespace()
            tpg.broadcast = lambda tensor, src: tensor
            tpg.broadcast_object = lambda obj, src: obj
        else:
            # Third Party
            from vllm.distributed.parallel_state import get_tp_group

            tpg = get_tp_group()

        if self.role != "scheduler":
            assert self.metadata is not None

            # The RBLN connector exposes the ``GPUConnectorInterface``
            # surface LMCache expects.
            num_layer = self.metadata.kv_shape[0]
            num_kv_head = self.metadata.kv_shape[3]
            head_size = self.metadata.kv_shape[4]
            groups = self.metadata.kv_head_groups
            if (
                self.metadata.kv_head_size is not None
                or self.metadata.sized_head_groups
            ):
                # kv_shape sizes the CPU budget here; the layout has the rest.
                num_layer = sum(g[0] for g in groups)
                num_kv_head = groups[0][1] if groups else 1
                head_size = (
                    groups[0][2]
                    if self.metadata.sized_head_groups
                    else self.metadata.kv_head_size
                )
            rbln_connector = RBLNConnector(
                num_layers=num_layer,
                num_kv_heads=num_kv_head,
                head_dim=head_size,
                block_size=self.vllm_config.cache_config.block_size,
                dtype=self.metadata.buffer_kv_dtype,
                use_mla=self.metadata.use_mla,
                kv_head_groups=self.metadata.kv_head_groups,
                aux_groups=self.metadata.aux_groups,
                # Blending and PD force unfull chunks, which break the grid.
                slot_registration=not self.lmcache_config.save_unfull_chunk,
                block_groups=self.metadata.block_groups,
            )

            logger.info(
                "Created %s: layers=%d, kv_heads=%d, head_dim=%d, block_size=%d, "
                "kv_head_groups=%s, aux_groups=%s, block_groups=%s",
                type(rbln_connector).__name__,
                num_layer,
                num_kv_head,
                head_size,
                self.vllm_config.cache_config.block_size,
                self.metadata.kv_head_groups,
                self.metadata.aux_groups,
                self.metadata.block_groups,
            )

        engine = LMCacheEngineBuilder.get_or_create(
            ENGINE_NAME,
            self.lmcache_config,
            self.metadata,
            rbln_connector,
            tpg.broadcast,
            tpg.broadcast_object,
        )
        self.lmcache_engine = engine

        if (
            self.role == "scheduler"
            and self.lmcache_config.enable_scheduler_bypass_lookup
        ):
            assert engine.save_only_first_rank or (
                self.lmcache_config.get_extra_config_value(
                    "remote_enable_mla_worker_id_as0", self.metadata.use_mla
                )
            ), (
                "enable_scheduler_bypass_lookup is only supported with "
                "save_only_first_rank or remote_enable_mla_worker_id_as0"
            )

        return engine

    def maybe_create_offload_server(self) -> Optional[ZMQOffloadServer]:
        """Bind the offload server on a per-rank IPC socket.

        Upstream builds the path as
        ``engine_<id>_service_offload_lmcache_rpc_port_{100 + tp_rank}`` with
        ``get_tensor_model_parallel_rank()``, assuming PP ranks live on
        different hosts. On RBLN a PP group runs as several worker processes on
        one host with the same ``VLLM_RPC_BASE_PATH`` and ``engine_id``, so with
        TP=1 every PP rank binds the identical ``..._rpc_port_100`` path. libzmq
        unlinks the existing file before ``bind``: usually the last binder
        silently takes the path and the others hold orphaned sockets, but when
        two ranks interleave between unlink and bind the loser gets
        ``EADDRINUSE`` and its LMCacheEngine is marked init-failed (seen as
        ``[0, N, N, N]`` lookup results across ranks and zero LMCache hits).
        Reproduced in 13/300 runs of four processes binding one path (#153).

        Use ``metadata.worker_id`` -- the global rank, the same key the lookup
        server already uses -- so every rank gets its own path. Nothing in
        LMCache / vLLM / vllm-rbln connects to this socket, so no client has
        to follow the rename.

        The server is :class:`InterruptibleZMQOffloadServer`, so an idle
        server stops without using up vLLM's shutdown deadline before CPU
        pools unregister.

        TEMPORARY: this belongs upstream. Delete once the ``lmcache`` pin moves
        to a release whose offload server keys the socket by global rank and
        stops its request thread before closing the cache engine.
        """
        # Only worker needs offload server
        if self.role != "worker":
            return None

        self._ensure_engine()
        assert self.lmcache_engine is not None
        assert self.metadata is not None
        return InterruptibleZMQOffloadServer(
            self.lmcache_engine, self.metadata.worker_id
        )

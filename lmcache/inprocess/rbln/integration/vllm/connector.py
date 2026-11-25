# SPDX-License-Identifier: Apache-2.0
"""RBLN KV connector for vLLM.

Provides RBLNLMCacheConnectorV1 as a drop-in replacement for the upstream
LMCacheConnectorV1Dynamic.  In vLLM-RBLN configure it via:

    kv_connector = "RBLNLMCacheConnectorV1"
    kv_connector_module_path = "lmcache.inprocess.rbln.integration.vllm.connector"

This connector uses :class:`RBLNServiceFactory` (which injects
:class:`RBLNConnector`) instead of the upstream CUDA-only factory.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
import math
import os
import time

# Third Party
from typing_extensions import override
from vllm.config import VllmConfig
from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheKVEvents,
)
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.outputs import KVConnectorOutput
import torch

# First Party
from lmcache.inprocess.integration.vllm.lmcache_connector_v1 import (
    LMCacheConnectorV1Dynamic,
)
from lmcache.inprocess.integration.vllm.utils import lmcache_get_or_create_config
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
)
from lmcache.inprocess.rbln.integration.vllm.cache_layout import (
    BlockAuxSpec,
    CacheLayout,
    is_checkpoint_spec,
)
from lmcache.inprocess.rbln.integration.vllm.kv_events import RBLNKVEvents
from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import RBLNServiceFactory
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector
from lmcache.inprocess.rbln.v1.rbln_connector.kv_ops import as_kv_first
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)
from lmcache.inprocess.v1.manager import LMCacheManager
from lmcache.logging import init_logger
from lmcache.utils import get_version

logger = init_logger(__name__)


def _get_async_lookup_client(client: Any) -> Optional[LMCacheAsyncLookupClient]:
    while client is not None and "actual_lookup_client" in vars(client):
        client = client.actual_lookup_client
    return client if isinstance(client, LMCacheAsyncLookupClient) else None


class _ResetSafeLookupClient:
    """Keep pending replies observable after a timeout request is allocated.

    LMCache's timeout is a miss, not a completed prefetch. Its allocation
    callback clears lookup status, which would also make aborted-lookup
    cleanup mistake the missing response slot for a completed task.

    Polling such a request again (on resumption) needs nothing here: since
    0.5.7 upstream ``lookup_cache`` answers 0 when the timer is gone.
    """

    def __init__(self, client: Any, actual: LMCacheAsyncLookupClient) -> None:
        self.actual_lookup_client = client
        self._async_client = actual

    def __getattr__(self, name: str) -> Any:
        return getattr(self.actual_lookup_client, name)

    def clear_lookup_status(self, lookup_id: str) -> None:
        with self._async_client.lock:
            if self._async_client.reqs_status.get(lookup_id, -1) is None:
                return
        self.actual_lookup_client.clear_lookup_status(lookup_id)


def check_block_engine(
    vllm_config: VllmConfig, chunk_size: int, save_decode_cache: bool
) -> None:
    """Refuse engine settings a chunk with block regions cannot serve.

    A checkpoint group (``MambaSpec``) holds a state only at block ends that
    prefix caching keeps, so a chunk must be one such block. With prefix
    caching off a request holds one running state, the state at its last
    step's end. vllm-rbln runs a prefill alone, in steps cut at the token
    budget, the longest ``prefill_chunk_size`` and
    ``long_prefill_token_threshold``. When that step divides the block size
    every chunk end is a step end, and the chunk is stored in that step.

    Speculative decoding is served while only prompt chunks are stored. A
    verify step stores its first token's state (vllm-rbln replays the accepted
    drafts next step), so a block a decode step crosses holds no state at its
    end; a prefill step has no drafts. The save path stops at the prompt's last
    chunk end, and ``save_decode_cache``, which asks for more, is refused.

    Pipeline parallelism is served: each stage registers, stores and
    restores only its own layers' caches under its own worker id, a lookup
    hit is confirmed by every rank (all ranks answer lookups when MLA dedup
    is off under PP, and by default without MLA), and a stage's load errors
    reach the scheduler through vLLM's KV output aggregation, which unions
    them over all workers. Tensor parallelism is not: a TP rank holds a
    head slice of each KDA state, whose layout vllm-rbln has not settled.
    """
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    problems = []
    if not cache.enable_prefix_caching:
        step = _prefill_step(vllm_config)
        if cache.block_size % step:
            problems.append(
                f"prefix caching is off and the prefill step {step} does not "
                f"divide block_size {cache.block_size}, so no step ends at a "
                "chunk end"
            )
    elif cache.mamba_block_size != cache.block_size:
        problems.append(
            f"mamba_block_size {cache.mamba_block_size} != block_size "
            f"{cache.block_size}"
        )
    if parallel.tensor_parallel_size != 1:
        problems.append(f"tensor_parallel_size is {parallel.tensor_parallel_size}")
    if vllm_config.speculative_config is not None and save_decode_cache:
        problems.append("speculative decoding is on with save_decode_cache")
    transfer = vllm_config.kv_transfer_config
    if transfer is not None and transfer.kv_role == "kv_producer":
        # A producer may resend chunks below save_spec.skip_leading_tokens,
        # where the null-block check does not look.
        problems.append("kv_role is kv_producer")
    if chunk_size != cache.block_size:
        problems.append(
            f"LMCACHE_CHUNK_SIZE {chunk_size} != block_size {cache.block_size}"
        )
    if problems:
        raise ValueError(
            "RBLN LMCache moves checkpointed (MambaSpec) KV cache groups only "
            "with mamba_block_size == block_size (prefix caching on) or "
            "the prefill step dividing it (off), LMCACHE_CHUNK_SIZE == "
            "block_size, TP1, no decode saves under speculative decoding and "
            "no kv_producer role: " + "; ".join(problems)
        )


def _prefill_step(vllm_config: VllmConfig) -> int:
    """The tokens vllm-rbln's scheduler gives a prefill step short of its end."""
    scheduler = vllm_config.scheduler_config
    caps = [scheduler.max_num_scheduled_tokens or scheduler.max_num_batched_tokens]
    chunks = getattr(vllm_config.additional_config, "prefill_chunk_size", None)
    if chunks:
        caps.append(max(chunks))
    if scheduler.long_prefill_token_threshold > 0:
        caps.append(scheduler.long_prefill_token_threshold)
    return min(caps)


def block_aux_specs(vllm_config: VllmConfig) -> dict[str, BlockAuxSpec]:
    """vllm-rbln's block-aux caches, by name, from the model's attention modules.

    They belong to no KV cache group, so only the modules describe them.
    """
    specs = {}
    for name, module in vllm_config.compilation_config.static_forward_context.items():
        get_spec = getattr(module, "get_aux_cache_spec", None)
        if get_spec is None:
            continue
        spec = get_spec(vllm_config)
        if not hasattr(spec, "prefix_cacheable"):
            raise ValueError(
                f"{name}: this vllm-rbln's block-aux spec has no prefix_cacheable, "
                "so which caches a prefix needs is unknown"
            )
        specs[name] = spec
    return specs


@dataclass
class RBLNLMCacheConnectorMetadata(LMCacheConnectorMetadata):
    reset_req_ids: set[str] = field(default_factory=set)


class _LMCacheClearDone(KVConnectorWorkerMetadata):
    """Worker -> scheduler: this step ran the ``reset_cache`` clear.

    That is ``LMCacheEngine.clear()`` over every storage backend (CPU, disk,
    RDS, ...). Workers apply it when they bind the step's metadata, so once the
    step's outputs are aggregated every rank has cleared; one report is enough,
    and each flagged step yields exactly one.
    """

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        return self


class RBLNLMCacheConnectorV1Impl(LMCacheConnectorV1Impl):
    """RBLN-specific LMCache connector implementation.

    Replaces the upstream CUDA-bound factory with :class:`RBLNServiceFactory`.
    Skips the upstream ``__init__`` entirely because that path calls
    ``get_vllm_torch_dev`` which fails on RBLN.
    """

    _has_checkpoints = False
    _save_prompt_only = False
    _running_state_groups: frozenset[int] = frozenset()

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        # RBLN compiles the full forward graph, so vLLM never invokes
        # per-layer save_kv_layer / wait_for_layer_load hooks. With
        # use_layerwise=True, all KV transfers are silently skipped
        # (no error, just empty cache). Reject loud at init.
        if getattr(config, "use_layerwise", False):
            raise ValueError(
                "RBLN does not support LMCACHE_USE_LAYERWISE=True: "
                "per-layer hooks are never invoked under full-graph "
                "compile, so KV transfers would be silently dropped. "
                "Unset LMCACHE_USE_LAYERWISE or set it to False."
            )
        self.config = config
        self._request_blocks: dict[str, KVCacheBlocks] = {}
        self._failed_load_req_ids: set[str] = set()
        # Requests whose reclaimed window sits past LMCache's contiguous
        # prefix; nothing they save could be looked up, so they stop saving.
        self._save_blocked_req_ids: set[str] = set()
        kv_cache_config = parent._kv_cache_config
        self._kv_cache_groups = (
            kv_cache_config.kv_cache_groups if kv_cache_config is not None else []
        )
        # A checkpoint group keeps a block only at block ends a request may
        # resume from. vLLM does not collapse MambaSpec groups for the
        # scheduler, so both roles see this.
        self._has_checkpoints = any(
            is_checkpoint_spec(group.kv_cache_spec) for group in self._kv_cache_groups
        )
        # With prefix caching off ("none") such a group holds one running
        # state a request instead.
        self._running_state_groups = frozenset(
            g
            for g, group in enumerate(self._kv_cache_groups)
            if is_checkpoint_spec(group.kv_cache_spec)
            and group.kv_cache_spec.mamba_cache_mode == "none"
        )
        # The block-end alignment and the per-group save checks live on the
        # grouped path, so a lone checkpoint group takes it too.
        self._use_grouped_slots = (
            self._has_checkpoints
            or len(self._kv_cache_groups) > 1
            or any(
                isinstance(group.kv_cache_spec, SlidingWindowSpec)
                for group in self._kv_cache_groups
            )
        )
        if self._has_checkpoints:
            check_block_engine(vllm_config, config.chunk_size, config.save_decode_cache)
            self._save_prompt_only = vllm_config.speculative_config is not None
        elif self._use_grouped_slots:
            parallel = vllm_config.parallel_config
            if (
                parallel.tensor_parallel_size != 1
                or parallel.pipeline_parallel_size != 1
            ):
                raise ValueError(
                    "RBLN LMCache HMA currently requires tensor_parallel_size=1 "
                    "and pipeline_parallel_size=1: KV load failures are not "
                    "coordinated across model-parallel workers."
                )
            for group in self._kv_cache_groups:
                spec = group.kv_cache_spec
                # Subclasses can store state rather than per-token KV (the
                # older RBLN shifting-window spec is one such layout).
                if type(spec) not in (FullAttentionSpec, SlidingWindowSpec):
                    raise ValueError(
                        "RBLN LMCache HMA requires paged full/sliding attention, "
                        f"got {type(spec).__name__}"
                    )
                # vLLM grows a smaller page's block to a multiple (Gemma4's
                # full layers: twice the sliding block). Slots address each
                # cache by its own kernel block, so only the chunk grid, one
                # cache block, must tile every group's block.
                if spec.block_size % vllm_config.cache_config.block_size:
                    raise ValueError(
                        "RBLN LMCache HMA requires every manager block size to "
                        "be a multiple of the cache block size"
                    )
                if spec.block_size % config.chunk_size:
                    raise ValueError(
                        "LMCache chunk_size must divide the manager block size "
                        "for complete-block HMA transfers"
                    )
            # Upstream's request tracker counts group 0's blocks at the cache
            # block size; a grown group must not come first.
            if self._kv_cache_groups[0].kv_cache_spec.block_size != (
                vllm_config.cache_config.block_size
            ):
                raise ValueError(
                    "RBLN LMCache HMA requires the first KV cache group to have "
                    "the cache block size"
                )

        # The scheduler's KVCacheConfig collapses each group to one layer's
        # spec, so only a worker can read the chunk layout from it.
        self._cache_layout: Optional[CacheLayout] = None
        self._block_aux_specs: dict[str, BlockAuxSpec] = {}
        if role == KVConnectorRole.WORKER and kv_cache_config is not None:
            self._block_aux_specs = block_aux_specs(vllm_config)
            self._cache_layout = CacheLayout.from_kv_cache_config(
                kv_cache_config, self._block_aux_specs
            )
            # The scheduler aligns hits to block ends and stops a store at a
            # null block only for checkpoint groups, which it can see.
            if self._cache_layout.block_regions and not self._has_checkpoints:
                raise ValueError(
                    "RBLN LMCache moves block regions (a cache with no row per "
                    "token, or a block-aux cache) only beside a MambaSpec "
                    "checkpoint group; got "
                    f"{[r.block_shape for r in self._cache_layout.block_regions]}"
                )
        service_factory = RBLNServiceFactory(
            config,
            vllm_config,
            role.name.lower(),
            self._cache_layout,
        )
        self._manager = LMCacheManager(
            config,
            service_factory,
            connector=self,
        )
        self._manager.start_services()
        self._init_connector_state(role, vllm_config, config)
        if self._use_grouped_slots:
            # Snapshot and reuse complete cache blocks so a restored page
            # comes from a single completed prefill block, including its
            # aliased Full/SWA views. A group of a longer block (Gemma4's full
            # layers) is restored a cache block at a time, as chunked prefill
            # fills it.
            # The storage engine still uses config.chunk_size-sized objects.
            self._lmcache_chunk_size = self._block_size
            self._discard_partial_chunks = True
        self._setup_metrics()

        # Third Party
        from vllm.version import __version__ as VLLM_VERSION

        logger.info(
            "RBLN LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    @property
    @override
    def lookup_client(self) -> Any:
        client = super().lookup_client
        actual = _get_async_lookup_client(client)
        if actual is None:
            return client
        wrapped = getattr(self, "_reset_safe_lookup_client", None)
        if wrapped is None or wrapped.actual_lookup_client is not client:
            wrapped = _ResetSafeLookupClient(client, actual)
            self._reset_safe_lookup_client = wrapped
        return wrapped

    @override
    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> Optional[int]:
        """[PATHPROBE experiment] Log the resolved cache path per request.

        ``num_computed_tokens`` is vLLM's device prefix-cache hit count;
        the return value is the extra tokens served from the external
        (RDS) backend.  The LMCache logger prefixes each line with the DP
        rank (``EngineCore_DPx``), so this pins down — per request and per
        rank — whether the path was COLD / PREFIX_HIT / RDS_RESTORE / MIXED
        instead of inferring it from aggregate metrics.
        """
        if self._use_grouped_slots and request.skip_reading_prefix_cache:
            # A resumed request still needs a load spec for tracker rebuilding.
            self.load_specs[request.request_id] = LoadSpec(
                num_computed_tokens, 0, False
            )
            return 0
        external = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if self._use_grouped_slots and external is not None:
            load_spec = self.load_specs.get(request.request_id)
            if load_spec is not None:
                # Eviction or a capped lookup can leave only part of a page.
                # Keep the last-token recompute rule for a full prompt hit.
                load_spec.lmcache_cached_tokens = (
                    load_spec.lmcache_cached_tokens
                    // self._block_size
                    * self._block_size
                )
                reusable = min(load_spec.lmcache_cached_tokens, request.num_tokens - 1)
                if self._has_checkpoints:
                    # A checkpoint exists only at a block end: resume there.
                    reusable = reusable // self._block_size * self._block_size
                    load_spec.lmcache_cached_tokens = reusable
                external = min(external, max(0, reusable - num_computed_tokens))
        ext = external or 0
        if num_computed_tokens > 0 and ext > 0:
            path = "MIXED(prefix+rds)"
        elif num_computed_tokens > 0:
            path = "PREFIX_HIT"
        elif ext > 0:
            path = "RDS_RESTORE"
        else:
            path = "COLD"
        logger.info(
            "[PATHPROBE] req=%s prefix_hit_tok=%d rds_ext_tok=%d -> %s",
            getattr(request, "request_id", "?"),
            num_computed_tokens,
            ext,
            path,
        )
        return external

    def update_state_after_alloc(
        self,
        request: Any,
        num_external_tokens: int,
        blocks: Optional[KVCacheBlocks] = None,
    ) -> None:
        super().update_state_after_alloc(request, num_external_tokens)
        if self._use_grouped_slots:
            assert blocks is not None
            # KVCacheBlocks retains the manager-owned lists: append and null
            # replacement must be observed again on every scheduled step.
            self._request_blocks[request.request_id] = blocks

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        if not self._use_grouped_slots:
            return metadata
        metadata = RBLNLMCacheConnectorMetadata(
            requests=metadata.requests,
            reset_req_ids=(
                scheduler_output.finished_req_ids | scheduler_output.preempted_req_ids
            ),
        )
        for request_id in scheduler_output.finished_req_ids:
            self._request_blocks.pop(request_id, None)
        self._save_blocked_req_ids.difference_update(metadata.reset_req_ids)
        for request in metadata.requests:
            groups = self._request_blocks[request.req_id].blocks
            num_tokens = len(request.token_ids)
            slots = []
            for g, (group, blocks) in enumerate(
                zip(self._kv_cache_groups, groups, strict=True)
            ):
                if g in self._running_state_groups:
                    slots.append(
                        self._running_state_slots(request, blocks[0], scheduler_output)
                    )
                    continue
                block_size = group.kv_cache_spec.block_size
                block_ids = torch.tensor(
                    [block.block_id if not block.is_null else -1 for block in blocks],
                    dtype=torch.long,
                )
                group_slots = block_ids[:, None] * block_size + torch.arange(block_size)
                group_slots[block_ids < 0] = -1
                slots.append(group_slots.flatten()[:num_tokens])
            request.slot_mapping = torch.stack(slots, dim=1)
            if request.save_spec is None or not request.save_spec.can_save:
                continue
            if request.req_id in self._save_blocked_req_ids:
                request.save_spec.can_save = False
                continue
            if self._has_checkpoints:
                self._stop_save_at_null_block(request)
                continue
            # Local prefix hits may have already reclaimed SWA blocks. Never
            # publish chunks containing those slots as valid external KV.
            missing = (request.slot_mapping < 0).any(dim=1).nonzero()
            if not missing.numel():
                continue
            available_start = int(missing[-1].item()) + 1
            chunk = self._lmcache_chunk_size
            available_start = (available_start + chunk - 1) // chunk * chunk
            if available_start <= request.save_spec.skip_leading_tokens:
                continue
            # Never hand reclaimed slots to the store. A producer's save is a
            # transfer that upstream runs whatever can_save says, reading
            # skip_leading_tokens as-is, so it resumes at the first complete
            # chunk, as before.
            request.save_spec.skip_leading_tokens = available_start
            if self.kv_role == "kv_producer":
                continue
            # Otherwise LMCache holds a contiguous prefix only up to the old
            # skip_leading_tokens, this request cannot refill the reclaimed
            # window, and lookups stop at the first missing chunk. Nothing
            # saved past the gap could be hit, so stop saving for the
            # request's lifetime. Upstream's tracker advances anyway, so a
            # one-off flip would let the next chunk boundary resume; a
            # preemption re-looks up and starts over.
            self._save_blocked_req_ids.add(request.req_id)
            request.save_spec.can_save = False
        return metadata

    def _running_state_slots(
        self, request: Any, block: KVCacheBlock, scheduler_output: Any
    ) -> torch.Tensor:
        """Slots of a request's one running state: two chunks' rows, else null.

        The state is a chunk's checkpoint only right after the step ending at
        the chunk's end, so that chunk alone may be saved; a load needs only
        the state of the last chunk it restores. Their rows address the block,
        at the block size the worker divides slots by. Every other row is
        null, so a save stops there as at a null checkpoint block, and a load
        skips it whatever order the chunks arrive in.
        """
        req_id = request.req_id
        # The step's end on the device. LMCache's tracker trails it by a token
        # in an async decode step, whose sampled token is not in yet.
        ends = [
            self._unfinished_requests[req_id].num_computed_tokens
            + scheduler_output.num_scheduled_tokens[req_id]
        ]
        load = request.load_spec
        if load is not None and load.can_load:
            ends.append(load.lmcache_cached_tokens)
        chunk = self._lmcache_chunk_size
        t = torch.arange(len(request.token_ids))
        kept = torch.zeros_like(t, dtype=torch.bool)
        for end in ends:
            if end % chunk == 0:
                kept |= (end - chunk <= t) & (t < end)
        bs = self._block_size
        return torch.where(kept, block.block_id * bs + t % bs, -1)

    def _stop_save_at_null_block(self, request: Any) -> None:
        """Store chunks only up to the first one with a null block.

        A checkpoint group keeps a block at chunk ``c`` only from the step
        that computes the chunk's last token through the next one. A step
        that crosses two chunk ends, or a local prefix hit, leaves earlier
        chunks without their state; storing them would publish a wrong state.
        Under speculative decoding only a prompt chunk is stored: a verify step
        leaves a block it crosses without the state at its end.
        """
        chunk = self._lmcache_chunk_size
        first = request.save_spec.skip_leading_tokens // chunk * chunk
        missing = (request.slot_mapping[first::chunk] < 0).any(dim=1).nonzero()
        stop = len(request.token_ids)
        if missing.numel():
            stop = first + int(missing[0].item()) * chunk
        if self._save_prompt_only:
            prompt_len = self._request_trackers[request.req_id].prompt_len
            stop = min(stop, prompt_len // chunk * chunk)
        if stop >= len(request.token_ids):
            return
        request.token_ids = request.token_ids[:stop]
        request.slot_mapping = request.slot_mapping[:stop]
        # Lookups stop at the first chunk not stored, so nothing this request saves
        # after it could be hit.
        self._save_blocked_req_ids.add(request.req_id)

    def record_failed_blocks(self, request_id, expected_mask, ret_mask, slot_mapping):
        # Upstream's unique() handles all columns; -1 denotes a reclaimed
        # block, which must not be reported to vLLM as an allocated block ID.
        if self._use_grouped_slots and slot_mapping.ndim == 2:
            # Upstream divides every column by the cache block. Scale a group
            # of a longer block (Gemma4's full layers) down to it first:
            # (s // r) // b == s // (r * b), and -1 stays -1.
            ratios = [
                group.kv_cache_spec.block_size // self._block_size
                if type(group.kv_cache_spec) in (FullAttentionSpec, SlidingWindowSpec)
                else 1
                for group in self._kv_cache_groups
            ]
            slot_mapping = slot_mapping // torch.tensor(ratios)
        failed = super().record_failed_blocks(
            request_id, expected_mask, ret_mask, slot_mapping
        ) - {-1}
        if self._use_grouped_slots and failed:
            self._failed_load_req_ids.add(request_id)
        return failed

    def start_load_kv(self, forward_context, **kwargs) -> None:
        if self._use_grouped_slots:
            metadata = self._parent._get_connector_metadata()
            self._failed_load_req_ids.difference_update(metadata.reset_req_ids)
        super().start_load_kv(forward_context, **kwargs)

    def wait_for_save(self) -> None:
        if self._use_grouped_slots and self._failed_load_req_ids:
            # Queued forwards may share the failed prefix. Resume publication
            # only after the scheduler invalidates it and resets the requests.
            metadata = self._parent._get_connector_metadata()
            for request in metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)
            return
        super().wait_for_save()

    @staticmethod
    def _normalize_kv_shape(
        kv_caches: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Normalize RBLN 6D KV cache tensors to the 5D format LMCache expects.

        RBLN attention backend produces shape ``[2, B, H, 1, S, D]``, or
        ``[B, 2, H, 1, S, D]`` on its default kernel path; upstream LMCache
        stores caches as ``[2, B, S, H, D]``.  Put the K/V axis back in front,
        squeeze the singleton dim and permute (all stride-only, no copy).
        """
        normalized = {}
        for name, tensor in kv_caches.items():
            if tensor.dim() == 6 and tensor.shape[3] == 1:
                tensor = as_kv_first(tensor).squeeze(3).permute(0, 1, 3, 2, 4)
            normalized[name] = tensor
        return normalized

    @override
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register KV caches with both upstream LMCache and the RBLN connector.

        Upstream LMCache gets the normalized 5D view; the RBLN connector
        gets the 6D RBLN-native tensors **exactly as vLLM allocated them**,
        in the order of the worker's :class:`CacheLayout` regions, for direct
        device-tensor transfer.

        Handing the connector a K/V-first view of a blocks-first cache instead
        would cost it the one thing only the allocation carries: a reference
        that covers its whole storage contiguously, which the RBLN runtime
        requires to lay the staging buffer out like the cache. It transposes
        for its own indexing and keeps the allocation for that call.
        """
        layout = self._cache_layout
        assert layout is not None
        kv_cache_names = [n for r in layout.kv_regions for n in r.layer_names]
        aux_cache_names = [list(r.layer_names) for r in layout.aux_regions]
        placed = layout.cache_names
        # A strict kv_producer runner does not allocate a KDA layer's MTP
        # verify scratch, which the layout does not store either way.
        missing = placed - set(kv_caches) - set(layout.unstored)
        if set(kv_caches) - placed or missing:
            raise ValueError(
                f"vLLM registered caches {sorted(set(kv_caches) - placed)} the "
                f"chunk layout does not place, and the layout names caches "
                f"{sorted(missing)} vLLM did not register"
            )
        normalized_kv_caches = self._normalize_kv_shape(kv_caches)
        super().register_kv_caches(normalized_kv_caches)

        assert self.lmcache_engine is not None
        connector: RBLNConnector = self.lmcache_engine.gpu_connector

        ordered_kv_caches = [kv_caches[name] for name in kv_cache_names]
        aux_kv_caches = [
            [kv_caches[name] for name in names] for names in aux_cache_names
        ]
        block_kv_caches = [
            [(kv_caches[c.name], c.group) for c in region.caches]
            for region in layout.block_regions
        ]
        layer_group_indices = None
        if self._use_grouped_slots:
            layer_groups = {
                name: group_id
                for group_id, group in enumerate(self._kv_cache_groups)
                for name in group.layer_names
            }
            layer_group_indices = [layer_groups[name] for name in kv_cache_names]
        connector.register_kv_caches(
            ordered_kv_caches,
            layer_group_indices=layer_group_indices,
            aux_kv_caches=aux_kv_caches,
            block_kv_caches=block_kv_caches,
        )
        logger.info(
            "Registered %d ordered KV caches (+ %s aux, %s block, not stored %s) on %s",
            len(ordered_kv_caches),
            [len(a) for a in aux_kv_caches],
            [len(b) for b in block_kv_caches],
            list(layout.unstored),
            type(connector).__name__,
        )

    def register_block_aux_kv_caches(self, specs: dict[str, BlockAuxSpec]) -> None:
        # The layout was built at __init__ from the modules; the caches the
        # runner allocated must be the ones it describes, and every cache it
        # stores must be among them. The runner may leave out an unstored one,
        # such as a KDA layer's MTP verify scratch (not the MTP layer's KV).
        unknown = sorted(
            n for n, s in specs.items() if self._block_aux_specs.get(n) != s
        )
        missing = sorted(
            n
            for n, s in self._block_aux_specs.items()
            if s.prefix_cacheable and n not in specs
        )
        if unknown or missing:
            raise ValueError(
                f"vLLM allocated block-aux caches the chunk layout does not hold "
                f"{unknown}, and left out ones it stores {missing}"
            )


class RBLNLMCacheConnectorV1(LMCacheConnectorV1Dynamic, SupportsHMA):
    """RBLN KV connector for vLLM v1.

    Uses :class:`RBLNLMCacheConnectorV1Impl` for device transfers and preserves
    the independent block tables of paged full/sliding attention under HMA.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ):
        # Fail fast if PYTHONHASHSEED is unset: vLLM v1's
        # ``init_none_hash`` falls back to ``BlockHash(os.urandom(32))``
        # per process when this env is missing, which makes the
        # block-hash chain diverge across DP workers + the scheduler.
        # Every LMCache CONTAINS then returns False on store keys, so
        # cache hits silently stay at 0.  LMCache itself emits a
        # WARNING at ``token_database.py:143`` but doesn't refuse to
        # run; raise here so the misconfiguration is caught at
        # connector construction rather than after a 10-minute compile
        # produces 0 cache hits.
        if os.getenv("PYTHONHASHSEED") is None:
            raise RuntimeError(
                "PYTHONHASHSEED is not set, but it is required when "
                "RBLNLMCacheConnectorV1 runs with vLLM v1 (especially "
                "with --data-parallel-size > 1 or --enable-expert-parallel). "
                "Without it, vLLM's NONE_HASH is randomised per process "
                "and every LMCache lookup silently misses.  Set it before "
                "launching vllm serve: `export PYTHONHASHSEED=0`."
            )

        # Device-tensor mode is the only mode RBLNConnector supports: it
        # transfers KV via torch ops on the tensors vLLM hands it. In
        # host-tensor mode a compiled model keeps its KV cache on ``meta``, so
        # there are no bytes to move and the first transfer fails. Guard here
        # — at connector construction, before the multi-minute compile —
        # rather than discovering it mid-run.
        # Ask the platform rather than re-parsing env vars: vllm-rbln derives
        # ``device_type`` from ``VLLM_RBLN_USE_VLLM_MODEL and
        # VLLM_RBLN_USE_DEVICE_TENSOR`` with its own defaults (the latter is
        # on when unset), so this follows whatever mode vllm-rbln actually
        # runs in.
        if current_platform.device_type != "rbln":
            raise RuntimeError(
                "RBLNLMCacheConnectorV1 requires vllm-rbln's device-tensor mode "
                f"(platform device_type is {current_platform.device_type!r}, "
                "expected 'rbln'). Export VLLM_RBLN_USE_VLLM_MODEL=1 before "
                "launching vllm serve; VLLM_RBLN_USE_DEVICE_TENSOR may stay "
                "unset (vllm-rbln defaults it on) or be 1/true, but not 0/false."
            )

        if kv_cache_config is not None:
            KVConnectorBase_V1.__init__(
                self,
                vllm_config=vllm_config,
                role=role,
                kv_cache_config=kv_cache_config,
            )
        else:
            KVConnectorBase_V1.__init__(
                self,
                vllm_config=vllm_config,
                role=role,
            )
        self._lmcache_engine = RBLNLMCacheConnectorV1Impl(vllm_config, role, self)
        extra_config = self._lmcache_engine.config.extra_config or {}
        self._reset_lookup_timeout_s = (
            float(extra_config.get("rbln_reset_lookup_timeout_ms", 30000)) / 1000
        )
        if (
            not math.isfinite(self._reset_lookup_timeout_s)
            or self._reset_lookup_timeout_s <= 0
        ):
            raise ValueError("rbln_reset_lookup_timeout_ms must be positive and finite")
        # Scheduler-side accumulator of worker KV events (see take_events).
        self._kv_cache_events: Optional[list[KVCacheEvent]] = None
        # reset_cache state (see "Cache reset" below). Scheduler: a clear
        # requested but not yet sent / the number of flagged steps not yet
        # confirmed — a count, because under async scheduling a second reset
        # can flag a later step before the first one's confirmation arrives.
        # Worker: this step's bind ran the clear, to report in
        # build_connector_worker_meta.
        self._reset_pending = False
        self._reset_inflight = 0
        self._reset_ran = False
        self._lookup_req_ids: set[str] = set()
        self._async_lookup_req_ids: set[str] = set()
        self._reset_waiting_lookup_ids: set[str] = set()
        self._reset_unpin_req_ids: set[str] = set()
        self._reset_lookup_deadline: Optional[float] = None
        # Scheduler: requests that finished (an abort, typically) after their
        # synchronous lookup but before vLLM allocated their blocks. The lookup
        # pinned the hit chunks on every worker, and with no load to retrieve
        # them nothing unpins them: LMCache 0.5.4's PinMonitor timed such pins
        # out, 0.5.7 has no timeout. The next step's metadata carries these to
        # the workers, which unpin them (bind_connector_metadata).
        self._abandoned_lookup_ids: set[str] = set()

    def register_block_aux_kv_caches(self, specs: dict[str, BlockAuxSpec]) -> None:
        self._lmcache_engine.register_block_aux_kv_caches(specs)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._lmcache_engine.update_state_after_alloc(
            request, num_external_tokens, blocks=blocks
        )
        self._lookup_req_ids.discard(request.request_id)
        self._track_async_lookup(request.request_id)

    def request_finished_all_groups(
        self, request: Any, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # LMCache's synchronous save completes before this callback. Its
        # completion handler only performs cleanup and does not read block IDs.
        self._lmcache_engine._request_blocks.pop(request.request_id, None)
        if (
            request.request_id in self._lookup_req_ids
            and not self._lmcache_engine.async_loading
        ):
            # Async lookups are upstream's to cancel (``request_finished``).
            self._abandoned_lookup_ids.add(request.request_id)
        self._lookup_req_ids.discard(request.request_id)
        result = self._lmcache_engine.request_finished(request, [])
        self._track_async_lookup(request.request_id)
        return result

    def _track_async_lookup(self, req_id: str) -> None:
        actual = _get_async_lookup_client(self._lmcache_engine.lookup_client)
        needs_cleanup = False
        if actual is not None:
            with actual.lock:
                needs_cleanup = actual.reqs_status.get(req_id, -1) is None or (
                    req_id in actual.reqs_status and req_id in actual.aborted_lookups
                )
        if needs_cleanup:
            self._async_lookup_req_ids.add(req_id)
        else:
            self._async_lookup_req_ids.discard(req_id)

    def get_connector(self) -> RBLNConnector:
        """Return the underlying RBLN connector for metrics integration."""
        return self._lmcache_engine.lmcache_engine.gpu_connector

    # ------------------------------------------------------------------
    # Cache reset (vLLM ``reset_prefix_cache(reset_connector=True)``)
    #
    # ``reset_cache`` runs in the scheduler, but the LMCache engine and its
    # RDS backend live in the workers. The clear rides the next step's
    # connector metadata; each worker runs ``LMCacheEngine.clear()`` when it
    # binds that metadata — before ``start_load_kv``, so before any load of
    # the step — and reports back through ``build_connector_worker_meta``.
    #
    # Until that report arrives the scheduler answers new LMCache lookups
    # with 0: a lookup in between would still see the old chunks (the lookup
    # server is in the worker) and promise a load the clear then drops.
    # Lookups made before the reset for requests not yet scheduled are cached
    # (``lookup_client`` status + ``load_specs``); those are discarded too,
    # or they would resurface as hits — and as "already stored, skip save" —
    # once the gate opens. Outstanding async lookups wait for all responses
    # before their status is dropped; workers release the discarded lookups'
    # pins and prefetch references before clear. Gated requests recompute.
    # ------------------------------------------------------------------

    @override
    def reset_cache(self) -> Optional[bool]:
        """Schedule an LMCache clear on every worker; True once accepted.

        The clear runs on the next step after pre-reset lookups have finished
        (an idle engine keeps it queued until a request arrives).
        """
        if self.role != KVConnectorRole.SCHEDULER:
            return None
        impl = self._lmcache_engine
        lookup_client = getattr(impl, "lookup_client", None)
        stale = self._lookup_req_ids | self._async_lookup_req_ids | set(impl.load_specs)
        self._reset_unpin_req_ids.update(stale)
        self._reset_waiting_lookup_ids.update(self._async_lookup_req_ids)
        if self._reset_waiting_lookup_ids and self._reset_lookup_deadline is None:
            # Repeated reset calls must not extend an outstanding wait.
            self._reset_lookup_deadline = (
                time.monotonic() + self._reset_lookup_timeout_s
            )
        if lookup_client is not None:
            for req_id in stale - self._reset_waiting_lookup_ids:
                lookup_client.clear_lookup_status(req_id)
        impl.load_specs.clear()
        self._reset_pending = True
        logger.info(
            "LMCache reset requested: clearing on the next step "
            "(dropped %d cached lookups)",
            len(stale),
        )
        return True

    def _drain_reset_lookups(self) -> bool:
        """Wait for actual async replies before forgetting their status.

        LMCache 0.5.7's async lookup_cache can return a timeout miss while
        prefetch is still running. Inspect its response table under its lock
        instead, unwrapping the optional statistics/hit-limit decorators.
        Once all replies arrive, worker lookup_unpin can release prefetch
        references before clear; no old reply can repopulate the status table.
        A missing reply fails the reset at its deadline: LMCache exposes no
        cancellation acknowledgement that would make early clear safe.
        """
        client = self._lmcache_engine.lookup_client
        actual = client
        while actual is not None and "actual_lookup_client" in vars(actual):
            actual = actual.actual_lookup_client
        for req_id in tuple(self._reset_waiting_lookup_ids):
            if actual is not None and hasattr(actual, "lock"):
                with actual.lock:
                    result = actual.reqs_status.get(req_id, -1)
                    if result is not None:
                        # Reset metadata now owns worker cleanup. Do not let
                        # a later lookup send another cleanup for this ID.
                        getattr(actual, "aborted_lookups", set[str]()).discard(req_id)
            else:
                result = client.lookup_cache(req_id) if client is not None else 0
            if result is None:
                continue
            if client is not None:
                client.clear_lookup_status(req_id)
            self._reset_waiting_lookup_ids.remove(req_id)
            self._async_lookup_req_ids.discard(req_id)
        if not self._reset_waiting_lookup_ids:
            self._reset_lookup_deadline = None
            return True
        if (
            self._reset_lookup_deadline is not None
            and time.monotonic() >= self._reset_lookup_deadline
        ):
            message = (
                f"LMCache reset timed out after {self._reset_lookup_timeout_s:g}s "
                "waiting for async lookup replies: "
                f"{sorted(self._reset_waiting_lookup_ids)}. "
                "Cannot safely clear unfinished prefetches; restart the engine."
            )
            logger.error(message)
            raise TimeoutError(message)
        return False

    @override
    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        if self._reset_pending or self._reset_inflight > 0:
            self._drain_reset_lookups()
            if request.request_id in self._reset_waiting_lookup_ids:
                # update_state_after_alloc would erase the response slot while
                # a pre-reset prefetch still owns pinned CPU cache objects.
                return None, False
            if not request.request_id.startswith("mock_req"):
                # Leave the load spec upstream records for a lookup miss: the
                # request tracker reads vllm_cached_tokens from it (a request
                # resumed after preemption included), and 0 LMCache tokens
                # means the whole prompt is saved again.
                self._lmcache_engine.load_specs[request.request_id] = LoadSpec(
                    num_computed_tokens, 0, False
                )
            return 0, False
        if not request.request_id.startswith("mock_req"):
            self._lookup_req_ids.add(request.request_id)
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if result[0] is None:
            self._async_lookup_req_ids.add(request.request_id)
        else:
            self._track_async_lookup(request.request_id)
        return result

    @override
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        if not self._reset_pending:
            # Drop orphaned timeout IDs only after normal aborted-lookup
            # cleanup has consumed the reply and released its references.
            actual = _get_async_lookup_client(self._lmcache_engine.lookup_client)
            if actual is not None:
                with actual.lock:
                    self._async_lookup_req_ids.intersection_update(actual.reqs_status)
        meta = super().build_connector_meta(scheduler_output)
        if self._abandoned_lookup_ids:
            meta.lmcache_rbln_unpin_lookup_ids = (  # type: ignore[attr-defined]
                self._abandoned_lookup_ids
            )
            self._abandoned_lookup_ids = set()
        if self._reset_pending and self._drain_reset_lookups():
            meta.clear_lmcache = True  # type: ignore[attr-defined]
            meta.lmcache_rbln_reset_lookup_ids = set(  # type: ignore[attr-defined]
                self._reset_unpin_req_ids
            )
            self._reset_unpin_req_ids.clear()
            self._lookup_req_ids.clear()
            self._reset_pending = False
            self._reset_inflight += 1
        return meta

    @override
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        engine = self._lmcache_engine.lmcache_engine
        if engine is None:
            return
        for req_id in getattr(connector_metadata, "lmcache_rbln_unpin_lookup_ids", ()):
            engine.lookup_unpin(req_id)
        if getattr(connector_metadata, "clear_lmcache", False):
            for req_id in getattr(
                connector_metadata, "lmcache_rbln_reset_lookup_ids", ()
            ):
                engine.lookup_unpin(req_id)
            num_tokens = engine.clear()
            self._reset_ran = True
            logger.info("LMCache reset: cleared %d tokens", num_tokens)

    @override
    def build_connector_worker_meta(self) -> Optional[KVConnectorWorkerMetadata]:
        if not self._reset_ran:
            return None
        self._reset_ran = False
        return _LMCacheClearDone()

    # ------------------------------------------------------------------
    # KV cache events (worker -> scheduler -> vLLM's KV event publisher)
    #
    # Same bridge as vLLM's in-tree ``LMCacheConnectorV1``, which the upstream
    # ``LMCacheConnectorV1Dynamic`` we subclass lacks: the worker converts the
    # engine's ``CacheStoreEvent`` queue (filled by ``store`` when
    # ``LMCACHE_ENABLE_KV_EVENTS`` is on) into ``BlockStored`` and rides it to
    # the scheduler on ``KVConnectorOutput.kv_cache_events``; the scheduler
    # hands it to ``kv_event_publisher`` via ``take_events``.
    #
    # Every storing rank reports, including an empty batch when nothing was
    # stored. RBLNKVEvents requires all ranks to advertise a store, but a
    # removal on any rank invalidates the pod's full KV. Passive MLA ranks do
    # not store and must not vote. Non-last PP stages reach the scheduler:
    # the model runner stashes their ``kv_connector_output`` and returns it
    # from ``sample_tokens``.
    # ------------------------------------------------------------------

    @override
    def get_kv_connector_kv_cache_events(self) -> Optional[RBLNKVEvents]:
        engine = self._lmcache_engine.lmcache_engine
        if engine is None:
            return None
        stored = engine.get_kv_events()
        if not engine.kv_events_enabled or (
            engine.save_only_first_rank and not engine.metadata.is_first_rank()
        ):
            return None
        events: list[KVCacheEvent] = [
            BlockStored(
                block_hashes=e.block_hashes,
                parent_block_hash=e.parent_block_hash,
                token_ids=e.token_ids,
                block_size=e.block_size,
                lora_id=e.lora_id,
                medium=e.medium,
                lora_name=e.lora_name,
            )
            for e in stored
        ]
        # Empty batches are votes too: otherwise a successful stage could
        # advertise KV while another stage failed or skipped its store.
        kv_events = RBLNKVEvents(num_workers=1)
        # Several requests in the same step can store the same prefix. The
        # consensus counts rank votes, so repeated events from one worker
        # must count only once.
        kv_events.add_events(list(dict.fromkeys(events)))
        return kv_events

    @override
    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        if isinstance(connector_output.kv_connector_worker_meta, _LMCacheClearDone):
            # One confirmation per flagged step; lookups resume after the last.
            self._reset_inflight = max(0, self._reset_inflight - 1)
            if self._reset_inflight == 0 and not self._reset_pending:
                logger.info("LMCache reset confirmed by workers; lookups resume")
        kv_cache_events = connector_output.kv_cache_events
        if not isinstance(kv_cache_events, LMCacheKVEvents):
            return
        if self._kv_cache_events is None:
            self._kv_cache_events = []
        # Consensus belongs to one executor step. Distinct steps accumulated
        # before take_events must not be mistaken for additional rank votes.
        self._kv_cache_events.extend(kv_cache_events.aggregate().get_all_events())

    @override
    def take_events(self) -> Iterable[KVCacheEvent]:
        if self._kv_cache_events is None:
            return ()
        events = self._kv_cache_events
        self._kv_cache_events = None
        return events

    @override
    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """No-op: LMCache is a process-local offload tier, not a party to
        cross-instance KV transfer, so PP-disaggregated handshake metadata
        (pp_rank > 0) doesn't apply here. The base class default raises on
        any pp_rank > 0 key, which kills the whole MultiConnector stack even
        when another member (e.g. NIXL) is the one actually doing PP
        transfer."""

# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of a checkpointed (``MambaSpec``) engine such as GLM-5.3.

A checkpoint group keeps a block at chunk ``c`` only around the step that
computes the chunk's last token; afterwards that block-table entry is null.
Only chunks whose every group has a block may be saved, and a hit resumes at
a block end, where a checkpoint exists. The engine settings that rule needs
are checked when the connector starts.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import CachedRequestData
import pytest
import torch

# First Party
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import LoadSpec
from lmcache.inprocess.rbln.integration.vllm.cache_layout import CacheLayout
from lmcache.inprocess.rbln.integration.vllm.connector import (
    RBLNLMCacheConnectorV1Impl,
    block_aux_specs,
)
from lmcache.inprocess.v1.config import LMCacheEngineConfig

# Local
from . import kv_spec_fixtures as fx

CONNECTOR = "lmcache.inprocess.rbln.integration.vllm.connector"

BS = fx.GLM_BLOCK_SIZE
NULL = KVCacheBlock(0, is_null=True)


def _impl() -> RBLNLMCacheConnectorV1Impl:
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = True
    impl._has_checkpoints = True
    impl._kv_cache_groups = fx.glm53(4).kv_cache_config.kv_cache_groups
    impl._request_blocks = {}
    impl._save_blocked_req_ids = set()
    impl._manager = MagicMock()
    impl.config = SimpleNamespace(priority_limit=None, save_decode_cache=False)
    impl.kv_role = "kv_both"
    impl.force_skip_save = False
    impl._request_trackers = {}
    impl._unfinished_requests = {}
    impl._requests_priority = {}
    impl.load_specs = {}
    impl._block_size = BS
    impl._lmcache_chunk_size = BS
    impl._discard_partial_chunks = True
    return impl


def _schedule_new(
    impl: RBLNLMCacheConnectorV1Impl,
    prompt: int,
    scheduled: int,
    groups: tuple[list[KVCacheBlock], ...],
    kv_transfer_params: dict[str, Any] | None = None,
    computed: int = 0,
) -> tuple[Any, SimpleNamespace, SimpleNamespace]:
    request = SimpleNamespace(
        request_id="req",
        num_computed_tokens=computed,
        num_tokens=prompt,
        all_token_ids=list(range(prompt)),
        kv_transfer_params=kv_transfer_params,
    )
    impl.update_state_after_alloc(request, computed, KVCacheBlocks(groups))
    new_request = SimpleNamespace(
        req_id="req",
        prompt_token_ids=request.all_token_ids,
        block_ids=tuple([b.block_id for b in g] for g in groups),
        num_computed_tokens=computed,
        sampling_params=None,
        mm_features=[],
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        num_scheduled_tokens={"req": scheduled},
    )
    return impl.build_connector_meta(output).requests[0], request, output


def _first_rows(meta: Any) -> list[list[int]]:
    """Each chunk's block per group, -1 for null."""
    return (meta.slot_mapping[::BS] // BS).clamp(min=-1).tolist()


def test_each_chunk_is_saved_in_the_step_that_completes_it() -> None:
    impl = _impl()
    groups = (
        [KVCacheBlock(1)],
        [KVCacheBlock(2)],
        [KVCacheBlock(3)],
        [KVCacheBlock(4)],
    )
    first, request, output = _schedule_new(impl, 3 * BS, BS, groups)
    assert first.save_spec.can_save
    assert len(first.token_ids) == BS
    assert _first_rows(first) == [[1, 2, 3, 4]]

    # The next chunk end: chunk 0's KDA entries are null by now.
    groups[0].append(KVCacheBlock(5))
    for group, block in zip(groups[1:], (6, 7, 8), strict=True):
        group[0] = NULL
        group.append(KVCacheBlock(block))
    request.num_computed_tokens = BS
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["req"], new_block_ids=[([5], [6], [7], [8])], resumed_req_ids=set()
    )
    output.num_scheduled_tokens = {"req": BS}
    second = impl.build_connector_meta(output).requests[0]
    assert second.save_spec.can_save
    assert second.save_spec.skip_leading_tokens == BS
    assert len(second.token_ids) == 2 * BS
    assert _first_rows(second) == [[1, -1, -1, -1], [5, 6, 7, 8]]
    assert "req" not in impl._save_blocked_req_ids


def test_a_step_past_two_chunk_ends_saves_none_of_them() -> None:
    impl = _impl()
    # Chunk 0's KDA states were never kept: the step computed past them.
    groups = (
        [KVCacheBlock(1), KVCacheBlock(5)],
        [NULL, KVCacheBlock(6)],
        [NULL, KVCacheBlock(7)],
        [NULL, KVCacheBlock(8)],
    )
    meta, _, _ = _schedule_new(impl, 3 * BS, 2 * BS, groups)
    assert meta.token_ids == []
    assert len(meta.slot_mapping) == 0
    assert "req" in impl._save_blocked_req_ids


def test_saving_stops_before_the_first_chunk_without_a_state() -> None:
    impl = _impl()
    groups = (
        [KVCacheBlock(1), KVCacheBlock(5)],
        [KVCacheBlock(2), NULL],
        [KVCacheBlock(3), KVCacheBlock(7)],
        [KVCacheBlock(4), KVCacheBlock(8)],
    )
    meta, _, _ = _schedule_new(impl, 3 * BS, 2 * BS, groups)
    assert meta.save_spec.can_save
    assert len(meta.token_ids) == BS
    assert _first_rows(meta) == [[1, 2, 3, 4]]
    assert "req" in impl._save_blocked_req_ids


def _impl_without_prefix_caching() -> RBLNLMCacheConnectorV1Impl:
    impl = _impl()
    fixture = fx.glm53(4, mamba_cache_mode="none")
    impl._kv_cache_groups = fixture.kv_cache_config.kv_cache_groups
    impl._running_state_groups = frozenset({1, 2, 3})
    return impl


def _running_states() -> tuple[list[KVCacheBlock], ...]:
    """A latent block, then each KDA group's one running state."""
    return ([KVCacheBlock(1)], [KVCacheBlock(2)], [KVCacheBlock(3)], [KVCacheBlock(4)])


def test_without_prefix_caching_a_chunk_is_saved_in_the_step_ending_at_it() -> None:
    impl = _impl_without_prefix_caching()
    groups = _running_states()
    first, request, output = _schedule_new(impl, 3 * BS, BS, groups)
    assert first.save_spec.can_save
    assert len(first.token_ids) == BS
    assert _first_rows(first) == [[1, 2, 3, 4]]

    # Only the chunk ending at the step's end holds the state, at a block start.
    groups[0].append(KVCacheBlock(5))
    request.num_computed_tokens = BS
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["req"], new_block_ids=[([5], [], [], [])], resumed_req_ids=set()
    )
    output.num_scheduled_tokens = {"req": BS}
    second = impl.build_connector_meta(output).requests[0]
    assert second.save_spec.can_save
    assert second.save_spec.skip_leading_tokens == BS
    assert _first_rows(second) == [[1, -1, -1, -1], [5, 2, 3, 4]]
    assert (second.slot_mapping[BS:, 1:] % BS == torch.arange(BS)[:, None]).all()
    assert "req" not in impl._save_blocked_req_ids


@pytest.mark.parametrize("scheduled", [BS + 512, 2 * BS])
def test_without_prefix_caching_a_chunk_ending_inside_a_step_stops_saving(
    scheduled: int,
) -> None:
    # The running state is past chunk 0's end by the time it could be saved.
    impl = _impl_without_prefix_caching()
    groups = _running_states()
    groups[0].append(KVCacheBlock(5))
    meta, _, _ = _schedule_new(impl, 3 * BS, scheduled, groups)
    assert meta.token_ids == []
    assert "req" in impl._save_blocked_req_ids


def test_without_prefix_caching_a_load_restores_only_the_resume_chunks_state() -> None:
    impl = _impl_without_prefix_caching()
    impl.load_specs["req"] = LoadSpec(0, 2 * BS, True)
    groups = _running_states()
    groups[0].extend([KVCacheBlock(5), KVCacheBlock(6)])
    meta, _, _ = _schedule_new(impl, 3 * BS, 512, groups, computed=2 * BS)
    assert meta.load_spec.can_load
    assert not meta.save_spec.can_save
    # Whatever order the chunks arrive in, only chunk 1's state lands.
    assert _first_rows(meta) == [[1, -1, -1, -1], [5, 2, 3, 4]]


def _step(
    impl: RBLNLMCacheConnectorV1Impl,
    request: SimpleNamespace,
    output: SimpleNamespace,
    groups: tuple[list[KVCacheBlock], ...],
    scheduled: int,
) -> list[Any]:
    """Schedule the running request's next ``scheduled`` tokens."""
    new: list[int] = []
    if request.num_computed_tokens % BS == 0:
        groups[0].append(KVCacheBlock(4 + len(groups[0])))
        new = [groups[0][-1].block_id]
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["req"], new_block_ids=[(new, [], [], [])], resumed_req_ids=set()
    )
    output.num_scheduled_tokens = {"req": scheduled}
    return impl.build_connector_meta(output).requests


def _saved(metas: list[Any]) -> list[list[int]]:
    """The first rows of the chunks a step saves."""
    return [
        row
        for m in metas
        if m.save_spec.can_save
        for row in _first_rows(m)[m.save_spec.skip_leading_tokens // BS :]
    ]


def test_without_prefix_caching_512_token_steps_save_each_chunk_at_its_end() -> None:
    impl = _impl_without_prefix_caching()
    groups = _running_states()
    meta, request, output = _schedule_new(impl, 3 * BS, 512, groups)
    saves = {512: _saved([meta])}
    for end in range(512, 2 * BS + 512, 512):
        request.num_computed_tokens = end
        saves[end + 512] = _saved(_step(impl, request, output, groups, 512))
        assert "req" not in impl._save_blocked_req_ids
    assert {end: rows for end, rows in saves.items() if rows} == {
        BS: [[1, 2, 3, 4]],
        2 * BS: [[5, 2, 3, 4]],
    }


def test_without_prefix_caching_a_hit_saves_the_next_chunk_at_its_end() -> None:
    impl = _impl_without_prefix_caching()
    impl.load_specs["req"] = LoadSpec(0, BS, True)
    groups = _running_states()
    groups[0].append(KVCacheBlock(5))
    meta, request, output = _schedule_new(impl, 3 * BS, 512, groups, computed=BS)
    assert meta.load_spec.can_load
    assert _first_rows(meta) == [[1, 2, 3, 4]]
    saves = {}
    for end in range(BS + 512, 2 * BS, 512):
        request.num_computed_tokens = end
        saves[end + 512] = _saved(_step(impl, request, output, groups, 512))
    assert {end: rows for end, rows in saves.items() if rows} == {
        2 * BS: [[5, 2, 3, 4]]
    }
    assert "req" not in impl._save_blocked_req_ids


@pytest.mark.parametrize("lag, saved", [(0, [[1, 2, 3, 4]]), (1, [])])
def test_without_prefix_caching_a_decode_saves_a_chunk_only_at_its_end(
    lag: int, saved: list[list[int]]
) -> None:
    # Under async scheduling a sampled token reaches all_token_ids a step
    # late, so LMCache's tracker holds 4096 tokens when the state is at 4097.
    impl = _impl_without_prefix_caching()
    impl.config.save_decode_cache = True
    groups = _running_states()
    prompt = BS - 2
    _, request, output = _schedule_new(impl, prompt, prompt, groups)
    saves = []
    for computed in range(prompt, BS + 1):
        request.num_computed_tokens = computed
        request.all_token_ids = list(range(computed + 1 - lag))
        saves += _saved(_step(impl, request, output, groups, 1))
    assert saves == saved


@pytest.mark.parametrize(
    "hit, prompt, computed, external",
    [
        (2 * BS, 2 * BS, 0, BS),  # a full-prompt hit resumes a block early
        (2 * BS, 3 * BS, 0, 2 * BS),
        (2 * BS, 2 * BS + 100, BS, BS),
        (BS, 2 * BS, BS, 0),
    ],
)
def test_a_hit_resumes_at_a_block_end(
    hit: int, prompt: int, computed: int, external: int
) -> None:
    impl = _impl()
    impl._max_tokens_per_load = 0
    impl.config = SimpleNamespace(min_retrieve_tokens=0)
    impl._manager.lookup_client.lookup_cache.return_value = hit
    request = SimpleNamespace(
        request_id="req", num_tokens=prompt, skip_reading_prefix_cache=False
    )
    assert impl.get_num_new_matched_tokens(request, computed) == external
    load_spec = impl.load_specs["req"]
    assert load_spec.lmcache_cached_tokens % BS == 0
    if external:
        assert load_spec.lmcache_cached_tokens - computed == external


def _vllm_config(drafts: int = 0, **overrides: Any) -> SimpleNamespace:
    fixture = fx.glm53(4, drafts)
    modules = {
        n: SimpleNamespace(get_aux_cache_spec=lambda config, s=spec: s)
        for n, spec in fixture.aux_specs.items()
    }
    config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        cache_config=SimpleNamespace(
            block_size=BS, mamba_block_size=BS, enable_prefix_caching=True
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=512,
            max_num_scheduled_tokens=None,
            long_prefill_token_threshold=0,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(kv_role="kv_both"),
        compilation_config=SimpleNamespace(static_forward_context=modules),
        additional_config={},
    )
    for key, value in overrides.items():
        if key in ("speculative_config", "kv_transfer_config", "additional_config"):
            setattr(config, key, value)
        elif hasattr(config.scheduler_config, key):
            setattr(config.scheduler_config, key, value)
        elif hasattr(config.cache_config, key):
            setattr(config.cache_config, key, value)
        else:
            setattr(config.parallel_config, key, value)
    return config


def _init(
    config: LMCacheEngineConfig,
    role: KVConnectorRole = KVConnectorRole.SCHEDULER,
    num_layers: int = 4,
    drafts: int = 0,
    mamba_cache_mode: str = "align",
    **overrides: Any,
) -> tuple[RBLNLMCacheConnectorV1Impl, Any]:
    engine_config = _vllm_config(drafts, **overrides)
    fixture = fx.glm53(num_layers, drafts, mamba_cache_mode=mamba_cache_mode)
    parent = SimpleNamespace(_kv_cache_config=fixture.kv_cache_config)

    def init_state(self: Any, role: Any, vllm_config: Any, config: Any) -> None:
        self._block_size = vllm_config.cache_config.block_size
        self._lmcache_chunk_size = config.chunk_size
        self._discard_partial_chunks = False

    with (
        patch(
            f"{CONNECTOR}.lmcache_get_or_create_config",
            return_value=config,
        ),
        patch.object(RBLNLMCacheConnectorV1Impl, "_apply_extra_config"),
        patch.object(RBLNLMCacheConnectorV1Impl, "_init_connector_state", init_state),
        patch.object(RBLNLMCacheConnectorV1Impl, "_setup_metrics"),
        patch(f"{CONNECTOR}.RBLNServiceFactory") as factory,
        patch(f"{CONNECTOR}.LMCacheManager"),
    ):
        impl = RBLNLMCacheConnectorV1Impl(engine_config, role, parent)
    return impl, factory


def test_a_lone_checkpoint_group_still_resumes_at_a_block_end() -> None:
    # One KDA layer leaves a single MambaSpec group and nothing else.
    impl, _ = _init(LMCacheEngineConfig(chunk_size=BS), num_layers=1)
    assert len(impl._kv_cache_groups) == 1
    impl._max_tokens_per_load = 0
    impl.load_specs = {}
    impl.config = SimpleNamespace(min_retrieve_tokens=0)
    impl._manager = MagicMock()
    impl._manager.lookup_client.lookup_cache.return_value = 2 * BS
    request = SimpleNamespace(
        request_id="req", num_tokens=2 * BS, skip_reading_prefix_cache=False
    )
    assert impl.get_num_new_matched_tokens(request, 0) == BS
    assert impl.load_specs["req"].lmcache_cached_tokens == BS


def test_scheduler_accepts_checkpoint_groups_without_a_layout() -> None:
    impl, factory = _init(LMCacheEngineConfig(chunk_size=BS))
    assert impl._has_checkpoints
    assert factory.call_args.args[3] is None
    assert impl._lmcache_chunk_size == BS
    assert impl._discard_partial_chunks


def test_worker_hands_the_block_layout_to_the_factory() -> None:
    impl, factory = _init(LMCacheEngineConfig(chunk_size=BS), KVConnectorRole.WORKER)
    fixture = fx.glm53(4)
    layout = factory.call_args.args[3]
    assert isinstance(layout, CacheLayout)
    assert layout == CacheLayout.from_kv_cache_config(
        fixture.kv_cache_config, fixture.aux_specs
    )


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"tensor_parallel_size": 2}, "tensor_parallel_size is 2"),
        ({"mamba_block_size": 1024}, "mamba_block_size 1024 != block_size 4096"),
        (
            {"enable_prefix_caching": False, "max_num_batched_tokens": 3000},
            "prefill step 3000 does not divide block_size 4096",
        ),
        (
            {"enable_prefix_caching": False, "long_prefill_token_threshold": 300},
            "prefill step 300 does not divide",
        ),
        (
            {
                "enable_prefix_caching": False,
                "additional_config": SimpleNamespace(prefill_chunk_size=[256, 384]),
            },
            "prefill step 384 does not divide",
        ),
        (
            {"kv_transfer_config": SimpleNamespace(kv_role="kv_producer")},
            "kv_role is kv_producer",
        ),
    ],
)
@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_engine_settings_the_block_layout_cannot_serve_are_refused(
    overrides: dict[str, Any], reason: str, role: KVConnectorRole
) -> None:
    with pytest.raises(ValueError, match=reason):
        _init(LMCacheEngineConfig(chunk_size=BS), role, **overrides)


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_speculative_decoding_is_served_while_decodes_are_not_saved(
    role: KVConnectorRole,
) -> None:
    # A verify step leaves a crossed block without its end state, and only
    # decode steps verify. The worker builds its layout with the MTP drafts.
    spec = SimpleNamespace(method="mtp", num_speculative_tokens=1)
    impl, _ = _init(
        LMCacheEngineConfig(chunk_size=BS), role, drafts=1, speculative_config=spec
    )
    assert impl._save_prompt_only
    with pytest.raises(ValueError, match="speculative decoding is on with save_"):
        _init(
            LMCacheEngineConfig(chunk_size=BS, save_decode_cache=True),
            role,
            drafts=1,
            speculative_config=spec,
        )


def _verify_step(
    impl: RBLNLMCacheConnectorV1Impl,
    request: SimpleNamespace,
    output: SimpleNamespace,
    groups: tuple[list[KVCacheBlock], ...],
    resumed: bool = False,
) -> Any:
    """Cross the first chunk end with the next step: 1 token + 1 draft, or
    the whole sequence again after a preemption."""
    for group, block in zip(groups, (5, 6, 7, 8), strict=True):
        group.append(KVCacheBlock(block))
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["req"],
        new_block_ids=[([5], [6], [7], [8])],
        resumed_req_ids={"req"} if resumed else set(),
    )
    computed = request.num_computed_tokens
    output.num_scheduled_tokens = {"req": len(request.all_token_ids) - computed + 1}
    return impl.build_connector_meta(output).requests[0]


DISAGG = {
    "disagg_spec": {
        "req_id": "req",
        "receiver_host": "localhost",
        "receiver_init_port": 1,
        "receiver_alloc_port": 2,
    }
}


@pytest.mark.parametrize("disagg", [False, True])
def test_under_speculative_decoding_a_verify_step_stores_no_chunk(
    disagg: bool,
) -> None:
    # LMCache saves a decode step of a request with a disagg_spec, or with
    # save_decode_cache switched on at runtime (POST /conf).
    impl = _impl()
    impl._save_prompt_only = True
    impl.config.save_decode_cache = not disagg
    groups = tuple([KVCacheBlock(b)] for b in (1, 2, 3, 4))
    _, request, output = _schedule_new(
        impl, BS - 1, BS - 1, groups, DISAGG if disagg else None
    )
    request.num_computed_tokens = BS - 1
    request.all_token_ids = list(range(BS))
    meta = _verify_step(impl, request, output, groups)
    assert meta.save_spec.can_save
    assert meta.token_ids == []
    assert "req" in impl._save_blocked_req_ids


def test_under_speculative_decoding_a_resumed_request_stores_only_its_prompt() -> None:
    # Preempted before its first decode, the request recomputes its prompt
    # and the sampled token in one prefill.
    impl = _impl()
    impl._save_prompt_only = True
    groups = tuple([KVCacheBlock(b)] for b in (1, 2, 3, 4))
    _, request, output = _schedule_new(impl, BS - 1, BS - 1, groups)
    impl.load_specs["req"] = LoadSpec(0, 0, False)
    request.all_token_ids = list(range(BS))
    meta = _verify_step(impl, request, output, groups, resumed=True)
    assert meta.save_spec.can_save
    assert meta.token_ids == []
    assert "req" in impl._save_blocked_req_ids


@pytest.mark.parametrize("threshold", [0, 1000])
@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_prefix_caching_off_is_served_when_steps_end_at_chunk_ends(
    role: KVConnectorRole, threshold: int
) -> None:
    # A threshold above the 512-token budget does not cut a step.
    impl, _ = _init(
        LMCacheEngineConfig(chunk_size=BS),
        role,
        mamba_cache_mode="none",
        enable_prefix_caching=False,
        mamba_block_size=fx.GLM_MAX_MODEL_LEN,
        long_prefill_token_threshold=threshold,
    )
    assert impl._running_state_groups == {1, 2, 3}


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_engine_accepts_pipeline_parallel_stages(role: KVConnectorRole) -> None:
    # Each stage stores and restores only its own layers; GLM-5.3-Flash needs
    # PP4 to hold all 45 layers.
    impl, _ = _init(LMCacheEngineConfig(chunk_size=BS), role, pipeline_parallel_size=4)
    assert impl._has_checkpoints


def test_engine_refuses_a_chunk_smaller_than_a_block() -> None:
    with pytest.raises(ValueError, match="LMCACHE_CHUNK_SIZE 256 != block_size"):
        _init(LMCacheEngineConfig(chunk_size=256))


def test_every_problem_is_listed_at_once() -> None:
    with pytest.raises(ValueError) as e:
        _init(
            LMCacheEngineConfig(chunk_size=BS),
            tensor_parallel_size=2,
            mamba_block_size=8,
        )
    assert "tensor_parallel_size is 2" in str(e.value)
    assert "mamba_block_size 8" in str(e.value)


@dataclass(frozen=True)
class _AuxSpecBeforeA1:
    owner: str
    block_shape: tuple[int, ...]
    dtype: torch.dtype


def test_a_vllm_rbln_without_prefix_cacheable_is_refused() -> None:
    config = _vllm_config()
    for module in config.compilation_config.static_forward_context.values():
        spec = module.get_aux_cache_spec(config)
        old = _AuxSpecBeforeA1(spec.owner, spec.block_shape, spec.dtype)
        module.get_aux_cache_spec = lambda c, s=old: s
    with pytest.raises(ValueError, match="has no prefix_cacheable"):
        block_aux_specs(config)  # type: ignore[arg-type]


def test_the_runner_must_hand_over_the_block_aux_caches_the_layout_holds() -> None:
    specs = block_aux_specs(_vllm_config())  # type: ignore[arg-type]
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._block_aux_specs = specs
    impl.register_block_aux_kv_caches(dict(specs))
    name, spec = next(iter(specs.items()))
    dropped = dict(list(specs.items())[1:])
    with pytest.raises(ValueError, match=rf"left out ones it stores \['{name}'\]"):
        impl.register_block_aux_kv_caches(dropped)
    changed = {**specs, name: fx.AuxSpec(spec.owner, (1,), spec.dtype)}
    with pytest.raises(ValueError, match=rf"does not hold \['{name}'\]"):
        impl.register_block_aux_kv_caches(changed)


def test_the_runner_may_leave_the_kda_verify_scratch_out() -> None:
    # vllm-rbln hands over the block-aux caches a connector moves, without a
    # KDA layer's MTP verify scratch; the layout does not store it either way.
    specs = fx.glm53(4, drafts=1).aux_specs
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._block_aux_specs = dict(specs)
    moved = {n: s for n, s in specs.items() if not s.verify_only}
    assert len(moved) < len(specs)
    impl.register_block_aux_kv_caches(dict(moved))
    impl.register_block_aux_kv_caches(dict(specs))


class _Registered(Exception):
    pass


@pytest.mark.parametrize("verify_scratch", [True, False])
def test_a_strict_producer_registers_its_kv_without_the_kda_verify_scratch(
    verify_scratch: bool,
) -> None:
    # A kv_producer engine (a PD prefill) does not allocate the scratch, so
    # vLLM registers every other cache; a stored cache may still not be missing.
    fixture = fx.glm53(4, drafts=1)
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._cache_layout = CacheLayout.from_kv_cache_config(
        fixture.kv_cache_config, fixture.aux_specs
    )
    scratch = {n for n, s in fixture.aux_specs.items() if s.verify_only}
    assert scratch
    caches = {
        name: torch.empty(0)
        for name in fixture.registered
        if verify_scratch or name not in scratch
    }
    with patch.object(
        RBLNLMCacheConnectorV1Impl.__base__,
        "register_kv_caches",
        side_effect=_Registered,
    ):
        with pytest.raises(_Registered):
            impl.register_kv_caches(caches)
        stored = impl._cache_layout.block_regions[-1].caches[0].name
        del caches[stored]
        with pytest.raises(ValueError, match=rf"\['{stored}'\] vLLM did not"):
            impl.register_kv_caches(caches)

# SPDX-License-Identifier: Apache-2.0
"""A synchronous lookup's pins are released when its request never gets blocks.

The lookup server pins the hit chunks on every worker under the request id,
and only a load or save of that request unpins them. A request that finishes
(an abort, typically) after its lookup but before vLLM allocates its blocks is
never loaded or saved, so without the scheduler handing its id to the workers
the chunks stay pinned, and unevictable, for the life of the engine.
"""

# Standard
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (  # noqa: E402
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
)
from lmcache.inprocess.v1.cache_engine import LMCacheEngine  # noqa: E402

# Local
from .test_retrieve_pin_ownership import (  # noqa: E402, F401 -- shared fixtures
    _CacheCase,
    cache_case,
)
from .test_vllm_sync_load_pin_cleanup import _CPUConnector  # noqa: E402

pytestmark = pytest.mark.no_shared_allocator


class _SyncLookupClient:
    """Answer lookups the way the sync lookup server does: pin on the engine."""

    def __init__(self, engine: LMCacheEngine) -> None:
        self.engine = engine
        self.status: dict[str, int] = {}

    def lookup_cache(self, lookup_id: str) -> int:
        return self.status.get(lookup_id, -1)

    def lookup(
        self,
        token_ids: list[int],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> int:
        hits = self.engine.lookup(token_ids, lookup_id=lookup_id, pin=True)
        self.status[lookup_id] = hits
        return hits

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.status.pop(lookup_id, None)


def _scheduler(lookup_client: _SyncLookupClient) -> LMCacheConnectorV1Impl:
    """A scheduler-role connector with only the state the lookup path reads."""
    impl = object.__new__(LMCacheConnectorV1Impl)
    impl._manager = MagicMock(lmcache_engine=None, lookup_client=lookup_client)
    impl.kv_role = "kv_both"
    impl.async_loading = False
    impl.use_layerwise = False
    impl.skip_last_n_tokens = 0
    impl.config = SimpleNamespace(
        min_retrieve_tokens=0, get_extra_config_value=lambda key, default: default
    )
    impl.force_skip_save = False
    impl._max_tokens_per_load = 0
    impl._lmcache_chunk_size = 4
    impl.load_specs = {}
    impl._requests_priority = {}
    impl._request_trackers = {}
    impl._unfinished_requests = {}
    impl._unallocated_sync_lookup_ids = set()
    impl._abandoned_lookup_ids = set()
    return impl


def _request(req_id: str, tokens: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id,
        all_token_ids=tokens,
        prompt_token_ids=tokens,
        num_tokens=len(tokens) + 1,
        sampling_params=None,
        mm_features=None,
        status=RequestStatus.WAITING,
        kv_transfer_params=None,
    )


def _empty_step(finished: set[str]) -> SimpleNamespace:
    return SimpleNamespace(
        finished_req_ids=finished,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            resumed_from_preemption=[],
            resumed_req_ids=set(),
            new_block_ids=[],
            num_computed_tokens=[],
        ),
        num_scheduled_tokens={},
        preempted_req_ids=set(),
    )


def _lookup(scheduler: LMCacheConnectorV1Impl, request: SimpleNamespace) -> int:
    hits = scheduler.get_num_new_matched_tokens(request, num_computed_tokens=0)
    assert hits is not None
    return hits


def test_aborted_before_allocation_unpins_on_the_next_step(
    cache_case: _CacheCase,  # noqa: F811 -- pytest injects the imported fixture
) -> None:
    engine = cache_case.engine
    first_obj = cache_case.memory_objs[0]
    scheduler = _scheduler(_SyncLookupClient(engine))
    request = _request("A", cache_case.tokens[:4])

    assert _lookup(scheduler, request) > 0
    assert "A" in engine.lookup_pins
    assert not first_obj.can_evict

    # Aborted while still waiting: vLLM never allocated its blocks.
    request.status = RequestStatus.FINISHED_ABORTED
    scheduler.request_finished(request, [])

    meta = scheduler.build_connector_meta(_empty_step({"A"}))
    assert meta.unpin_lookup_ids == {"A"}

    # The worker releases the pins on that step, which runs no forward.
    worker = _CPUConnector([], engine, "sync")
    worker._parent._get_connector_metadata.return_value = meta
    worker.start_load_kv(MagicMock(attn_metadata=object()))

    assert not engine.lookup_pins
    assert first_obj.metadata.pin_count == 0
    assert first_obj.can_evict

    # Reported once: the following step carries nothing.
    assert scheduler.build_connector_meta(_empty_step(set())).unpin_lookup_ids == set()


def test_allocated_request_is_left_to_its_load(
    cache_case: _CacheCase,  # noqa: F811
) -> None:
    engine = cache_case.engine
    scheduler = _scheduler(_SyncLookupClient(engine))
    request = _request("A", cache_case.tokens[:4])

    assert _lookup(scheduler, request) > 0
    scheduler.update_state_after_alloc(request, num_external_tokens=4)

    # Finished after allocation: its load or save owns the unpin, so the
    # workers must not be told to drop pins the load may still need.
    request.status = RequestStatus.FINISHED_ABORTED
    scheduler.request_finished(request, [])
    assert scheduler.build_connector_meta(_empty_step({"A"})).unpin_lookup_ids == set()
    assert "A" in engine.lookup_pins
    engine.lookup_unpin("A")


def test_async_lookup_is_left_to_lookup_cancel(
    cache_case: _CacheCase,  # noqa: F811
) -> None:
    engine = cache_case.engine
    lookup_client = _SyncLookupClient(engine)
    lookup_client.cancel_lookup = MagicMock()
    scheduler = _scheduler(lookup_client)
    scheduler.async_loading = True
    request = _request("A", cache_case.tokens[:4])

    _lookup(scheduler, request)
    request.status = RequestStatus.FINISHED_ABORTED
    scheduler.request_finished(request, [])

    lookup_client.cancel_lookup.assert_called_once_with("A")
    assert scheduler.build_connector_meta(_empty_step({"A"})).unpin_lookup_ids == set()
    engine.lookup_unpin("A")


def test_step_with_nothing_abandoned_unpins_nothing() -> None:
    engine = MagicMock(spec=LMCacheEngine)
    worker = _CPUConnector([], engine, "sync")
    worker._parent._get_connector_metadata.return_value = LMCacheConnectorMetadata()
    worker.start_load_kv(MagicMock(attn_metadata=object()))
    engine.lookup_unpin.assert_not_called()

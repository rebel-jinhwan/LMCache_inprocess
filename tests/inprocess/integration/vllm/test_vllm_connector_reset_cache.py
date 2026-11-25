# SPDX-License-Identifier: Apache-2.0
"""``reset_cache`` clears LMCache on every worker, and lookups wait for it.

vLLM's ``reset_prefix_cache(reset_connector=True)`` calls the scheduler
connector's ``reset_cache``. The storage lives in the workers, so the clear
rides a later step's metadata, and until the workers report it every lookup
answers 0: the lookup server would still find the chunks being dropped.
"""

# Standard
from typing import Optional
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

# First Party
from lmcache.inprocess.integration.vllm.lmcache_connector_v1 import (  # noqa: E402
    LMCacheConnectorV1Dynamic,
)
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (  # noqa: E402
    LMCacheClearDone,
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
)
from lmcache.inprocess.v1.lookup_client.abstract_client import (  # noqa: E402
    LookupClientInterface,
)
from lmcache.inprocess.v1.lookup_client.lmcache_async_lookup_client import (  # noqa: E402
    LMCacheAsyncLookupClient,
)
import lmcache.inprocess.integration.vllm.vllm_v1_adapter as adapter  # noqa: E402

# Local
from .test_retrieve_pin_ownership import (  # noqa: E402, F401 -- shared fixtures
    _CacheCase,
    cache_case,
)
from .test_vllm_abandoned_sync_lookup_unpin import (  # noqa: E402
    _empty_step,
    _request,
    _scheduler,
    _SyncLookupClient,
)
from .test_vllm_sync_load_pin_cleanup import _CPUConnector  # noqa: E402

pytestmark = pytest.mark.no_shared_allocator


class _SyncClient(_SyncLookupClient, LookupClientInterface):
    """The sync fake, with the lookup client interface's reset defaults."""

    def close(self) -> None:
        return


class _AsyncLookupClient(_SyncClient):
    """An async client whose replies the test delivers by hand."""

    def __init__(self, engine) -> None:
        super().__init__(engine)
        self.awaiting: set[str] = set()
        self.discarded: list[str] = []

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        if lookup_id in self.awaiting:
            return None
        return super().lookup_cache(lookup_id)

    def lookup(self, token_ids, lookup_id, request_configs=None) -> None:
        self.awaiting.add(lookup_id)
        return None

    def reply(self, lookup_id: str, hits: int) -> None:
        self.awaiting.discard(lookup_id)
        self.status[lookup_id] = hits

    def outstanding_lookup_ids(self) -> set[str]:
        return set(self.status) | self.awaiting

    def pending_lookup_ids(self) -> set[str]:
        return set(self.awaiting)

    def discard_lookup(self, lookup_id: str) -> None:
        self.discarded.append(lookup_id)
        self.clear_lookup_status(lookup_id)


def _reset_scheduler(lookup_client, timeout_s: float = 30.0) -> LMCacheConnectorV1Impl:
    scheduler = _scheduler(lookup_client)
    scheduler._role = KVConnectorRole.SCHEDULER
    scheduler._reset_waiting_lookup_ids = set()
    scheduler._reset_unpin_lookup_ids = set()
    scheduler._reset_lookup_deadline = None
    scheduler._reset_lookup_timeout_s = timeout_s
    return scheduler


def _run_worker_step(engine, meta: LMCacheConnectorMetadata) -> Optional[object]:
    """Start one worker step on ``meta`` and return its worker metadata."""
    worker = _CPUConnector([], engine, "sync")
    worker._parent._get_connector_metadata.return_value = meta
    worker.start_load_kv(MagicMock(attn_metadata=object()))
    return worker.build_connector_worker_meta()


def _confirm(scheduler: LMCacheConnectorV1Impl, worker_meta) -> None:
    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=worker_meta)
    )


def _hits(scheduler, req_id: str, tokens: list[int]) -> Optional[int]:
    return scheduler.get_num_new_matched_tokens(
        _request(req_id, tokens), num_computed_tokens=0
    )


def test_reset_clears_workers_and_gates_lookups_until_confirmed(
    cache_case: _CacheCase,  # noqa: F811 -- pytest injects the imported fixture
) -> None:
    engine = cache_case.engine
    tokens = cache_case.tokens[:4]
    scheduler = _reset_scheduler(_SyncClient(engine))

    # A pre-reset lookup: pinned on the worker, cached on the scheduler.
    assert _hits(scheduler, "A", tokens) > 0
    assert "A" in engine.lookup_pins

    assert scheduler.reset_cache() is True
    # Gated: the lookup server would still answer from the old chunks.
    assert _hits(scheduler, "B", tokens) == 0
    assert scheduler.load_specs["B"].lmcache_cached_tokens == 0

    meta = scheduler.build_connector_meta(_empty_step(set()))
    assert meta.clear_cache
    assert "A" in meta.unpin_lookup_ids

    report = _run_worker_step(engine, meta)
    assert isinstance(report, LMCacheClearDone)
    assert not engine.lookup_pins
    assert engine.lookup(tokens, lookup_id="probe", pin=False) == 0

    # Still gated until the workers' report reaches the scheduler.
    assert _hits(scheduler, "C", tokens) == 0
    _confirm(scheduler, report)

    # A's pre-reset hit was dropped with the reset: it looks up afresh.
    assert _hits(scheduler, "A", tokens) == 0
    assert not scheduler.build_connector_meta(_empty_step(set())).clear_cache


def test_worker_reports_the_clear_once(
    cache_case: _CacheCase,  # noqa: F811
) -> None:
    worker = _CPUConnector([], cache_case.engine, "sync")
    worker._parent._get_connector_metadata.return_value = LMCacheConnectorMetadata(
        clear_cache=True
    )
    worker.start_load_kv(MagicMock(attn_metadata=object()))

    assert isinstance(worker.build_connector_worker_meta(), LMCacheClearDone)
    assert worker.build_connector_worker_meta() is None


def test_clear_waits_for_pending_async_replies(
    cache_case: _CacheCase,  # noqa: F811
) -> None:
    client = _AsyncLookupClient(cache_case.engine)
    scheduler = _reset_scheduler(client)
    scheduler.async_loading = True
    tokens = cache_case.tokens[:4]

    assert _hits(scheduler, "A", tokens) is None  # reply outstanding
    assert scheduler.reset_cache() is True

    # A's prefetch still owns buffers: no clear, and A stays unscheduled.
    assert not scheduler.build_connector_meta(_empty_step(set())).clear_cache
    assert _hits(scheduler, "A", tokens) is None
    assert _hits(scheduler, "B", tokens) == 0

    client.reply("A", 4)
    meta = scheduler.build_connector_meta(_empty_step(set()))
    assert meta.clear_cache
    assert "A" in meta.unpin_lookup_ids
    # Its late reply was forgotten, not left to resurface as a hit.
    assert "A" in client.discarded
    assert client.lookup_cache("A") == -1


def test_missing_reply_fails_the_reset_at_its_deadline(
    cache_case: _CacheCase,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _AsyncLookupClient(cache_case.engine)
    scheduler = _reset_scheduler(client, timeout_s=5.0)
    scheduler.async_loading = True
    now = [100.0]
    monkeypatch.setattr(adapter.time, "monotonic", lambda: now[0])

    _hits(scheduler, "A", cache_case.tokens[:4])
    scheduler.reset_cache()
    # A repeated reset does not extend the wait.
    now[0] = 103.0
    scheduler.reset_cache()
    assert not scheduler.build_connector_meta(_empty_step(set())).clear_cache

    now[0] = 105.0
    with pytest.raises(TimeoutError, match="'A'"):
        scheduler.build_connector_meta(_empty_step(set()))


def test_lookups_resume_after_the_last_of_overlapping_resets(
    cache_case: _CacheCase,  # noqa: F811
) -> None:
    scheduler = _reset_scheduler(_SyncClient(cache_case.engine))
    tokens = cache_case.tokens[:4]

    scheduler.reset_cache()
    assert scheduler.build_connector_meta(_empty_step(set())).clear_cache
    # Under async scheduling a second reset can flag a later step before the
    # first step's report arrives.
    scheduler.reset_cache()
    assert scheduler.build_connector_meta(_empty_step(set())).clear_cache

    _confirm(scheduler, LMCacheClearDone())
    assert _hits(scheduler, "A", tokens) == 0
    assert "A" not in scheduler.lookup_client.status  # still gated

    _confirm(scheduler, LMCacheClearDone())
    assert _hits(scheduler, "A", tokens) > 0
    cache_case.engine.lookup_unpin("A")


def test_reset_is_a_scheduler_call(cache_case: _CacheCase) -> None:  # noqa: F811
    worker = _reset_scheduler(_SyncClient(cache_case.engine))
    worker._role = KVConnectorRole.WORKER
    assert worker.reset_cache() is None


def test_reset_without_lookup_client_is_refused(
    cache_case: _CacheCase,  # noqa: F811
) -> None:
    scheduler = _reset_scheduler(_SyncClient(cache_case.engine))
    scheduler._manager.lookup_client = None
    assert scheduler.reset_cache() is False
    assert not scheduler._reset_pending


def test_dynamic_connector_delegates_reset() -> None:
    connector = object.__new__(LMCacheConnectorV1Dynamic)
    connector._lmcache_engine = MagicMock()
    connector._lmcache_engine.reset_cache.return_value = True

    assert connector.reset_cache() is True
    output = KVConnectorOutput()
    connector.update_connector_output(output)
    connector._lmcache_engine.update_connector_output.assert_called_once_with(output)
    connector.build_connector_worker_meta()
    connector._lmcache_engine.build_connector_worker_meta.assert_called_once_with()


def _async_client() -> LMCacheAsyncLookupClient:
    client = object.__new__(LMCacheAsyncLookupClient)
    client.lock = threading.Lock()
    client.reqs_status = {}
    client.first_lookup_time = {}
    client.aborted_lookups = set()
    return client


def test_async_client_reports_pending_and_outstanding_lookups() -> None:
    client = _async_client()
    client.reqs_status = {"waiting": None, "answered": 4}
    # Timed out (cancelled), then vLLM allocated the request, which cleared
    # its status: its replies are still outstanding.
    client.aborted_lookups = {"timed_out"}

    assert client.pending_lookup_ids() == {"waiting", "timed_out"}
    assert client.outstanding_lookup_ids() == {"waiting", "answered", "timed_out"}

    client.reqs_status["timed_out"] = 0  # its replies arrived
    assert client.pending_lookup_ids() == {"waiting"}

    client.discard_lookup("timed_out")
    assert "timed_out" not in client.aborted_lookups
    assert "timed_out" not in client.reqs_status


def test_wrapped_clients_forward_the_reset_api() -> None:
    # First Party
    from lmcache.inprocess.v1.lookup_client.hit_limit_lookup_client import (
        HitLimitLookupClient,
    )

    inner = _async_client()
    inner.reqs_status = {"A": None}
    wrapper = object.__new__(HitLimitLookupClient)
    wrapper.actual_lookup_client = inner

    assert wrapper.pending_lookup_ids() == {"A"}
    assert wrapper.outstanding_lookup_ids() == {"A"}
    wrapper.discard_lookup("A")
    assert inner.reqs_status == {}


def test_metadata_defaults_to_no_clear() -> None:
    assert LMCacheConnectorMetadata().clear_cache is False

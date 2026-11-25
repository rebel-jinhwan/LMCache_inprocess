# SPDX-License-Identifier: Apache-2.0
"""Tests for RBLN connector integration (clean, no monkeypatch)."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest


class TestRBLNConnectorImport:
    def test_connector_importable(self):
        """RBLNLMCacheConnectorV1 should be importable."""
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1,
        )

        assert RBLNLMCacheConnectorV1 is not None

    def test_connector_impl_importable(self):
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1Impl,
        )

        assert RBLNLMCacheConnectorV1Impl is not None

    def test_service_factory_importable(self):
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import (
            RBLNServiceFactory,
        )

        assert RBLNServiceFactory is not None


class TestRBLNConnectorV1Impl:
    def test_is_subclass_of_upstream_impl(self):
        """Should extend upstream LMCacheConnectorV1Impl."""
        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorV1Impl,
        )
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1Impl,
        )

        assert issubclass(RBLNLMCacheConnectorV1Impl, LMCacheConnectorV1Impl)

    def test_overrides_init(self):
        """Should override __init__ to use RBLNServiceFactory."""
        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorV1Impl,
        )
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1Impl,
        )

        assert (
            RBLNLMCacheConnectorV1Impl.__init__ is not LMCacheConnectorV1Impl.__init__
        )


class TestRBLNConnectorV1:
    def test_is_subclass_of_kv_connector_base(self):
        """Should extend vLLM's KVConnectorBase_V1."""
        # Third Party
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

        # First Party
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1,
        )

        assert issubclass(RBLNLMCacheConnectorV1, KVConnectorBase_V1)

    def test_pp_aware_handshake_metadata_is_noop(self):
        """LMCache doesn't do PP-disaggregated transfer, so pp_rank > 0
        handshake metadata must not raise (base class default raises)."""
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1,
        )

        connector = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
        connector.set_xfer_handshake_metadata_pp_aware({(1, 0): MagicMock()})

    def test_has_all_required_methods(self):
        """Should have all KVConnectorBase_V1 interface methods."""
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1,
        )

        required_methods = [
            "register_kv_caches",
            "start_load_kv",
            "wait_for_layer_load",
            "save_kv_layer",
            "wait_for_save",
            "get_finished",
            "shutdown",
            "get_num_new_matched_tokens",
            "update_state_after_alloc",
            "build_connector_meta",
            "request_finished",
            # KV event bridge (worker -> scheduler -> kv_event_publisher)
            "get_kv_connector_kv_cache_events",
            "update_connector_output",
            "take_events",
        ]
        for method in required_methods:
            assert hasattr(RBLNLMCacheConnectorV1, method), f"Missing method: {method}"

    @pytest.mark.parametrize("copies", [1, 2, 3])
    def test_kv_events_round_trip_worker_to_scheduler(self, copies):
        """Worker side turns the engine's CacheStoreEvents + RDS evictions into
        BlockStored/BlockRemoved; scheduler side hands them to take_events."""
        # Third Party
        from vllm.distributed.kv_events import BlockRemoved, BlockStored
        from vllm.v1.outputs import KVConnectorOutput

        # First Party
        from lmcache.inprocess.rbln._rds_plugin import (
            RDS_KV_EVENT_MEDIUM as KV_EVENT_MEDIUM,
        )
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1,
        )
        from lmcache.utils import CacheStoreEvent

        stored = CacheStoreEvent(
            block_hashes=[7],
            parent_block_hash=3,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
            medium=KV_EVENT_MEDIUM,
            lora_name=None,
        )
        rds = MagicMock()
        type(rds).__name__ = "RDSBackend"
        rds.take_evicted_hashes.return_value = [5]
        engine = MagicMock()
        engine.kv_events_enabled = True
        engine.save_only_first_rank = False
        # Multiple requests can save an identical prefix in one worker step.
        engine.get_kv_events.return_value = [stored] * copies
        engine.storage_manager.storage_backends = {"RDSBackend": rds}

        worker = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
        worker._lmcache_engine = MagicMock(lmcache_engine=engine)
        kv_events = worker.get_kv_connector_kv_cache_events()
        engine.get_kv_events.assert_called_once()
        rds.take_evicted_hashes.assert_called_once()
        assert kv_events is not None

        scheduler = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
        scheduler._kv_cache_events = None
        scheduler.update_connector_output(KVConnectorOutput(kv_cache_events=kv_events))
        events = list(scheduler.take_events())
        assert events == [
            BlockStored(
                block_hashes=[7],
                parent_block_hash=3,
                token_ids=[1, 2],
                block_size=2,
                lora_id=None,
                medium=KV_EVENT_MEDIUM,
                lora_name=None,
            ),
            BlockRemoved(block_hashes=[5], medium=KV_EVENT_MEDIUM),
        ]
        # Drained: nothing left for the next step.
        assert list(scheduler.take_events()) == []
        # No engine (scheduler role) -> no events, no crash (before any TP
        # group lookup).
        worker._lmcache_engine = MagicMock(lmcache_engine=None)
        assert worker.get_kv_connector_kv_cache_events() is None


class TestRBLNConnectorResetCache:
    """vLLM ``reset_prefix_cache(reset_connector=True)`` -> scheduler
    ``reset_cache`` -> next step's metadata -> worker ``LMCacheEngine.clear()``
    -> ``build_connector_worker_meta`` -> scheduler lifts the lookup gate."""

    @staticmethod
    def _connector(role):
        # Third Party
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )

        # First Party
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1,
        )

        conn = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
        conn._role = getattr(KVConnectorRole, role)
        conn._lmcache_engine = MagicMock(name="impl")
        conn._kv_cache_events = None
        conn._reset_pending = conn._reset_ran = False
        conn._reset_inflight = 0
        conn._lookup_req_ids = set()
        conn._async_lookup_req_ids = set()
        conn._reset_waiting_lookup_ids = set()
        conn._reset_unpin_req_ids = set()
        conn._reset_lookup_timeout_s = 30.0
        conn._reset_lookup_deadline = None
        conn._abandoned_lookup_ids = set()
        return conn

    def test_scheduler_round_trip(self):
        """reset_cache drops cached pre-reset lookups and holds lookups at 0
        until the workers confirm; the clear rides exactly one step's meta."""
        # Third Party
        from vllm.v1.outputs import KVConnectorOutput

        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
            LoadSpec,
        )
        from lmcache.inprocess.rbln.integration.vllm.connector import _LMCacheClearDone

        sched = self._connector("SCHEDULER")
        impl = sched._lmcache_engine
        impl.load_specs = {"waiting-a": object(), "waiting-b": object()}
        impl.get_num_new_matched_tokens.return_value = 1024
        impl.build_connector_meta.side_effect = lambda _: LMCacheConnectorMetadata()
        request = MagicMock(request_id="waiting-a")

        assert sched.get_num_new_matched_tokens(request, 0) == (1024, False)
        assert sched.reset_cache() is True
        assert impl.load_specs == {}
        cleared = sorted(
            c.args[0] for c in impl.lookup_client.clear_lookup_status.call_args_list
        )
        assert cleared == ["waiting-a", "waiting-b"]

        impl.get_num_new_matched_tokens.reset_mock()
        # Pending: gated without asking LMCache. The request keeps upstream's
        # lookup-miss load spec (vLLM-computed tokens, 0 from LMCache).
        request.request_id = "gated"
        assert sched.get_num_new_matched_tokens(request, 16) == (0, False)
        impl.get_num_new_matched_tokens.assert_not_called()
        assert impl.load_specs["gated"] == LoadSpec(16, 0, False)
        mock_req = MagicMock(request_id="mock_req_0")
        assert sched.get_num_new_matched_tokens(mock_req, 0) == (0, False)
        assert "mock_req_0" not in impl.load_specs
        meta = sched.build_connector_meta(MagicMock())
        assert meta.clear_lmcache is True
        assert meta.lmcache_rbln_reset_lookup_ids == {"waiting-a", "waiting-b"}
        assert not getattr(
            sched.build_connector_meta(MagicMock()), "clear_lmcache", False
        )
        # In flight (sent, not confirmed): still gated, even across unrelated outputs.
        sched.update_connector_output(KVConnectorOutput())
        assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
        # Confirmed: lookups resume.
        sched.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=_LMCacheClearDone())
        )
        assert sched.get_num_new_matched_tokens(request, 0) == (1024, False)
        impl.get_num_new_matched_tokens.assert_called_once()

    def test_worker_clears_on_bind_and_reports_once(self):
        """The worker clears when it binds the flagged metadata — before the
        step's loads — and reports it once; unflagged steps do nothing."""
        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )
        from lmcache.inprocess.rbln.integration.vllm.connector import _LMCacheClearDone

        worker = self._connector("WORKER")
        engine = worker._lmcache_engine.lmcache_engine
        engine.clear.return_value = 3072
        assert worker.reset_cache() is None  # worker role: not ours to answer

        worker.bind_connector_metadata(LMCacheConnectorMetadata())
        engine.clear.assert_not_called()
        assert worker.build_connector_worker_meta() is None

        flagged = LMCacheConnectorMetadata()
        flagged.clear_lmcache = True  # type: ignore[attr-defined]
        worker.bind_connector_metadata(flagged)
        engine.clear.assert_called_once_with()
        assert worker._get_connector_metadata() is flagged
        assert isinstance(worker.build_connector_worker_meta(), _LMCacheClearDone)
        assert worker.build_connector_worker_meta() is None

    def test_worker_meta_aggregates_across_ranks(self):
        """Every rank reports the same fact; aggregation keeps one."""
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.connector import _LMCacheClearDone

        a, b = _LMCacheClearDone(), _LMCacheClearDone()
        assert a.aggregate(b) is a

    def test_overlapping_resets_keep_gate_until_last_confirmation(self):
        """Async scheduling: reset#1 flags step A, reset#2 flags step B before
        A is confirmed. A's confirmation alone must not reopen lookups — B's
        clear has not run yet."""
        # Third Party
        from vllm.v1.outputs import KVConnectorOutput

        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )
        from lmcache.inprocess.rbln.integration.vllm.connector import _LMCacheClearDone

        sched = self._connector("SCHEDULER")
        impl = sched._lmcache_engine
        impl.load_specs = {}
        impl.get_num_new_matched_tokens.return_value = 1024
        impl.build_connector_meta.side_effect = lambda _: LMCacheConnectorMetadata()
        request = MagicMock(request_id="req")
        done = KVConnectorOutput(kv_connector_worker_meta=_LMCacheClearDone())

        assert sched.reset_cache() is True
        assert sched.build_connector_meta(MagicMock()).clear_lmcache is True  # A
        assert sched.reset_cache() is True
        assert sched.build_connector_meta(MagicMock()).clear_lmcache is True  # B

        sched.update_connector_output(done)  # A confirmed, B still in flight
        assert sched.get_num_new_matched_tokens(request, 0) == (0, False)
        sched.update_connector_output(done)  # B confirmed
        assert sched.get_num_new_matched_tokens(request, 0) == (1024, False)
        # A stray extra confirmation cannot drive the count negative.
        sched.update_connector_output(done)
        assert sched._reset_inflight == 0

    @pytest.mark.parametrize("async_prefetch", [False, True])
    def test_reset_unpins_real_cpu_cache_before_clear(self, async_prefetch):
        """A lookup hit waiting for allocation must not survive CPU clear."""
        # Standard
        from concurrent.futures import Future
        import threading

        # Third Party
        import torch

        # First Party
        from lmcache.inprocess.v1.cache_engine import LMCacheEngine
        from lmcache.inprocess.v1.event_manager import EventStatus
        from lmcache.inprocess.v1.storage_backend.local_cpu_backend import (
            LocalCPUBackend,
        )
        from lmcache.inprocess.v1.storage_backend.storage_manager import StorageManager
        from lmcache.v1.memory_management import (
            MemoryFormat,
            MemoryObjMetadata,
            TensorMemoryObj,
        )

        # Real pin/unpin, backend removal, storage manager and engine clear.
        obj = TensorMemoryObj(
            torch.zeros(8),
            MemoryObjMetadata(
                torch.Size([2, 1, 4, 1]),
                torch.float32,
                0,
                32,
                1,
                fmt=MemoryFormat.KV_2LTD,
            ),
            None,
        )
        obj.pin()
        cpu = LocalCPUBackend.__new__(LocalCPUBackend)
        cpu.use_hot = True
        cpu.cpu_lock = threading.Lock()
        cpu.hot_cache = {"old": obj}
        cpu.cache_policy = MagicMock()
        cpu.batched_msg_sender = None
        manager = StorageManager.__new__(StorageManager)
        manager.storage_backends = {"LocalCPUBackend": cpu}
        engine = LMCacheEngine.__new__(LMCacheEngine)
        engine.save_only_first_rank = False
        engine.storage_manager = manager
        engine.lookup_pins = {"waiting": {"LocalCPUBackend": ["old"]}}
        engine.async_loading = None
        if async_prefetch:
            # Async prefetch owns both a pin and an extra reference. Clear
            # must release both, using the engine's completed-event cleanup.
            obj.ref_count_up()
            engine.lookup_pins = {}
            engine.async_loading = object()
            engine.event_manager = MagicMock()
            engine.event_manager.get_event_status.return_value = EventStatus.DONE
            future = Future()
            future.set_result([[("old", obj)]])
            engine.event_manager.pop_event.return_value = future

        worker = self._connector("WORKER")
        worker._lmcache_engine.lmcache_engine = engine
        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        meta = LMCacheConnectorMetadata()
        meta.clear_lmcache = True
        meta.lmcache_rbln_reset_lookup_ids = {"waiting"}
        worker.bind_connector_metadata(meta)
        assert cpu.hot_cache == {}
        assert engine.lookup_pins == {}
        assert worker.build_connector_worker_meta() is not None

    @pytest.mark.parametrize("lifecycle", ["waiting", "allocated", "finished"])
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_reset_waits_for_timed_out_prefetch(self, lifecycle, wrapped):
        """Timeout misses still own prefetches, even after allocation/abort."""
        # Standard
        from types import SimpleNamespace
        import threading
        import time

        # Third Party
        from vllm.v1.outputs import KVConnectorOutput

        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
            LoadSpec,
        )
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1Impl,
            _LMCacheClearDone,
        )
        from lmcache.inprocess.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupClient,
        )

        sched = self._connector("SCHEDULER")
        impl = sched._lmcache_engine
        actual = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
        actual.lock = threading.Lock()
        actual.reqs_status = {"slow": None}
        actual.first_lookup_time = {"slow": time.time()}
        actual.aborted_lookups = set()
        actual.lookup_backoff_time = 0
        actual.config = SimpleNamespace(lookup_timeout_ms=1000)
        actual.world_size = 1
        actual.push_sockets = [MagicMock()]
        client = actual
        if wrapped:
            client = SimpleNamespace(
                actual_lookup_client=actual,
                lookup_cache=actual.lookup_cache,
                clear_lookup_status=actual.clear_lookup_status,
                cancel_lookup=actual.cancel_lookup,
            )
        # Exercise the production property that protects upstream's status
        # clearing, not a mock implementation of clear_lookup_status.
        owner = RBLNLMCacheConnectorV1Impl.__new__(RBLNLMCacheConnectorV1Impl)
        owner._manager = SimpleNamespace(lookup_client=client)
        owner._use_grouped_slots = False
        owner._unfinished_requests = {}
        impl.lookup_client = owner.lookup_client
        impl.load_specs = owner.load_specs = {}
        impl.get_num_new_matched_tokens.side_effect = (
            lambda req, _: impl.lookup_client.lookup_cache(req.request_id)
        )
        impl.build_connector_meta.side_effect = lambda _: LMCacheConnectorMetadata()
        req = SimpleNamespace(request_id="slow")
        assert sched.get_num_new_matched_tokens(req, 0) == (None, False)
        actual.first_lookup_time["slow"] = time.time() - 2
        # Recompute remains allowed: only reset must wait for the real reply.
        assert sched.get_num_new_matched_tokens(req, 0) == (0, False)
        assert actual.reqs_status["slow"] is None
        assert actual.aborted_lookups == {"slow"}
        # A scheduler retry/preempt-resume must keep returning the timeout
        # miss without asking upstream to read its removed timeout timer.
        assert sched.get_num_new_matched_tokens(req, 0) == (0, False)

        if lifecycle == "allocated":
            impl.update_state_after_alloc.side_effect = owner.update_state_after_alloc
            sched.update_state_after_alloc(req, MagicMock(), 0)
            assert sched.get_num_new_matched_tokens(req, 0) == (0, False)
        elif lifecycle == "finished":
            impl.request_finished.side_effect = lambda req, _: (
                impl.lookup_client.cancel_lookup(req.request_id),
                None,
            )
            sched.request_finished_all_groups(req, ([],))
        # Consuming its load spec/finishing the request must not lose ownership.
        impl.load_specs.clear()
        sched.build_connector_meta(MagicMock())
        sched.reset_cache()
        meta = sched.build_connector_meta(MagicMock())
        assert not getattr(meta, "clear_lmcache", False)
        assert actual.reqs_status == {"slow": None}
        # Other lookups run abort cleanup; they must not mistake a cleared
        # pending slot for completion and release prefetch references early.
        assert impl.lookup_client.lookup_cache("other") == -1
        actual.push_sockets[0].send.assert_not_called()

        actual.reqs_status["slow"] = 1024
        assert sched.get_num_new_matched_tokens(req, 0) == (0, False)
        assert impl.load_specs["slow"] == LoadSpec(0, 0, False)
        meta = sched.build_connector_meta(MagicMock())
        assert meta.clear_lmcache is True
        assert meta.lmcache_rbln_reset_lookup_ids == {"slow"}
        assert "slow" not in actual.reqs_status
        assert "slow" not in actual.aborted_lookups

        worker = self._connector("WORKER")
        worker._lmcache_engine.lmcache_engine.clear.return_value = 1024
        worker.bind_connector_metadata(meta)
        worker._lmcache_engine.lmcache_engine.lookup_unpin.assert_called_once_with(
            "slow"
        )
        worker._lmcache_engine.lmcache_engine.clear.assert_called_once_with()
        assert isinstance(worker.build_connector_worker_meta(), _LMCacheClearDone)
        sched.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=_LMCacheClearDone())
        )
        assert sched._reset_inflight == 0

    @pytest.mark.parametrize("entrypoint", ["lookup", "metadata"])
    def test_missing_reset_reply_fails_at_deadline(self, monkeypatch, entrypoint):
        """A lost reply fails explicitly without clearing a running prefetch."""
        # Standard
        from types import SimpleNamespace
        import threading

        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )
        from lmcache.inprocess.rbln.integration.vllm import connector
        from lmcache.inprocess.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupClient,
        )

        now = [100.0]
        monkeypatch.setattr(connector.time, "monotonic", lambda: now[0])
        sched = self._connector("SCHEDULER")
        actual = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
        actual.lock = threading.Lock()
        actual.reqs_status = {"lost": None}
        actual.first_lookup_time = {"lost": 0}
        actual.aborted_lookups = {"lost"}
        impl = sched._lmcache_engine
        impl.lookup_client = actual
        impl.load_specs = {}
        impl.build_connector_meta.side_effect = lambda _: LMCacheConnectorMetadata()
        sched._async_lookup_req_ids = {"lost"}
        req = SimpleNamespace(request_id="lost")

        sched.reset_cache()
        assert sched._reset_lookup_deadline == 130.0
        assert sched.get_num_new_matched_tokens(req, 0) == (None, False)
        now[0] = 129.0
        sched.reset_cache()
        assert sched._reset_lookup_deadline == 130.0  # no deadline extension
        assert not getattr(
            sched.build_connector_meta(MagicMock()), "clear_lmcache", False
        )
        now[0] = 130.0
        with patch.object(connector.logger, "error") as log:
            with pytest.raises(TimeoutError, match="lost.*restart the engine"):
                if entrypoint == "lookup":
                    sched.get_num_new_matched_tokens(req, 0)
                else:
                    sched.build_connector_meta(MagicMock())
            assert "lost" in log.call_args.args[0]
        assert actual.reqs_status == {"lost": None}
        assert actual.aborted_lookups == {"lost"}
        assert sched._reset_pending
        assert sched._reset_inflight == 0
        impl.lmcache_engine.lookup_unpin.assert_not_called()
        impl.lmcache_engine.clear.assert_not_called()

    def test_timeout_resume_preserves_prefetch_until_normal_cleanup(self):
        """Without reset, a resumed timeout miss waits for safe abort cleanup."""
        # Standard
        from types import SimpleNamespace
        import threading
        import time

        # First Party
        from lmcache.inprocess.rbln.integration.vllm.connector import (
            RBLNLMCacheConnectorV1Impl,
        )
        from lmcache.inprocess.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupClient,
        )

        actual = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
        actual.lock = threading.Lock()
        actual.reqs_status = {"preempted": None}
        actual.first_lookup_time = {"preempted": time.time() - 2}
        actual.aborted_lookups = set()
        actual.lookup_backoff_time = 0
        actual.config = SimpleNamespace(lookup_timeout_ms=1000)
        actual.world_size = 1
        actual.push_sockets = [MagicMock()]
        owner = RBLNLMCacheConnectorV1Impl.__new__(RBLNLMCacheConnectorV1Impl)
        owner._manager = SimpleNamespace(lookup_client=actual)
        client = owner.lookup_client

        assert client.lookup_cache("preempted") == 0
        client.clear_lookup_status("preempted")  # allocation
        assert client.lookup_cache("preempted") == 0  # preempt/resume
        assert client.lookup_cache("preempted") == 0  # repeated scheduler poll
        assert actual.reqs_status == {"preempted": None}
        assert "preempted" not in actual.first_lookup_time
        actual.push_sockets[0].send.assert_not_called()

        # Once the reply arrives, upstream's ordinary cleanup releases the
        # old task before a new lookup can reuse this request ID.
        actual.reqs_status["preempted"] = 1024
        assert client.lookup_cache("preempted") == -1
        actual.push_sockets[0].send.assert_called_once()
        assert actual.aborted_lookups == set()
        assert actual.reqs_status == {"preempted": None}
        assert "preempted" in actual.first_lookup_time

    def test_reply_at_reset_deadline_can_complete(self, monkeypatch):
        """A completed lookup needs no forced cancellation, even at the deadline."""
        # Standard
        from types import SimpleNamespace
        import threading

        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )
        from lmcache.inprocess.rbln.integration.vllm import connector
        from lmcache.inprocess.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupClient,
        )

        now = [100.0]
        monkeypatch.setattr(connector.time, "monotonic", lambda: now[0])
        sched = self._connector("SCHEDULER")
        actual = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
        actual.lock = threading.Lock()
        actual.reqs_status = {"slow": None}
        actual.first_lookup_time = {"slow": 0}
        actual.aborted_lookups = {"slow"}
        impl = sched._lmcache_engine
        impl.lookup_client = SimpleNamespace(
            actual_lookup_client=actual,
            clear_lookup_status=actual.clear_lookup_status,
        )
        impl.load_specs = {}
        impl.build_connector_meta.side_effect = lambda _: LMCacheConnectorMetadata()
        sched._async_lookup_req_ids = {"slow"}
        sched.reset_cache()
        now[0] = 130.0
        actual.reqs_status["slow"] = 1024
        meta = sched.build_connector_meta(MagicMock())
        assert meta.clear_lmcache
        assert meta.lmcache_rbln_reset_lookup_ids == {"slow"}
        assert sched._reset_lookup_deadline is None
        assert actual.reqs_status == {}
        assert actual.aborted_lookups == set()

    def test_reset_waits_for_async_replies_without_a_load_spec(self):
        """Late replies cannot revive a hit, including through client wrappers."""
        # Standard
        from types import SimpleNamespace
        import threading

        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )
        from lmcache.inprocess.v1.lookup_client.lmcache_async_lookup_client import (
            LMCacheAsyncLookupClient,
        )

        sched = self._connector("SCHEDULER")
        impl = sched._lmcache_engine
        actual = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
        actual.lock = threading.Lock()
        actual.reqs_status = {"waiting": None}
        actual.first_lookup_time = {"waiting": 0}
        impl.lookup_client = SimpleNamespace(
            actual_lookup_client=actual,
            clear_lookup_status=actual.clear_lookup_status,
        )
        impl.load_specs = {}
        impl.get_num_new_matched_tokens.return_value = None
        impl.build_connector_meta.side_effect = lambda _: LMCacheConnectorMetadata()
        req = SimpleNamespace(request_id="waiting")
        assert sched.get_num_new_matched_tokens(req, 0) == (None, False)
        assert impl.load_specs == {}
        sched.reset_cache()
        assert sched.get_num_new_matched_tokens(req, 0) == (None, False)
        meta = sched.build_connector_meta(MagicMock())
        assert not getattr(meta, "clear_lmcache", False)
        assert actual.reqs_status == {"waiting": None}

        # The response arrives after reset was requested. A real async client's
        # timeout miss must not be mistaken for completion (we read under lock).
        actual.reqs_status["waiting"] = 1024
        assert sched.get_num_new_matched_tokens(req, 0) == (0, False)
        assert actual.reqs_status == {}
        meta = sched.build_connector_meta(MagicMock())
        assert meta.lmcache_rbln_reset_lookup_ids == {"waiting"}
        assert meta.clear_lmcache is True
        # No repeated old lookup while waiting for worker acknowledgement.
        impl.get_num_new_matched_tokens.assert_called_once()


class TestAbandonedLookupPins:
    """A synchronous lookup pins its hit chunks on every worker; the load that
    follows block allocation unpins them. A request that finishes in between
    (aborted while waiting for blocks) never loads, and since LMCache 0.5.7
    dropped its pin timeout nothing else unpins: the chunks stay unevictable
    until a cache reset."""

    @staticmethod
    def _scheduler(async_loading: bool = False):
        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        sched = TestRBLNConnectorResetCache._connector("SCHEDULER")
        impl = sched._lmcache_engine
        impl.async_loading = async_loading
        impl._request_blocks = {}
        impl.get_num_new_matched_tokens.return_value = 1024
        impl.request_finished.return_value = (False, None)
        impl.build_connector_meta.side_effect = lambda _: LMCacheConnectorMetadata()
        return sched

    @staticmethod
    def _unpinned(meta) -> set:
        return set(getattr(meta, "lmcache_rbln_unpin_lookup_ids", ()))

    def test_a_request_finished_before_allocation_is_unpinned(self):
        sched = self._scheduler()
        request = MagicMock(request_id="aborted")
        assert sched.get_num_new_matched_tokens(request, 0) == (1024, False)
        sched.request_finished_all_groups(request, ([],))

        meta = sched.build_connector_meta(MagicMock())
        assert self._unpinned(meta) == {"aborted"}
        assert not getattr(meta, "clear_lmcache", False)  # not a reset
        assert self._unpinned(sched.build_connector_meta(MagicMock())) == set()

        worker = TestRBLNConnectorResetCache._connector("WORKER")
        engine = worker._lmcache_engine.lmcache_engine
        worker.bind_connector_metadata(meta)
        engine.lookup_unpin.assert_called_once_with("aborted")
        engine.clear.assert_not_called()

    def test_an_allocated_request_is_left_to_its_load(self):
        sched = self._scheduler()
        request = MagicMock(request_id="loaded")
        sched.get_num_new_matched_tokens(request, 0)
        sched.update_state_after_alloc(request, MagicMock(), 1024)
        sched.request_finished_all_groups(request, ([],))

        assert self._unpinned(sched.build_connector_meta(MagicMock())) == set()

    def test_an_async_lookup_is_left_to_upstream_cancel(self):
        sched = self._scheduler(async_loading=True)
        request = MagicMock(request_id="async")
        sched.get_num_new_matched_tokens(request, 0)
        sched.request_finished_all_groups(request, ([],))

        assert self._unpinned(sched.build_connector_meta(MagicMock())) == set()

    def test_the_worker_unpins_before_a_reset_clears(self):
        """``clear`` skips pinned objects, so a reset in the same step must
        see these chunks unpinned."""
        # First Party
        from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        worker = TestRBLNConnectorResetCache._connector("WORKER")
        engine = worker._lmcache_engine.lmcache_engine
        engine.clear.return_value = 0
        meta = LMCacheConnectorMetadata()
        meta.lmcache_rbln_unpin_lookup_ids = {"aborted"}  # type: ignore[attr-defined]
        meta.clear_lmcache = True  # type: ignore[attr-defined]
        worker.bind_connector_metadata(meta)

        assert [c[0] for c in engine.method_calls] == ["lookup_unpin", "clear"]


class TestRBLNCalculateLocalRank:
    def _make_mock_vllm_config(self, rank=0, world_size=1, tp_size=1, pp_size=1):
        config = MagicMock()
        config.parallel_config.rank = rank
        config.parallel_config.world_size = world_size
        config.parallel_config.tensor_parallel_size = tp_size
        config.parallel_config.pipeline_parallel_size = pp_size
        return config

    def test_single_device(self):
        """Single device should return (0, 1)."""
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import (
            _rbln_calculate_local_rank_and_world_size,
        )

        config = self._make_mock_vllm_config()
        local_rank, local_world_size = _rbln_calculate_local_rank_and_world_size(config)
        assert local_rank == 0
        assert local_world_size == 1

    def test_multi_device(self):
        """Multiple devices should return correct rank."""
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import (
            _rbln_calculate_local_rank_and_world_size,
        )

        with patch(
            "lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter.physical_device_count",
            return_value=4,
        ):
            config = self._make_mock_vllm_config(
                rank=2, world_size=4, tp_size=4, pp_size=1
            )
            local_rank, local_world_size = _rbln_calculate_local_rank_and_world_size(
                config
            )
            assert local_rank == 2
            assert local_world_size == 4

    def test_world_fits_on_one_node_keeps_global_rank(self):
        """A world that fits the host's NPUs keeps the global rank/world size.

        This is the branch a broken device count silently skips: with the count
        stuck at 1 the ``else`` branch runs instead and returns intra-node
        values, which differ from these whenever ``pp_size > 1``.
        """
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import (
            _rbln_calculate_local_rank_and_world_size,
        )

        with patch(
            "lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter.physical_device_count",
            return_value=8,
        ):
            config = self._make_mock_vllm_config(
                rank=5, world_size=8, tp_size=4, pp_size=2
            )
            local_rank, local_world_size = _rbln_calculate_local_rank_and_world_size(
                config
            )
            assert local_rank == 5
            assert local_world_size == 8

    def test_device_count_unavailable_defaults_to_1(self):
        """If the NPU count can't be read, should default to 1 device."""
        # First Party
        from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import (
            _rbln_calculate_local_rank_and_world_size,
        )

        with patch(
            "lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter.physical_device_count",
            side_effect=RuntimeError("no runtime"),
        ):
            config = self._make_mock_vllm_config()
            local_rank, local_world_size = _rbln_calculate_local_rank_and_world_size(
                config
            )
        assert local_rank == 0
        assert local_world_size == 1


class TestNoMonkeypatch:
    """Verify that importing lmcache.inprocess.rbln does NOT monkeypatch upstream."""

    def test_import_does_not_patch_upstream_manager(self):
        """Importing the package should not modify upstream LMCacheManager."""
        # First Party
        import lmcache.inprocess.rbln  # noqa: F401
        import lmcache.inprocess.v1.manager as manager_module

        # upstream LMCacheManager should NOT be replaced
        assert manager_module.LMCacheManager.__module__ == manager_module.__name__


class TestCpuSlabHugeHost:
    """``_patch_cpu_slab_huge_host`` routes LMCache's CPU slab through huge host memory.

    The cases it does not own -- shm, zero size, a failed allocation -- must
    fall through, and its free must not hand upstream's allocator an address it
    never handed out.
    """

    @staticmethod
    def _patched_pair():
        """Apply the patch over mocked upstream globals; return ``(alloc, free, orig)``.

        Patched in place rather than via ``sys.modules``: ``import a.b.c as x``
        walks attributes from the real parent package. The replacements are read
        back inside the patch and outlive it as closures over the mocks.
        """
        # Standard
        from types import SimpleNamespace

        # First Party
        from lmcache.inprocess.rbln._patches import _patch_cpu_slab_huge_host
        import lmcache.v1.memory_management as mm

        orig = SimpleNamespace(
            alloc=MagicMock(name="_orig_allocate_cpu_memory"),
            free=MagicMock(name="_orig_free_cpu_memory"),
        )
        with (
            patch.object(mm, "_allocate_cpu_memory", orig.alloc),
            patch.object(mm, "_free_cpu_memory", orig.free),
        ):
            _patch_cpu_slab_huge_host()
            return mm._allocate_cpu_memory, mm._free_cpu_memory, orig

    def test_alloc_returns_zeroed_huge_buffer(self):
        # Third Party
        import torch

        alloc, _, orig = self._patched_pair()
        size = 2 * 1024 * 1024
        buf = alloc(size)
        assert buf.dtype == torch.uint8
        assert buf.numel() == size
        # Huge-page aligned and prefaulted.
        assert buf.data_ptr() % (2 * 1024 * 1024) == 0
        assert bool((buf == 0).all())
        orig.alloc.assert_not_called()

    def test_zero_size_and_shm_fall_through(self):
        alloc, _, orig = self._patched_pair()
        alloc(0)
        alloc(4096, None, "some-shm")
        assert orig.alloc.call_count == 2

    def test_alloc_failure_falls_back_to_upstream(self):
        # Third Party
        import torch_rbln.memory

        # Patch first: the replacement closes over ``huge_host_empty``.
        with patch.object(
            torch_rbln.memory,
            "huge_host_empty",
            side_effect=MemoryError("no host memory"),
        ):
            alloc, _, orig = self._patched_pair()
            alloc(4096)
        orig.alloc.assert_called_once()

    def test_free_skips_upstream_for_our_buffer(self):
        alloc, free, orig = self._patched_pair()
        free(alloc(4096))
        orig.free.assert_not_called()

    def test_free_delegates_for_pageable_buffer(self):
        # Third Party
        import torch

        _, free, orig = self._patched_pair()
        free(torch.empty(4096, dtype=torch.uint8))
        orig.free.assert_called_once()

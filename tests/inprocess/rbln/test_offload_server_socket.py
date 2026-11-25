# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the offload server's IPC socket key.

``RBLNServiceFactory`` overrides ``maybe_create_offload_server`` so the offload
server's IPC socket is keyed by global rank. Upstream keys it by TP rank only,
so with TP=1 every PP rank on a host binds the same
``..._service_offload_lmcache_rpc_port_100`` path and races on ``bind``
(libzmq unlinks-then-binds; the loser gets ``EADDRINUSE`` and its
LMCacheEngine is marked init-failed).
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Event
from unittest.mock import MagicMock, patch
import sys
import time

# Third Party
import msgspec
import pytest
import zmq

# First Party
from lmcache.inprocess.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import RBLNServiceFactory
from lmcache.inprocess.v1.offload_server.message import OffloadMsg, OffloadRetMsg
from lmcache.inprocess.v1.rpc_utils import get_zmq_rpc_path_lmcache

# Where each factory looks the server class up. Upstream's is taken from the
# class's own module, so a module move upstream does not break the patch.
RBLN_ZMQ_SERVER = (
    "lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter"
    ".InterruptibleZMQOffloadServer"
)
UPSTREAM_FACTORY_MODULE = sys.modules[VllmServiceFactory.__module__]
TP_RANK = "vllm.distributed.parallel_state.get_tensor_model_parallel_rank"


def _make_factory(
    role: str, worker_id: int, cls: type = RBLNServiceFactory
) -> VllmServiceFactory:
    factory = cls(MagicMock(), MagicMock(), role)
    factory.metadata = MagicMock()
    factory.metadata.worker_id = worker_id
    factory.metadata.engine_id = "e"
    factory.lmcache_engine = MagicMock()
    factory.lmcache_engine.metadata = factory.metadata
    return factory


class TestOffloadServerRankKey:
    def test_scheduler_gets_none(self) -> None:
        with patch(RBLN_ZMQ_SERVER) as server_cls:
            assert _make_factory("scheduler", 0).maybe_create_offload_server() is None
        server_cls.assert_not_called()

    @pytest.mark.parametrize("worker_id", [0, 1, 2, 3])
    def test_worker_binds_with_global_rank(self, worker_id: int) -> None:
        """PP rank *worker_id* with TP=1 has tp_rank 0; the socket key must
        still be the global rank."""
        factory = _make_factory("worker", worker_id)
        with patch(RBLN_ZMQ_SERVER) as server_cls, patch(TP_RANK, return_value=0):
            server = factory.maybe_create_offload_server()
        server_cls.assert_called_once_with(factory.lmcache_engine, worker_id)
        assert server is server_cls.return_value

    def test_upstream_still_keys_by_tp_rank(self) -> None:
        """The upstream behavior the override works around, which also shows
        importing ``lmcache.inprocess.rbln`` leaves the upstream factory alone. If this
        fails, upstream keys by global rank: delete the override."""
        factory = _make_factory("worker", 3, VllmServiceFactory)
        with (
            patch.object(UPSTREAM_FACTORY_MODULE, "ZMQOffloadServer") as server_cls,
            patch(TP_RANK, return_value=0),
        ):
            factory.maybe_create_offload_server()
        server_cls.assert_called_once_with(factory.lmcache_engine, 0)

    def test_pp_ranks_get_distinct_paths(self, tmp_path) -> None:
        """Four PP ranks on one host -> four different IPC paths (upstream
        collapses them all onto rpc_port_100)."""
        base = str(tmp_path)
        paths = {
            get_zmq_rpc_path_lmcache("e", "offload", 100, rank, base_url=base)
            for rank in range(4)
        }
        assert len(paths) == 4
        upstream_tp0 = {
            get_zmq_rpc_path_lmcache("e", "offload", 100, 0, base_url=base)
            for _ in range(4)
        }
        assert len(upstream_tp0) == 1


class TestOffloadServerRealBind:
    def test_four_pp_ranks_bind_concurrently(self, monkeypatch, tmp_path) -> None:
        """Four live server threads bind distinct sockets and close promptly."""
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", str(tmp_path))
        servers = []
        try:
            for rank in range(4):
                factory = _make_factory("worker", rank)
                with patch(TP_RANK, return_value=0):
                    server = factory.maybe_create_offload_server()
                assert server is not None
                servers.append(server)
            assert len({server.endpoint for server in servers}) == 4
        finally:
            started = time.monotonic()
            for server in servers:
                server.close()
                server.close()
                assert not server.thread.is_alive()
            assert time.monotonic() - started < 2

    def test_close_waits_for_active_store(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", str(tmp_path))
        factory = _make_factory("worker", 0)
        started, finish, closing = Event(), Event(), Event()
        completed = []

        def store(**kwargs):
            started.set()
            assert finish.wait(timeout=10)
            completed.append("store")

        factory.lmcache_engine.store.side_effect = store
        server = factory.maybe_create_offload_server()
        assert server is not None
        client = zmq.Context.instance().socket(zmq.REQ)
        client.setsockopt(zmq.RCVTIMEO, 2000)
        client.connect(server.endpoint)

        def close():
            closing.set()
            server.close()
            completed.append("close")

        try:
            client.send(msgspec.msgpack.encode(OffloadMsg([1], [2], [3])))
            assert started.wait(timeout=5)
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(close)
                try:
                    assert closing.wait(timeout=5)
                    # Pool cleanup cannot proceed while this store is active.
                    with pytest.raises(FutureTimeoutError):
                        pending.result(timeout=0.1)
                finally:
                    finish.set()
                pending.result(timeout=2)
            response = msgspec.msgpack.decode(client.recv(), type=OffloadRetMsg)
            assert completed == ["store", "close"]
            assert response.success
            factory.lmcache_engine.store.assert_called_once_with(
                hashes=[1], slot_mapping=[2], offsets=[3]
            )
            assert not server.thread.is_alive()
        finally:
            finish.set()
            server.close()
            client.close(linger=0)

    def test_failed_request_does_not_break_next_request(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", str(tmp_path))
        factory = _make_factory("worker", 0)
        factory.lmcache_engine.store.side_effect = [RuntimeError("store failed"), None]
        server = factory.maybe_create_offload_server()
        assert server is not None
        client = zmq.Context.instance().socket(zmq.REQ)
        client.setsockopt(zmq.RCVTIMEO, 2000)
        client.connect(server.endpoint)
        try:
            for success in (False, True):
                client.send(msgspec.msgpack.encode(OffloadMsg([1], [2], [3])))
                response = msgspec.msgpack.decode(client.recv(), type=OffloadRetMsg)
                assert response.success == success
        finally:
            server.close()
            client.close(linger=0)

    def test_bind_error_reaches_factory(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", str(tmp_path / "missing"))
        with pytest.raises(zmq.ZMQError):
            _make_factory("worker", 0).maybe_create_offload_server()

# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for ``ZMQOffloadServer`` and its socket key.

The vLLM service factory keys the offload server's IPC socket by the worker's
global rank. Keyed by TP rank, every pipeline stage with TP=1 on one host
binds the same ``..._service_offload_lmcache_rpc_port_100`` path: libzmq
unlinks the file and binds again, so a racing stage can get ``EADDRINUSE`` and
mark its cache engine init-failed.

The server owns its socket on its request thread. close() stops an idle
server within one poll interval and waits for an active store.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import time

# Third Party
import msgspec
import pytest
import zmq

# First Party
from lmcache.inprocess.integration.vllm import vllm_service_factory
from lmcache.inprocess.integration.vllm.vllm_service_factory import (
    VllmServiceFactory,
)
from lmcache.inprocess.v1.offload_server.message import OffloadMsg, OffloadRetMsg
from lmcache.inprocess.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.inprocess.v1.rpc_utils import get_zmq_rpc_path_lmcache


def _engine(worker_id: int = 0) -> MagicMock:
    engine = MagicMock()
    engine.metadata = SimpleNamespace(engine_id="e", worker_id=worker_id)
    return engine


def _factory(role: str, worker_id: int) -> VllmServiceFactory:
    factory = VllmServiceFactory(MagicMock(), MagicMock(), role)
    factory.lmcache_engine = _engine(worker_id)
    return factory


def _client(server: ZMQOffloadServer) -> zmq.Socket:
    client = zmq.Context.instance().socket(zmq.REQ)
    client.setsockopt(zmq.RCVTIMEO, 2000)
    client.connect(server.endpoint)
    return client


class TestSocketKey:
    def test_scheduler_gets_none(self) -> None:
        with patch.object(vllm_service_factory, "ZMQOffloadServer") as server_cls:
            assert _factory("scheduler", 0).maybe_create_offload_server() is None
        server_cls.assert_not_called()

    @pytest.mark.parametrize("worker_id", [0, 1, 2, 3])
    def test_worker_binds_with_global_rank(self, worker_id: int) -> None:
        factory = _factory("worker", worker_id)
        with patch.object(vllm_service_factory, "ZMQOffloadServer") as server_cls:
            server = factory.maybe_create_offload_server()
        server_cls.assert_called_once_with(factory.lmcache_engine, worker_id)
        assert server is server_cls.return_value

    def test_ranks_get_distinct_paths(self, tmp_path) -> None:
        paths = {
            get_zmq_rpc_path_lmcache("e", "offload", 100, rank, base_url=str(tmp_path))
            for rank in range(4)
        }
        assert len(paths) == 4


class TestServer:
    def test_four_ranks_bind_concurrently(self, monkeypatch, tmp_path) -> None:
        """Four live servers bind distinct sockets and close promptly."""
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", str(tmp_path))
        servers = []
        try:
            for rank in range(4):
                server = _factory("worker", rank).maybe_create_offload_server()
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
        engine = _engine()
        started, finish, closing = Event(), Event(), Event()
        completed = []

        def store(**kwargs):
            started.set()
            assert finish.wait(timeout=10)
            completed.append("store")

        engine.store.side_effect = store
        server = ZMQOffloadServer(engine, 0)
        client = _client(server)

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
                    with pytest.raises(FutureTimeoutError):
                        pending.result(timeout=0.1)
                finally:
                    finish.set()
                pending.result(timeout=2)
            response = msgspec.msgpack.decode(client.recv(), type=OffloadRetMsg)
            assert completed == ["store", "close"]
            assert response.success
            engine.store.assert_called_once_with(
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
        engine = _engine()
        engine.store.side_effect = [RuntimeError("store failed"), None]
        server = ZMQOffloadServer(engine, 0)
        client = _client(server)
        try:
            for success in (False, True):
                client.send(msgspec.msgpack.encode(OffloadMsg([1], [2], [3])))
                response = msgspec.msgpack.decode(client.recv(), type=OffloadRetMsg)
                assert response.success == success
        finally:
            server.close()
            client.close(linger=0)

    def test_bind_error_reaches_the_caller(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", str(tmp_path / "missing"))
        with pytest.raises(zmq.ZMQError):
            ZMQOffloadServer(_engine(), 0)

# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import os
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.inprocess.v1.cache_engine import LMCacheEngine
from lmcache.inprocess.v1.offload_server.abstract_server import OffloadServerInterface
from lmcache.inprocess.v1.offload_server.message import OffloadMsg, OffloadRetMsg
from lmcache.inprocess.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_rpc_path_lmcache,
    get_zmq_socket,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)

# How often the request thread checks for close() while idle.
_POLL_INTERVAL_MS = 100


class ZMQOffloadServer(OffloadServerInterface):
    """Serve offload requests on a per-rank IPC socket.

    The socket is created, used and closed by the request thread only: ZMQ
    sockets are not thread-safe, so close() asks that thread to stop instead
    of closing the socket under a blocking ``recv``. The thread polls, so an
    idle server stops within one poll interval, and close() returns only once
    any active store has finished.
    """

    def __init__(self, lmcache_engine: LMCacheEngine, rank: int):
        """``rank`` keys the IPC socket path, so it must be unique among the
        workers that share an engine id on one host: pass the worker's global
        rank, not its TP rank, or pipeline stages on one host collide."""
        self.lmcache_engine = lmcache_engine
        self.ctx = get_zmq_context(use_asyncio=False)
        offload_rpc_port = int(os.environ.get("LMCACHE_OFFLOAD_RPC_PORT", 100))
        engine_id = lmcache_engine.metadata.engine_id or "default"
        self.socket_path = get_zmq_rpc_path_lmcache(
            engine_id, "offload", offload_rpc_port, rank
        )
        self.endpoint = f"ipc://{self.socket_path}"

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: Optional[Exception] = None
        self.thread = threading.Thread(
            target=self._serve, daemon=True, name="offload-server-thread"
        )
        self.thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self.thread.join()
            raise self._startup_error

    def _serve(self) -> None:
        socket = None
        try:
            socket = get_zmq_socket(
                self.ctx,
                self.socket_path,
                "ipc",
                zmq.REP,  # type: ignore[attr-defined]
                "bind",
            )
            self._ready.set()
            while not self._stop.is_set():
                if not socket.poll(timeout=_POLL_INTERVAL_MS) or self._stop.is_set():
                    continue
                frame = socket.recv(copy=False)
                try:
                    request = msgspec.msgpack.decode(frame, type=OffloadMsg)
                    result = self.offload(
                        request.hashes, request.slot_mapping, request.offsets
                    )
                except Exception:
                    logger.exception("Offload request failed")
                    result = False
                # A REP socket must reply before it can receive again.
                socket.send(msgspec.msgpack.encode(OffloadRetMsg(success=result)))
        except Exception as e:
            if not self._ready.is_set():
                self._startup_error = e
            elif not self._stop.is_set():
                logger.exception("Offload server stopped unexpectedly")
        finally:
            if socket is not None:
                socket.close(linger=0)
            self._ready.set()

    def offload(
        self,
        hashes: List[int],
        slot_mapping: List[int],
        offsets: List[int],
    ) -> bool:
        self.lmcache_engine.store(
            hashes=hashes, slot_mapping=slot_mapping, offsets=offsets
        )
        return True

    def close(self) -> None:
        """Stop accepting requests and wait for an active store to finish.

        A stuck store keeps close() from returning; the manager's close
        timeout bounds how long shutdown waits for it.
        """
        logger.info("Closing ZMQOffloadServer...")
        self._stop.set()
        self.thread.join()
        logger.info("Offload server thread terminated")

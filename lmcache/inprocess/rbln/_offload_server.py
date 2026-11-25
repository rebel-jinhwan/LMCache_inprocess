# SPDX-License-Identifier: Apache-2.0
"""LMCache 0.5.7 compatibility fix for offload-server shutdown.

The upstream close interrupts recv by closing its socket from another thread,
then waits five seconds. That can exhaust vLLM's shutdown deadline before CPU
pools unregister. Keep socket operations in one thread and poll for shutdown.
Remove this shim when the pinned LMCache release provides equivalent cleanup.
"""

# Standard
import os
import threading

# Third Party
import msgspec
import zmq

# First Party
from lmcache.inprocess.v1.cache_engine import LMCacheEngine
from lmcache.inprocess.v1.offload_server.message import OffloadMsg, OffloadRetMsg
from lmcache.inprocess.v1.offload_server.zmq_server import ZMQOffloadServer
from lmcache.inprocess.v1.rpc_utils import get_zmq_context, get_zmq_rpc_path_lmcache
from lmcache.logging import init_logger

logger = init_logger(__name__)


class InterruptibleZMQOffloadServer(ZMQOffloadServer):
    """Serve the existing RPC protocol and drain active work before close returns."""

    def __init__(self, lmcache_engine: LMCacheEngine, tp_rank: int) -> None:
        self.lmcache_engine = lmcache_engine
        self.ctx = get_zmq_context(use_asyncio=False)
        path = get_zmq_rpc_path_lmcache(
            lmcache_engine.metadata.engine_id or "default",
            "offload",
            int(os.environ.get("LMCACHE_OFFLOAD_RPC_PORT", 100)),
            tp_rank,
        )
        self.endpoint = f"ipc://{path}"
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self.thread = threading.Thread(
            target=self._serve, daemon=True, name="offload-server-thread"
        )
        self.thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self.thread.join()
            raise self._startup_error

    def close(self) -> None:
        """Stop accepting requests and wait for an active store to finish.

        A stuck store prevents close from returning. LMCache 0.5.7's service
        close stops waiting after its timeout and then destroys the engine.
        """
        self._stop.set()
        self.thread.join()

    def _serve(self) -> None:
        socket = None
        try:
            socket = self.ctx.socket(zmq.REP)
            socket.bind(self.endpoint)
            self._ready.set()
            while not self._stop.is_set():
                if not socket.poll(timeout=100) or self._stop.is_set():
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
                socket.send(msgspec.msgpack.encode(OffloadRetMsg(success=result)))
        except Exception as exc:
            if not self._ready.is_set():
                self._startup_error = exc
            elif not self._stop.is_set():
                logger.exception("Offload server stopped unexpectedly")
        finally:
            if socket is not None:
                socket.close(linger=0)
            self._ready.set()

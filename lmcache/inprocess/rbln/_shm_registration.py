# SPDX-License-Identifier: Apache-2.0
"""Own the existing RBLN MP DMA staging without changing POSIX SHM layout.

Upstream transposes between token-major SHM and contiguous host DMA staging.
Registering SHM itself would not cover those transfers.
"""

# Standard
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Iterator
import math
import threading

# Third Party
import torch

# First Party
from lmcache.inprocess.rbln._host_memory import (
    allocate_cpu_pool,
    ensure_cpu_pool_registered,
    host_register_enabled,
    release_cpu_pool,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)


class _Staging:
    """Reusable buffers owned by one synchronous engine transfer context."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.closing = False
        self.buffers: dict[
            tuple[tuple[int, ...], torch.dtype, torch.device], torch.Tensor
        ] = {}

    def get(
        self,
        blocks: int,
        per_block: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.closing:
            raise RuntimeError("Cannot transfer through a closed RBLN MP context")
        key = (per_block, dtype, device)
        buffer = self.buffers.get(key)
        if buffer is None or buffer.shape[0] < blocks:
            if buffer is not None:
                release_cpu_pool(buffer)
                del self.buffers[key]
            size = (
                blocks
                * math.prod(per_block)
                * torch.empty((), dtype=dtype).element_size()
            )
            buffer = allocate_cpu_pool(size).view(dtype).reshape(blocks, *per_block)
            self.buffers[key] = buffer
        ensure_cpu_pool_registered(buffer, device)
        return buffer[:blocks]

    def close(self) -> None:
        with self.lock:
            self.closing = True
            for key in list(self.buffers):
                release_cpu_pool(self.buffers[key])
                del self.buffers[key]


_active: ContextVar[_Staging | None] = ContextVar("rbln_mp_staging", default=None)
_device: ContextVar[torch.device | None] = ContextVar("rbln_mp_device", default=None)
_applied = False
_warned_unowned = False


def patch_shm_registration() -> None:
    """Install owned staging on the synchronous RBLN engine-driven path."""
    global _applied
    # Reject an invalid setting before upstream selects a transfer context;
    # an error raised later from the pin hook would select pickle instead.
    host_register_enabled()
    if _applied:
        return
    # First Party
    from lmcache.v1.multiprocess.transfer_context import shm
    from lmcache.v1.platform.devices.rbln import kv_ops

    context = shm.EngineDrivenContextShm
    original_pin = context._pin_shm_buffer
    original_unpin = context._unpin_shm_buffer
    original_guard = context.transfer_guard
    original_staging = kv_ops._host_staging

    def pin(self: Any) -> None:
        if shm.current_device_spec.device_type != "rbln":
            original_pin(self)
            return
        if getattr(self, "_rbln_staging", None) is None:
            self._rbln_staging = _Staging()

    def unpin(self: Any) -> None:
        manager = getattr(self, "_rbln_staging", None)
        if manager is None:
            original_unpin(self)
            return
        # Upstream unpins only after in-flight transfers drain. Keep the
        # closed manager attached so a late transfer fails instead of
        # allocating new staging.
        manager.close()

    def staging(
        n_blocks: int,
        per_block: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        global _warned_unowned
        manager, device = _active.get(), _device.get()
        if manager is None or device is None or device.type != "rbln":
            if (
                manager is None
                and device is not None
                and device.type == "rbln"
                and not _warned_unowned
                and host_register_enabled()
            ):
                # E.g. the pickle or async engine contexts: neither installs
                # an owned staging buffer, so the upstream one stays unregistered.
                _warned_unowned = True
                logger.warning(
                    "RBLN MP staging outside an owned engine context; "
                    "using upstream unregistered staging"
                )
            return original_staging(n_blocks, per_block, dtype)
        return manager.get(n_blocks, per_block, dtype, device)

    @contextmanager
    def guard(self: Any) -> Iterator[None]:
        # Every SHM gather/scatter runs inside upstream's transfer guard.
        with original_guard(self):
            manager = getattr(self, "_rbln_staging", None)
            if manager is None:
                yield
                return
            # Staging buffers are shared by the context, not per thread.
            with manager.lock:
                token = _active.set(manager)
                try:
                    yield
                finally:
                    _active.reset(token)

    def scope_device(original: Any) -> Any:
        @wraps(original)
        def transfer(paged_layers: Any, *args: Any, **kwargs: Any) -> Any:
            token = _device.set(paged_layers[0].device)
            try:
                return original(paged_layers, *args, **kwargs)
            finally:
                _device.reset(token)

        return transfer

    context._pin_shm_buffer = pin
    context._unpin_shm_buffer = unpin
    context.transfer_guard = guard
    kv_ops._host_staging = staging
    # Both HND layouts reach host staging through these; MLA stages on device.
    for name in ("_gather_hnd", "_scatter_hnd"):
        setattr(kv_ops, name, scope_device(getattr(kv_ops, name)))
    _applied = True

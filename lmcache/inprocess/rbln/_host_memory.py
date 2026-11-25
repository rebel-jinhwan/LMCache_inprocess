# SPDX-License-Identifier: Apache-2.0
"""Host registration owned by a CPU pool's tensor storage.

Allocation does not initialize an NPU. The connector registers a pool on each
consuming device before its first transfer. Missing registration support or
exhausted admission yields no or partial coverage, and uncovered transfers use
the ordinary copy path. Runtime, contract and configuration errors propagate.
Explicit allocator close and storage destruction share the same
unregister-before-free operation.
"""

# Standard
from bisect import bisect_right
from typing import Optional, Protocol
import os
import threading
import weakref

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# torch-rbln submits at most 1 GiB per native registration.
_MAX_SLOT_BYTES = 1 << 30
_HUGE_PAGE_BYTES = 2 << 20
# torch-rbln scans every live registration on each call, so setup is
# quadratic in the number of calls; group slots beyond this count.
_MAX_SLOT_REGISTRATIONS = 4096


class HostMemoryRegistration(Protocol):
    """Optional platform result; partial coverage must not become a bool success."""

    @property
    def registered_ranges(self) -> tuple[tuple[int, int], ...]: ...

    @property
    def registered_bytes(self) -> int: ...

    @property
    def fully_registered(self) -> bool: ...

    def covers(self, ptr: int, size: int) -> bool: ...

    def close(self) -> None: ...


def host_register_enabled() -> bool:
    """Parse ``LMCACHE_RBLN_HOST_REGISTER``; values other than 0 and 1 raise."""
    setting = os.environ.get("LMCACHE_RBLN_HOST_REGISTER", "1")
    if setting not in {"0", "1"}:
        raise ValueError("LMCACHE_RBLN_HOST_REGISTER must be 0 or 1")
    return setting == "1"


def register_cpu_memory(
    buffer: torch.Tensor, device: int
) -> Optional[HostMemoryRegistration]:
    """Delegate allocation-independent registration policy to torch-rbln."""
    # Third Party
    import torch_rbln.memory as memory

    register = getattr(memory, "host_register_memory", None)
    if not callable(register):
        logger.warning(
            "RBLN managed host registration is unavailable; "
            "using unregistered transfers"
        )
        return None
    return register(buffer, device=device)


def _slot_segments(
    base: int, nbytes: int, allocation: Optional[tuple[int, int]]
) -> Optional[list[tuple[int, int]]]:
    """Split a pool on its allocator's fixed object grid.

    The runtime's per-copy cost grows with the copy's offset inside its native
    registration. Registering on slot boundaries starts every chunk at or near
    the beginning of its own range and keeps chunks off torch-rbln's 1 GiB
    range boundaries. ``allocation`` is one object's ``(address, aligned
    size)``; with full chunks only, every object shares that size and grid.
    A pool with more than ``_MAX_SLOT_REGISTRATIONS`` slots is split into
    units of several whole slots instead.

    Only units that are 2 MiB multiples on a 2 MiB grid and no larger than one
    native range qualify. They keep huge-page credit and cost no extra page
    tables. Other geometries return None and the pool is registered whole.
    """
    if allocation is None:
        return None
    ptr, stride = allocation
    offset = ptr - base
    if (
        not 0 <= offset < nbytes
        or not 0 < stride <= _MAX_SLOT_BYTES
        or base % _HUGE_PAGE_BYTES
        or stride % _HUGE_PAGE_BYTES
    ):
        return None
    first = offset % stride
    span = _MAX_SLOT_REGISTRATIONS * stride
    unit = stride * max(1, -(-(nbytes - first) // span))
    count = (nbytes - first) // unit
    if first % _HUGE_PAGE_BYTES or unit > _MAX_SLOT_BYTES or count < 2:
        return None
    segments = [(first + index * unit, unit) for index in range(count)]
    if first:
        segments.insert(0, (0, first))
    end = first + count * unit
    if end < nbytes:
        segments.append((end, nbytes - end))
    return segments


def _close_registrations(
    registrations: list[tuple[int, HostMemoryRegistration]],
) -> tuple[list[tuple[int, HostMemoryRegistration]], Optional[Exception]]:
    """Close every registration; return those whose release failed."""
    failed = []
    error: Optional[Exception] = None
    for entry in registrations:
        try:
            entry[1].close()
        except Exception as exc:
            failed.append(entry)
            if error is None:
                error = exc
    return failed, error


class _Pool:
    def __init__(
        self, buffer: torch.Tensor, storage: torch.UntypedStorage, enabled: bool
    ) -> None:
        self.buffer: Optional[torch.Tensor] = buffer
        self.storage = weakref.ref(storage)
        self.backing_storage = weakref.ref(buffer.untyped_storage())
        self.ptr = buffer.data_ptr()
        self.nbytes = buffer.numel()
        self.enabled = enabled
        # (start address, registration) in address order, per device.
        self.devices: dict[int, list[tuple[int, HostMemoryRegistration]]] = {}
        self.attempted: set[int] = set()
        self.closing = False
        self.closed = False
        self.lock = threading.RLock()

    def register(
        self,
        device: int,
        ptr: int,
        size: int,
        allocation: Optional[tuple[int, int]] = None,
    ) -> bool:
        with self.lock:
            if self.closing:
                raise RuntimeError("Cannot transfer through a closed RBLN CPU pool")
            if not self.enabled:
                return False
            if device not in self.attempted:
                self._register_device(device, allocation)
            registrations = self.devices.get(device, [])
            index = bisect_right(registrations, ptr, key=lambda entry: entry[0]) - 1
            return index >= 0 and registrations[index][1].covers(ptr, size)

    def _register_device(
        self, device: int, allocation: Optional[tuple[int, int]]
    ) -> None:
        assert self.buffer is not None
        segments = _slot_segments(self.ptr, self.nbytes, allocation)
        created: list[tuple[int, HostMemoryRegistration]] = []
        try:
            for offset, nbytes in segments or [(0, self.nbytes)]:
                view = self.buffer[offset : offset + nbytes]
                registration = register_cpu_memory(view, device)
                if registration is None:
                    break
                created.append((self.ptr + offset, registration))
        except BaseException:
            # Release this attempt's ranges so a later transfer can retry.
            # Retain any whose release failed, with the pool's memory.
            failed, _ = _close_registrations(created)
            if failed:
                self.devices[device] = failed
                self.attempted.add(device)
            raise
        if created:
            self.devices[device] = created
            registered = sum(entry[1].registered_bytes for entry in created)
            if registered == 0:
                logger.warning(
                    "No host memory registered; using unregistered transfers"
                )
            logger.info(
                "RBLN host registration: device=%d total_bytes=%d "
                "registered_bytes=%d registrations=%d",
                device,
                self.nbytes,
                registered,
                len(created),
            )
        self.attempted.add(device)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closing = True
            for device in sorted(self.devices):
                failed, error = _close_registrations(self.devices[device])
                if error is not None:
                    self.devices[device] = failed
                    raise error
                del self.devices[device]
            self.closed = True
            self.buffer = None


_pools: weakref.WeakValueDictionary[int, _Pool] = weakref.WeakValueDictionary()
_pools_lock = threading.RLock()
# A failed finalizer cannot raise to an allocator owner. Retain its backing
# allocation rather than freeing pages that the device still has registered.
_failed_releases: dict[int, _Pool] = {}


def _finalize_pool(pool: _Pool) -> None:
    try:
        pool.close()
    except Exception:
        with _pools_lock:
            _failed_releases[pool.ptr] = pool
        logger.exception(
            "RBLN CPU pool unregister failed; retaining %d bytes at %#x",
            pool.nbytes,
            pool.ptr,
        )


def allocate_cpu_pool(size: int) -> torch.Tensor:
    """Allocate zeroed huge-host memory whose views own registration lifetime."""
    # Third Party
    from torch_rbln.memory import huge_host_empty

    enabled = host_register_enabled()
    backing = huge_host_empty(size)
    # A new Storage shares the bytes via NumPy's buffer ownership, without a
    # copy. The finalizer can retain the original allocation without retaining
    # the Storage it watches. Every slice/view keeps that Storage alive.
    buffer = torch.from_numpy(backing.numpy())
    pool = _Pool(backing, buffer.untyped_storage(), enabled=enabled)
    with _pools_lock:
        _pools[pool.ptr] = pool
    weakref.finalize(buffer.untyped_storage(), _finalize_pool, pool)
    return buffer


def _get_pool(buffer: torch.Tensor) -> Optional[_Pool]:
    if buffer.device.type != "cpu":
        return None
    storage = buffer.untyped_storage()
    with _pools_lock:
        pool = _pools.get(storage.data_ptr())
    # Exception tracebacks can outlive freed storage and retain a closed pool.
    # A matching address alone does not establish ownership after address reuse.
    if pool is not None and (
        storage is pool.storage() or storage is pool.backing_storage()
    ):
        return pool
    return None


def is_cpu_pool(buffer: torch.Tensor) -> bool:
    """Whether a tensor shares a CPU pool allocated by this module."""
    return _get_pool(buffer) is not None


def cpu_pool_registered_bytes(buffer: torch.Tensor, device: torch.device) -> int:
    """Return confirmed registered bytes of the owned pool, including all views."""
    pool = _get_pool(buffer)
    if pool is None or device.type != "rbln":
        return 0
    with pool.lock:
        registrations = pool.devices.get(device.index, [])
        return sum(entry[1].registered_bytes for entry in registrations)


def release_cpu_pool(buffer: torch.Tensor) -> bool:
    """Unregister an owned pool; return False for an upstream allocation.

    A registration failure propagates while keeping the original memory alive.
    Repeated release is harmless. Tensor views retain the allocation itself.
    """
    pool = _get_pool(buffer)
    if pool is None:
        return False
    pool.close()
    with _pools_lock:
        _failed_releases.pop(pool.ptr, None)
    return True


def ensure_cpu_pool_registered(
    buffer: torch.Tensor,
    device: Optional[torch.device],
    allocation: Optional[tuple[int, int]] = None,
) -> bool:
    """Register a pooled CPU tensor once on its actual consuming RBLN device.

    Return whether one registered range covers this contiguous transfer.
    Disabled, unsupported, unregistered and boundary-crossing transfers
    retain their ordinary copy path. ``allocation`` is the transferred
    object's ``(address, aligned size)`` when the pool's allocator keeps a
    fixed object grid; the first registration on a device then follows it.
    """
    if device is None or device.type != "rbln" or buffer.device.type != "cpu":
        return False
    pool = _get_pool(buffer)
    if pool is None:
        # Not a pool owned here, e.g. an independent storage over bytes that
        # another owner registered. Ask torch-rbln for that owner's coverage.
        # Third Party
        import torch_rbln.memory as memory

        query = getattr(memory, "is_host_memory_registered", None)
        return bool(
            callable(query)
            and buffer.is_contiguous()
            and query(buffer.data_ptr(), buffer.numel() * buffer.element_size(), device)
        )
    if device.index is None:
        raise ValueError("CPU pool registration requires an explicit RBLN device")
    ptr = buffer.data_ptr()
    size = buffer.numel() * buffer.element_size()
    registered = pool.register(device.index, ptr, size, allocation)
    return buffer.is_contiguous() and registered

# SPDX-License-Identifier: Apache-2.0
"""Registration follows pool storage lifetime, not individual chunk tensors."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, call
import gc
import logging
import weakref

# Third Party
import pytest
import torch
import torch_rbln.memory as memory

# First Party
from lmcache.inprocess.rbln import _host_memory as host_memory
from lmcache.inprocess.rbln._host_memory import (
    allocate_cpu_pool,
    ensure_cpu_pool_registered,
    is_cpu_pool,
    release_cpu_pool,
)


class FakeRegistration:
    """Owned result without retaining tensors in a Mock call history."""

    def __init__(self, buffer, device, unregister, enabled=True):
        self.owner = buffer
        self.device = device
        self.unregister = unregister
        self.registered_ranges = (
            ((buffer.data_ptr(), buffer.numel()),) if enabled else ()
        )
        self.size = buffer.numel()

    @property
    def registered_bytes(self):
        return sum(size for _, size in self.registered_ranges)

    @property
    def fully_registered(self):
        return self.registered_bytes == self.size

    def covers(self, ptr, size):
        return any(
            start <= ptr and ptr + size <= start + length
            for start, length in self.registered_ranges
        )

    def close(self):
        if self.registered_ranges:
            self.unregister(self.owner.data_ptr(), device=self.device)
            self.registered_ranges = ()
        self.owner = None


@pytest.fixture
def registration(monkeypatch):
    register = Mock(return_value=True)
    unregister = Mock()
    monkeypatch.setattr(memory, "host_register", register, raising=False)
    monkeypatch.setattr(memory, "host_unregister", unregister, raising=False)

    def managed(buffer, device):
        enabled = register(buffer.data_ptr(), buffer.numel(), device=device)
        return FakeRegistration(buffer, device, unregister, enabled)

    monkeypatch.setattr(memory, "host_register_memory", managed, raising=False)
    monkeypatch.setattr(
        memory, "is_host_memory_registered", lambda *args: False, raising=False
    )
    monkeypatch.delenv("LMCACHE_RBLN_HOST_REGISTER", raising=False)
    yield register, unregister
    gc.collect()


def test_pool_is_registered_once_for_all_chunk_views(registration):
    register, unregister = registration
    pool = allocate_cpu_pool(8192)
    register.assert_not_called()
    for chunk in (pool[:4096], pool[4096:].view(torch.float32)):
        assert ensure_cpu_pool_registered(chunk, torch.device("rbln:0"))
    register.assert_called_once_with(pool.data_ptr(), 8192, device=0)
    assert release_cpu_pool(pool)
    assert release_cpu_pool(pool)
    unregister.assert_called_once_with(pool.data_ptr(), device=0)


def test_registration_uses_each_consuming_device(registration):
    register, unregister = registration
    pool = allocate_cpu_pool(4096)
    for device in (1, 0, 1):
        assert ensure_cpu_pool_registered(pool, torch.device(f"rbln:{device}"))
    assert register.call_args_list == [
        call(pool.data_ptr(), 4096, device=1),
        call(pool.data_ptr(), 4096, device=0),
    ]
    release_cpu_pool(pool)
    assert unregister.call_args_list == [
        call(pool.data_ptr(), device=0),
        call(pool.data_ptr(), device=1),
    ]


def test_last_storage_view_unregisters_before_backing_is_freed(
    registration, monkeypatch
):
    register, unregister = registration
    backing = torch.zeros(4096, dtype=torch.uint8)
    backing_ref = weakref.ref(backing)
    monkeypatch.setattr(memory, "huge_host_empty", lambda size: backing_ref())
    pool = allocate_cpu_pool(4096)
    view = pool[128:]
    ptr = pool.data_ptr()
    alive_during_unregister = []
    unregister.side_effect = lambda *a, **kw: alive_during_unregister.append(
        backing_ref() is not None
    )
    assert ensure_cpu_pool_registered(view, torch.device("rbln:0"))
    del pool, backing
    gc.collect()
    unregister.assert_not_called()
    assert is_cpu_pool(view)
    del view
    gc.collect()
    unregister.assert_called_once_with(ptr, device=0)
    assert alive_during_unregister == [True]
    assert backing_ref() is None
    register.assert_called_once()


def test_drain_failure_retains_registration_for_retry(registration):
    register, unregister = registration
    active = set()
    fail_drain = True

    def pin(ptr, size, *, device):
        active.add((ptr, device))
        return True

    def unpin(ptr, *, device):
        nonlocal fail_drain
        assert (ptr, device) in active
        if device == 1 and fail_drain:
            fail_drain = False
            raise RuntimeError("pending transfer wait failed")
        active.remove((ptr, device))

    register.side_effect = pin
    unregister.side_effect = unpin
    pool = allocate_cpu_pool(4096)
    ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
    ensure_cpu_pool_registered(pool, torch.device("rbln:1"))
    with pytest.raises(RuntimeError, match="pending transfer wait failed"):
        release_cpu_pool(pool)
    assert active == {(pool.data_ptr(), 1)}
    assert is_cpu_pool(pool)
    with pytest.raises(RuntimeError, match="closed RBLN CPU pool"):
        ensure_cpu_pool_registered(pool, torch.device("rbln:1"))
    assert release_cpu_pool(pool)
    assert active == set()
    assert unregister.call_args_list == [
        call(pool.data_ptr(), device=0),
        call(pool.data_ptr(), device=1),
        call(pool.data_ptr(), device=1),
    ]


def test_unpin_failure_after_registry_erase_stays_quarantined(
    registration, monkeypatch
):
    register, unregister = registration
    # Isolate the deliberately unrecoverable, mocked registration from other tests.
    monkeypatch.setattr(host_memory, "_failed_releases", {})
    backing = torch.zeros(4096, dtype=torch.uint8)
    backing_ref = weakref.ref(backing)
    monkeypatch.setattr(memory, "huge_host_empty", lambda size: backing_ref())
    active = set()

    def pin(ptr, size, *, device):
        active.add((ptr, device))
        return True

    def unpin(ptr, *, device):
        if (ptr, device) not in active:
            raise RuntimeError("address is not registered")
        # Mirrors HostMemoryRegistry::Unregister: erase precedes UMD Unpin.
        active.remove((ptr, device))
        raise RuntimeError("UMD unpin failed")

    register.side_effect = pin
    unregister.side_effect = unpin
    pool = allocate_cpu_pool(4096)
    ptr = pool.data_ptr()
    ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
    with pytest.raises(RuntimeError, match="UMD unpin failed"):
        release_cpu_pool(pool)
    assert active == set()
    with pytest.raises(RuntimeError, match="address is not registered"):
        release_cpu_pool(pool)
    with pytest.raises(RuntimeError, match="closed RBLN CPU pool"):
        ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
    del pool, backing
    gc.collect()
    assert backing_ref() is not None
    assert ptr in host_memory._failed_releases
    assert unregister.call_count == 3


@pytest.mark.parametrize("available", [False, True])
def test_unsupported_registration_falls_back_once(
    registration, monkeypatch, caplog, available
):
    register, unregister = registration
    monkeypatch.setattr(
        logging.getLogger("lmcache.inprocess.rbln._host_memory"), "propagate", True
    )
    if available:
        register.return_value = False
    else:
        monkeypatch.delattr(memory, "host_register_memory")
    pool = allocate_cpu_pool(4096)
    for _ in range(2):
        assert not ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
    assert register.call_count == int(available)
    assert "unregistered transfers" in caplog.text
    release_cpu_pool(pool)
    unregister.assert_not_called()


def test_registration_error_is_not_reported_as_fallback(registration):
    register, unregister = registration
    register.side_effect = RuntimeError("overlapping range")
    pool = allocate_cpu_pool(4096)
    with pytest.raises(RuntimeError, match="overlapping range"):
        ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
    release_cpu_pool(pool)
    unregister.assert_not_called()


def test_partial_device_registration_is_cleaned_up(registration):
    register, unregister = registration
    register.side_effect = [True, RuntimeError("registration refused")]
    pool = allocate_cpu_pool(4096)
    ptr = pool.data_ptr()
    assert ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
    with pytest.raises(RuntimeError, match="registration refused"):
        ensure_cpu_pool_registered(pool, torch.device("rbln:1"))
    # The configured exception retains the failing call's tensor via its traceback.
    register.side_effect = None
    del pool
    gc.collect()
    unregister.assert_called_once_with(ptr, device=0)


def test_unused_pool_destruction_does_not_initialize_registration(registration):
    register, unregister = registration
    pool = allocate_cpu_pool(4096)
    del pool
    gc.collect()
    register.assert_not_called()
    unregister.assert_not_called()


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, OSError])
def test_finalizer_failure_keeps_registered_pages_alive(
    registration: tuple[Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
) -> None:
    _, unregister = registration
    monkeypatch.setattr(host_memory, "_failed_releases", {})
    backing = torch.zeros(4096, dtype=torch.uint8)
    backing_ref = weakref.ref(backing)
    monkeypatch.setattr(memory, "huge_host_empty", lambda size: backing_ref())
    pool = allocate_cpu_pool(4096)
    ptr = pool.data_ptr()
    ensure_cpu_pool_registered(pool, torch.device("rbln:0"))

    def fail_unregister(*args: object, **kwargs: object) -> None:
        # Do not keep an exception whose traceback would itself retain the pool.
        raise error_type("unregister failed")

    unregister.side_effect = fail_unregister
    del pool, backing
    gc.collect()
    unregister.assert_called_once_with(ptr, device=0)
    assert ptr in host_memory._failed_releases
    assert backing_ref() is not None
    # Retry through a surviving backing reference, then verify the quarantine
    # no longer retains memory after a successful release.
    unregister.side_effect = None
    retained = backing_ref()
    assert retained is not None
    assert release_cpu_pool(retained)
    del retained
    assert host_memory._failed_releases == {}
    caplog.clear()
    gc.collect()
    assert backing_ref() is None


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, OSError])
def test_explicit_release_propagates_unregister_errors(
    registration: tuple[Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    _, unregister = registration
    pool = allocate_cpu_pool(4096)
    assert ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
    unregister.side_effect = error_type("unregister failed")
    try:
        with pytest.raises(error_type, match="unregister failed"):
            release_cpu_pool(pool)
        assert is_cpu_pool(pool)
    finally:
        unregister.side_effect = None
        release_cpu_pool(pool)


def test_unowned_and_cpu_only_transfers_are_unchanged(registration):
    register, _ = registration
    unowned = torch.empty(4096, dtype=torch.uint8)
    assert not ensure_cpu_pool_registered(unowned, torch.device("rbln:0"))
    assert not release_cpu_pool(unowned)
    pool = allocate_cpu_pool(4096)
    assert not ensure_cpu_pool_registered(pool, torch.device("cpu"))
    release_cpu_pool(pool)
    register.assert_not_called()


def test_reused_address_does_not_inherit_closed_pool_ownership(
    registration, monkeypatch
):
    register, _ = registration
    pool = allocate_cpu_pool(4096)
    owner = host_memory._pools[pool.data_ptr()]
    release_cpu_pool(pool)
    del pool
    gc.collect()
    assert owner.closed and owner.buffer is None

    unowned = torch.empty(4096, dtype=torch.uint8)
    # Model allocator address reuse while an exception retains the old owner.
    monkeypatch.setitem(host_memory._pools, unowned.data_ptr(), owner)
    assert not is_cpu_pool(unowned)
    assert not ensure_cpu_pool_registered(unowned, torch.device("rbln:0"))
    assert not release_cpu_pool(unowned)
    register.assert_not_called()


def test_opt_out_keeps_the_same_huge_host_allocation(registration, monkeypatch):
    register, _ = registration
    monkeypatch.setenv("LMCACHE_RBLN_HOST_REGISTER", "0")
    pool = allocate_cpu_pool(4096)
    assert pool.data_ptr() % (2 * 1024 * 1024) == 0
    assert is_cpu_pool(pool)
    assert not ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
    release_cpu_pool(pool)
    register.assert_not_called()


def test_concurrent_first_use_registers_once(registration):
    register, _ = registration
    pool = allocate_cpu_pool(4096)
    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(
            executor.map(
                lambda _: ensure_cpu_pool_registered(pool, torch.device("rbln:0")),
                range(16),
            )
        )
    assert all(outcomes)
    register.assert_called_once()
    release_cpu_pool(pool)


def test_closed_pool_cannot_be_registered_again(registration):
    register, _ = registration
    pool = allocate_cpu_pool(4096)
    release_cpu_pool(pool)
    with pytest.raises(RuntimeError, match="closed RBLN CPU pool"):
        ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
    register.assert_not_called()


def test_partial_registration_uses_actual_transfer_coverage(registration, monkeypatch):
    _, unregister = registration
    pool = allocate_cpu_pool(8192)
    ptr = pool.data_ptr()

    def partial(buffer, device):
        result = FakeRegistration(buffer, device, unregister)
        result.registered_ranges = ((ptr, 2048), (ptr + 4096, 4096))
        return result

    monkeypatch.setattr(memory, "host_register_memory", partial)
    try:
        assert ensure_cpu_pool_registered(pool[:2048], torch.device("rbln:0"))
        assert not ensure_cpu_pool_registered(pool[2048:4096], torch.device("rbln:0"))
        assert ensure_cpu_pool_registered(pool[4096:], torch.device("rbln:0"))
        assert not ensure_cpu_pool_registered(pool, torch.device("rbln:0"))
        assert not ensure_cpu_pool_registered(pool[::2], torch.device("rbln:0"))
    finally:
        release_cpu_pool(pool)
    unregister.assert_called_once_with(ptr, device=0)


def test_adjacent_registrations_are_not_one_transfer(registration, monkeypatch):
    _, unregister = registration
    pool = allocate_cpu_pool(8192)

    def partial(buffer, device):
        result = FakeRegistration(buffer, device, unregister)
        result.registered_ranges = (
            (buffer.data_ptr(), 4096),
            (buffer.data_ptr() + 4096, 4096),
        )
        assert result.fully_registered
        return result

    monkeypatch.setattr(memory, "host_register_memory", partial)
    try:
        assert ensure_cpu_pool_registered(pool[:4096], torch.device("rbln:0"))
        assert not ensure_cpu_pool_registered(pool[2048:6144], torch.device("rbln:0"))
    finally:
        release_cpu_pool(pool)


def test_mp_views_query_registration_instead_of_registering_again(
    registration, monkeypatch
):
    register, _ = registration
    shared_view = torch.zeros(8192, dtype=torch.uint8)
    query = Mock(return_value=True)
    monkeypatch.setattr(memory, "is_host_memory_registered", query)
    device = torch.device("rbln:0")
    assert ensure_cpu_pool_registered(shared_view[4096:], device)
    query.assert_called_once_with(shared_view.data_ptr() + 4096, 4096, device)
    register.assert_not_called()


@pytest.fixture
def page_grid(monkeypatch):
    """Use 4 KiB pages as the huge-page unit so grids fit in small pools."""
    monkeypatch.setattr(host_memory, "_HUGE_PAGE_BYTES", 4096)
    monkeypatch.setattr(host_memory, "_MAX_SLOT_BYTES", 16384)

    def aligned(size):
        raw = torch.zeros(size + 4096, dtype=torch.uint8)
        start = -raw.data_ptr() % 4096
        return raw[start : start + size]

    monkeypatch.setattr(memory, "huge_host_empty", aligned)


@pytest.mark.parametrize("first_slot", [0, 4096])
def test_full_chunk_grid_registers_one_range_per_slot(
    registration, page_grid, first_slot
):
    register, unregister = registration
    pool = allocate_cpu_pool(5 * 8192 + 4096)
    base = pool.data_ptr()
    device = torch.device("rbln:0")
    allocation = (base + first_slot + 2 * 8192, 8192)
    chunk = pool[first_slot + 2 * 8192 : first_slot + 3 * 8192]
    assert ensure_cpu_pool_registered(chunk, device, allocation)
    tail = first_slot + 5 * 8192
    expected = [(0, first_slot)] if first_slot else []
    expected += [(first_slot + slot * 8192, 8192) for slot in range(5)]
    expected += [(tail, pool.numel() - tail)] if tail < pool.numel() else []
    assert register.call_args_list == [
        call(base + offset, size, device=0) for offset, size in expected
    ]
    # Each slot is covered; a transfer across two slots is not.
    start = first_slot + 8192
    assert ensure_cpu_pool_registered(pool[start : start + 8192], device)
    assert not ensure_cpu_pool_registered(pool[start + 4096 : start + 12288], device)
    assert host_memory.cpu_pool_registered_bytes(pool, device) == pool.numel()
    release_cpu_pool(pool)
    assert unregister.call_args_list == [
        call(base + offset, device=0) for offset, _ in expected
    ]


@pytest.mark.parametrize(
    ("offset", "stride"),
    [
        (0, 6144),  # not a multiple of the huge-page unit
        (6144, 8192),  # grid not on the huge-page unit
        (0, 32768),  # larger than one native range
        (12288, 16384),  # fewer than two slots fit after the first grid offset
    ],
)
def test_ineligible_grid_registers_the_whole_pool(
    registration, page_grid, offset, stride
):
    register, _ = registration
    pool = allocate_cpu_pool(3 * 8192 + 4096)
    base = pool.data_ptr()
    assert ensure_cpu_pool_registered(
        pool[offset : offset + 1024], torch.device("rbln:0"), (base + offset, stride)
    )
    register.assert_called_once_with(base, pool.numel(), device=0)
    release_cpu_pool(pool)


def test_failed_slot_registration_releases_the_attempt_for_retry(
    registration, page_grid
):
    register, unregister = registration
    pool = allocate_cpu_pool(4 * 8192)
    base = pool.data_ptr()
    device = torch.device("rbln:0")
    register.side_effect = [True, True, RuntimeError("native error")]
    with pytest.raises(RuntimeError, match="native error"):
        ensure_cpu_pool_registered(pool[:8192], device, (base, 8192))
    assert unregister.call_args_list == [
        call(base, device=0),
        call(base + 8192, device=0),
    ]
    register.side_effect = None
    assert ensure_cpu_pool_registered(pool[:8192], device, (base, 8192))
    assert register.call_count == 3 + 4
    release_cpu_pool(pool)


def test_failed_slot_release_retains_only_unreleased_slots(registration, page_grid):
    _, unregister = registration
    pool = allocate_cpu_pool(4 * 8192)
    base = pool.data_ptr()
    device = torch.device("rbln:0")
    assert ensure_cpu_pool_registered(pool[:8192], device, (base, 8192))

    def release(ptr, *, device):
        if ptr == base + 8192:
            raise RuntimeError("unpin failed")

    unregister.side_effect = release
    with pytest.raises(RuntimeError, match="unpin failed"):
        release_cpu_pool(pool)
    assert unregister.call_count == 4
    assert [start for start, _ in host_memory._get_pool(pool).devices[0]] == [
        base + 8192
    ]
    unregister.side_effect = None
    assert release_cpu_pool(pool)
    assert unregister.call_args_list[-1] == call(base + 8192, device=0)


def test_many_slots_are_registered_in_bounded_units(
    registration, page_grid, monkeypatch
):
    register, _ = registration
    monkeypatch.setattr(host_memory, "_MAX_SLOT_REGISTRATIONS", 2)
    pool = allocate_cpu_pool(6 * 4096)
    base = pool.data_ptr()
    device = torch.device("rbln:0")
    # Six 4 KiB slots exceed two registrations: units of three whole slots.
    assert ensure_cpu_pool_registered(pool[:4096], device, (base, 4096))
    assert register.call_args_list == [
        call(base, 12288, device=0),
        call(base + 12288, 12288, device=0),
    ]
    # A unit boundary stays on the slot grid, so no slot spans two units.
    for slot in range(6):
        assert ensure_cpu_pool_registered(pool[slot * 4096 : (slot + 1) * 4096], device)
    release_cpu_pool(pool)

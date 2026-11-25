# SPDX-License-Identifier: Apache-2.0
"""The engine owns DMA staging; POSIX SHM retains its upstream wire format."""

# Standard
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import Mock
import gc
import logging
import threading

# Third Party
import pytest
import torch
import torch_rbln.memory as memory

# First Party
from lmcache.inprocess.rbln import _patches
from lmcache.inprocess.rbln import _shm_registration as adapter
from lmcache.inprocess.rbln._host_memory import ensure_cpu_pool_registered
from lmcache.v1.multiprocess.transfer_context import shm
from lmcache.v1.multiprocess.transfer_context.base import (
    ContextClosedError,
    EngineDrivenContext,
)
from lmcache.v1.platform.devices.rbln import kv_ops


@pytest.fixture
def context(monkeypatch):
    adapter.patch_shm_registration()
    server = SharedMemory(create=True, size=8192)
    engine = SharedMemory(name=server.name)
    ctx = shm.EngineDrivenContextShm.__new__(shm.EngineDrivenContextShm)
    # The base initializer only sets up the transfer guard state.
    EngineDrivenContext.__init__(ctx, None, None, 1.0)
    ctx._shm, ctx._shm_buffer = engine, engine.buf
    ctx._shm_name, ctx._pool_size = server.name, 8192
    ctx._pinned = False
    ctx._pinned_ptr = ctx._pinned_size = 0
    ctx._release_started = False
    monkeypatch.setattr(shm, "current_device_spec", SimpleNamespace(device_type="rbln"))
    monkeypatch.delenv("LMCACHE_RBLN_HOST_REGISTER", raising=False)
    yield ctx, server
    ctx.close()
    server.close()
    server.unlink()


class Registration:
    def __init__(self, buffer, size):
        self.owner = buffer
        self.registered_bytes = size
        self.fully_registered = size == buffer.numel()
        self.registered_ranges = ((buffer.data_ptr(), size),) if size else ()
        self.closed = False
        self.fail = False

    def covers(self, ptr, size):
        return any(
            start <= ptr and ptr + size <= start + length
            for start, length in self.registered_ranges
        )

    def close(self):
        if self.fail:
            raise RuntimeError("unpin failed")
        self.owner = None
        self.registered_ranges = ()
        self.closed = True


@pytest.fixture
def handles(monkeypatch):
    handles = []

    def register(buffer, device):
        assert device == 0
        handle = Registration(buffer, buffer.numel())
        handles.append(handle)
        return handle

    monkeypatch.setattr(memory, "host_register_memory", register, raising=False)
    yield handles
    for handle in handles:
        handle.fail = False


@pytest.mark.parametrize("registered_size", [0, 4096, 8192])
def test_context_registers_staging_and_closes_before_shm(
    context, monkeypatch, registered_size
):
    ctx, server = context
    handles = []

    def register(buffer, device):
        assert device == 0
        handle = Registration(buffer, registered_size)
        handles.append(handle)
        return handle

    monkeypatch.setattr(memory, "host_register_memory", register, raising=False)
    ctx._pin_shm_buffer()
    manager = ctx._rbln_staging
    ctx._pin_shm_buffer()
    assert ctx._rbln_staging is manager
    assert not handles and not ctx._pinned
    with manager.lock:
        staged = manager.get(2, (4096,), torch.uint8, torch.device("rbln:0"))
    handle = handles[0]
    assert len(handles) == 1
    assert ensure_cpu_pool_registered(staged[0], torch.device("rbln:0")) == (
        registered_size >= 4096
    )
    ctx._shm_buffer[0] = 19
    assert server.buf[0] == 19
    close_mapping = Mock(wraps=ctx._shm.close)

    def check_close():
        assert handle.closed
        close_mapping()

    monkeypatch.setattr(ctx._shm, "close", check_close)
    ctx.close()
    ctx.close()
    close_mapping.assert_called_once()
    assert ctx._rbln_staging is manager and manager.closing
    assert not manager.buffers
    with pytest.raises(RuntimeError, match="closed RBLN MP"):
        manager.get(1, (4096,), torch.uint8, torch.device("rbln:0"))
    still_exists = SharedMemory(name=server.name)
    still_exists.close()


def test_staging_growth_returns_old_budget_before_replacement(context, handles):
    ctx, _ = context
    ctx._pin_shm_buffer()
    manager = ctx._rbln_staging
    device = torch.device("rbln:0")
    first = manager.get(1, (4096,), torch.uint8, device)
    assert manager.get(1, (4096,), torch.uint8, device).data_ptr() == first.data_ptr()
    grown = manager.get(2, (4096,), torch.uint8, device)
    assert grown.shape == (2, 4096)
    assert handles[0].closed and len(handles) == 2
    assert len(manager.buffers) == 1


def test_failed_growth_keeps_old_buffer_until_release_succeeds(context, handles):
    ctx, _ = context
    ctx._pin_shm_buffer()
    manager = ctx._rbln_staging
    device = torch.device("rbln:0")
    first = manager.get(1, (4096,), torch.uint8, device)
    handles[0].fail = True
    try:
        with pytest.raises(RuntimeError, match="unpin failed"):
            manager.get(2, (4096,), torch.uint8, device)
        assert next(iter(manager.buffers.values())).data_ptr() == first.data_ptr()
        with pytest.raises(RuntimeError, match="unpin failed"):
            ctx.close()
        assert ctx._shm is not None and handles[0].owner is not None
    finally:
        handles[0].fail = False
    # Upstream does not retry a started release: the failed staging and the
    # SHM mapping stay until the staging pool itself is released.
    ctx.close()
    assert ctx._shm is not None and not handles[0].closed
    manager.close()
    assert handles[0].closed


def test_failed_growth_allocation_does_not_cache_closed_pool(
    context, handles, monkeypatch
):
    ctx, _ = context
    ctx._pin_shm_buffer()
    manager = ctx._rbln_staging
    device = torch.device("rbln:0")
    manager.get(1, (4096,), torch.uint8, device)
    allocate = adapter.allocate_cpu_pool
    with monkeypatch.context() as patch:
        patch.setattr(
            adapter, "allocate_cpu_pool", Mock(side_effect=MemoryError("no room"))
        )
        with pytest.raises(MemoryError, match="no room"):
            manager.get(2, (4096,), torch.uint8, device)
    assert handles[0].closed and not manager.buffers
    assert adapter.allocate_cpu_pool is allocate
    manager.get(2, (4096,), torch.uint8, device)
    assert len(handles) == 2


@pytest.mark.parametrize("disabled", [False, True])
def test_missing_api_or_opt_out_never_calls_unguarded_registration(
    context, monkeypatch, disabled
):
    ctx, _ = context
    legacy = Mock(side_effect=AssertionError("unsafe legacy API"))
    # torch-rbln 0.12.0 has neither API; the fake legacy hook must still not run.
    monkeypatch.setattr(memory, "host_register", legacy, raising=False)
    monkeypatch.setattr(memory, "host_register_memory", None, raising=False)
    if disabled:
        monkeypatch.setenv("LMCACHE_RBLN_HOST_REGISTER", "0")
    ctx._pin_shm_buffer()
    ctx._rbln_staging.get(1, (4096,), torch.uint8, torch.device("rbln:0"))
    assert not ctx._pinned
    legacy.assert_not_called()


def test_other_platform_uses_upstream_bool_backend(context, monkeypatch):
    ctx, _ = context
    backend = SimpleNamespace(
        device_type="cuda", pin_memory=Mock(return_value=True), unpin_memory=Mock()
    )
    monkeypatch.setattr(shm, "current_device_spec", backend)
    monkeypatch.setattr(shm, "torch_dev", SimpleNamespace(is_available=lambda: True))
    before = type(ctx)._pin_shm_buffer
    adapter.patch_shm_registration()
    assert type(ctx)._pin_shm_buffer is before
    ctx._pin_shm_buffer()
    assert ctx._pinned
    backend.pin_memory.assert_called_once()
    ctx.close()
    backend.unpin_memory.assert_called_once()


@pytest.mark.parametrize("setting", ["true", "2", ""])
def test_invalid_setting_fails_before_transport_selection(
    context, handles, monkeypatch, setting
):
    ctx, _ = context
    monkeypatch.setenv("LMCACHE_RBLN_HOST_REGISTER", setting)
    # Upstream turns an exception from the pin hook into a pickle context.
    with pytest.raises(ValueError, match="must be 0 or 1"):
        adapter.patch_shm_registration()
    # In-process mode fails the same way, before its pool exists.
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _patches.apply()
    ctx._pin_shm_buffer()
    assert ctx._rbln_staging is not None and not handles


def test_staging_outside_owned_context_warns_once(context, monkeypatch, caplog):
    monkeypatch.setattr(adapter, "_warned_unowned", False)
    monkeypatch.setattr(
        logging.getLogger("lmcache.inprocess.rbln._shm_registration"), "propagate", True
    )
    token = adapter._device.set(torch.device("rbln:0"))
    try:
        with caplog.at_level("WARNING"):
            kv_ops._host_staging(1, (4096,), torch.uint8)
            first = caplog.text.count("outside an owned engine context")
            kv_ops._host_staging(1, (4096,), torch.uint8)
    finally:
        adapter._device.reset(token)
    assert first >= 1
    assert caplog.text.count("outside an owned engine context") == first


def test_transfer_guard_scopes_staging_and_resets_after_failure(context):
    ctx, _ = context
    ctx._pin_shm_buffer()
    with pytest.raises(RuntimeError, match="MQ failed"):
        with ctx.transfer_guard():
            assert adapter._active.get() is ctx._rbln_staging
            raise RuntimeError("MQ failed")
    assert adapter._active.get() is None
    ctx.close()
    with pytest.raises(ContextClosedError):
        with ctx.transfer_guard():
            pass
    assert adapter._active.get() is None


def test_close_releases_staging_after_transfer_drains(context, handles):
    ctx, _ = context
    ctx._pin_shm_buffer()
    manager = ctx._rbln_staging
    entered, proceed, closed = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def transfer():
        try:
            with ctx.transfer_guard():
                manager.get(1, (4096,), torch.uint8, torch.device("rbln:0"))
                entered.set()
                assert proceed.wait(5)
                assert ctx._shm is not None and not handles[0].closed
        except BaseException as error:
            errors.append(error)

    def close():
        try:
            ctx.close()
            closed.set()
        except BaseException as error:
            errors.append(error)

    active = threading.Thread(target=transfer)
    closer = threading.Thread(target=close)
    active.start()
    assert entered.wait(5)
    closer.start()
    try:
        assert not closed.wait(0.05)
    finally:
        proceed.set()
        active.join(5)
        closer.join(5)
    assert not errors and closed.is_set()
    assert handles[0].closed and manager.closing


def test_abandoned_context_staging_unregisters_on_storage_finalization(
    monkeypatch, handles
):
    manager = adapter._Staging()
    manager.get(1, (4096,), torch.uint8, torch.device("rbln:0"))
    del manager
    gc.collect()
    assert handles[0].closed


@pytest.mark.parametrize("kv_first", [False, True], ids=["blocks-first", "kv-first"])
@pytest.mark.parametrize("copy_fails", [False, True])
def test_kv_ops_register_actual_dma_staging_on_operand_device(
    context, handles, monkeypatch, copy_fails, kv_first
):
    ctx, _ = context
    ctx._pin_shm_buffer()
    # [2, NB, H, BS, D]; the blocks-first layout stores [NB, 2, H, BS, D].
    kv = torch.arange(2 * 2 * 4 * 32 * 64, dtype=torch.float32).reshape(2, 2, 4, 32, 64)
    raw = kv if kv_first else kv.transpose(0, 1).contiguous()
    if kv_first:
        gather = kv_ops.gather_blocks_to_chunk_hnd_kv_first
        scatter = kv_ops.scatter_chunk_to_blocks_hnd_kv_first
    else:
        gather = kv_ops.gather_blocks_to_chunk_hnd
        scatter = kv_ops.scatter_chunk_to_blocks_hnd

    class PagedLayer:
        # CPU storage exercises the real transpose/copy algorithm. Only the
        # operand device label and native registration are test substitutes.
        device = torch.device("rbln:0")
        shape, dtype = raw.shape, raw.dtype

        def __getitem__(self, item):
            return raw[item]

    original_copy = torch._foreach_copy_
    dma_addresses = []

    def copy(dsts, srcs):
        assert handles
        for dst, src in zip(dsts, srcs, strict=True):
            size = src.numel() * src.element_size()
            assert handles[-1].covers(dst.data_ptr(), size) or handles[-1].covers(
                src.data_ptr(), size
            )
            dma_addresses.append((dst.data_ptr(), src.data_ptr()))
        if copy_fails:
            raise RuntimeError("copy failed")
        return original_copy(dsts, srcs)

    monkeypatch.setattr(torch, "_foreach_copy_", copy)
    chunk = torch.empty((2, 1, 64, 256), dtype=raw.dtype)
    token = adapter._active.set(ctx._rbln_staging)
    try:
        if copy_fails:
            with pytest.raises(RuntimeError, match="copy failed"):
                gather([PagedLayer()], [0, 1], chunk)
        else:
            gather([PagedLayer()], [0, 1], chunk)
            expected = kv.permute(0, 1, 3, 2, 4).reshape_as(chunk)
            assert torch.equal(chunk, expected)
            chunk.fill_(17)
            scatter([PagedLayer()], [0, 1], chunk)
            assert torch.all(raw == 17)
        assert dma_addresses
        assert adapter._device.get() is None
        assert not handles[0].covers(chunk.data_ptr(), chunk.numel() * 4)
    finally:
        adapter._active.reset(token)

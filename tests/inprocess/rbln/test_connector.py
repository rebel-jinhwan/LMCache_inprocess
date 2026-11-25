# SPDX-License-Identifier: Apache-2.0
"""Tests for RBLNConnector (device-tensor mode).

These tests exercise the connector against ordinary CPU torch tensors that
mimic the RBLN attention backend layout ``[2, B, H, 1, S, D]``.  The data
path is identical to running on ``device='rbln'``: per-block stack +
``.cpu()`` for D2H, per-block ``.to(device)`` + scatter for H2D.  CPU is
chosen here because unit tests run without NPU hardware; ``.cpu()`` on a
CPU tensor is a no-op so the structural and numerical behavior is equivalent.
"""

# Standard
from unittest.mock import MagicMock, Mock, call
import sys
import types

# Third Party
import pytest
import torch
import torch_rbln.memory as memory

# First Party
from lmcache.inprocess.rbln._host_memory import allocate_cpu_pool, release_cpu_pool
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)


def _make_slot_mapping(num_tokens: int, offset: int = 0) -> torch.Tensor:
    """Create a contiguous slot mapping starting at ``offset``."""
    return torch.arange(offset, offset + num_tokens, dtype=torch.long)


def _make_kv_caches(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    num_blocks: int = 4,
    dtype: torch.dtype = torch.float16,
) -> list[torch.Tensor]:
    return [
        torch.zeros(2, num_blocks, num_kv_heads, 1, block_size, head_dim, dtype=dtype)
        for _ in range(num_layers)
    ]


def _make_registered_connector(
    num_layers: int = 2,
    num_kv_heads: int = 4,
    head_dim: int = 64,
    block_size: int = 16,
    num_blocks: int = 4,
    dtype: torch.dtype = torch.float16,
) -> RBLNConnector:
    conn = RBLNConnector(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    conn.register_kv_caches(
        _make_kv_caches(
            num_layers, num_kv_heads, head_dim, block_size, num_blocks, dtype
        )
    )
    return conn


@pytest.mark.parametrize("direction", ["from_gpu", "to_gpu"])
@pytest.mark.parametrize("enabled", [False, True])
def test_transfers_register_whole_pool_once_per_consuming_device(
    monkeypatch: pytest.MonkeyPatch, direction: str, enabled: bool
) -> None:
    register, unregister = Mock(return_value=True), Mock()

    def managed(buffer, device):
        ptr, size = buffer.data_ptr(), buffer.numel()
        register(ptr, size, device=device)
        return types.SimpleNamespace(
            registered_bytes=size,
            fully_registered=True,
            covers=lambda ptr, size: True,
            close=lambda: unregister(ptr, device=device),
        )

    monkeypatch.setattr(memory, "host_register_memory", managed, raising=False)
    monkeypatch.setattr(
        memory, "huge_host_empty", lambda size: torch.zeros(size, dtype=torch.uint8)
    )
    monkeypatch.setenv("LMCACHE_RBLN_HOST_REGISTER", str(int(enabled)))
    connector = _make_registered_connector(num_kv_heads=2, head_dim=4)

    def cpu_staging(
        numel: int, dtype: torch.dtype, like: torch.Tensor | None
    ) -> torch.Tensor:
        return torch.empty(numel, dtype=dtype)

    # Keep transfer math on CPU while exercising the real pool ownership and
    # registration hook for a logical consuming NPU. No driver call is made.
    monkeypatch.setattr(connector, "_alloc_staging", cpu_staging)
    pool = allocate_cpu_pool(4096)
    ptr = pool.data_ptr()
    shape = connector.get_shape(8)
    nbytes = shape.numel() * 2
    chunks = [
        types.SimpleNamespace(
            tensor=pool[offset : offset + nbytes].view(torch.float16).view(shape)
        )
        for offset in (0, nbytes)
    ]
    # Partial blocks avoid the full-block gather's independent registration
    # check, so deleting the common hook must fail this test in either direction.
    slots = torch.arange(8)
    try:
        register.assert_not_called()
        for device in (0, 1, 0):
            monkeypatch.setattr(connector, "_device", torch.device(f"rbln:{device}"))
            for chunk in chunks:
                getattr(connector, direction)(chunk, 0, 8, slot_mapping=slots)
        assert register.call_args_list == (
            [call(ptr, 4096, device=0), call(ptr, 4096, device=1)] if enabled else []
        )
    finally:
        release_cpu_pool(pool)
    assert unregister.call_args_list == (
        [call(ptr, device=0), call(ptr, device=1)] if enabled else []
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_chunk_views_pass_the_allocator_slot_when_enabled(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    connector = RBLNConnector(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        block_size=8,
        dtype=torch.float16,
        slot_registration=enabled,
    )
    monkeypatch.setattr(connector, "_device", torch.device("rbln:0"))
    seen = []
    module = sys.modules[RBLNConnector.__module__]
    monkeypatch.setattr(
        module,
        "ensure_cpu_pool_registered",
        lambda view, device, allocation=None: seen.append(allocation),
    )
    shape = connector.get_shape(8)
    raw = torch.zeros(shape.numel() * 2 + 4096, dtype=torch.uint8)
    obj = types.SimpleNamespace(
        tensor=raw[: shape.numel() * 2].view(torch.float16).view(shape),
        raw_data=raw[: shape.numel() * 2],
        metadata=types.SimpleNamespace(phy_size=raw.numel()),
    )
    connector._chunk_views(obj, 8)
    assert seen == [(raw.data_ptr(), raw.numel()) if enabled else None]
    # Objects without an allocator slot keep whole-pool registration.
    seen.clear()
    connector._chunk_views(types.SimpleNamespace(tensor=obj.tensor), 8)
    assert seen == [None]


class TestRBLNConnectorInit:
    def test_default_init(self):
        conn = RBLNConnector(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
        )
        assert conn._kv_caches == []
        assert conn._device is None

    def test_hidden_dim_computation(self):
        conn = RBLNConnector(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
        )
        assert conn.hidden_dim_size == 1024


class TestGetShape:
    def test_shape_format(self):
        conn = RBLNConnector(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
        )
        assert conn.get_shape(64) == torch.Size([2, 32, 8, 64, 128])

    def test_shape_single_token(self):
        conn = RBLNConnector(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=torch.float16,
        )
        assert conn.get_shape(1) == torch.Size([2, 32, 8, 1, 128])

    @pytest.mark.parametrize(
        "num_kv_heads,head_dim",
        [(1, 64), (4, 256), (32, 128), (128, 32)],
    )
    def test_shape_various_head_configs(self, num_kv_heads, head_dim):
        conn = RBLNConnector(
            num_layers=16,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=16,
            dtype=torch.float16,
        )
        assert conn.get_shape(32) == torch.Size([2, 16, num_kv_heads, 32, head_dim])


class TestRegisterKVCaches:
    """register_kv_caches stores ordered tensors and infers device + dtype."""

    def test_basic_registration(self):
        conn = _make_registered_connector(num_layers=3, num_blocks=4)
        assert len(conn._kv_caches) == 3
        assert conn._device == torch.device("cpu")

    def test_layer_count_mismatch_raises(self):
        conn = RBLNConnector(
            num_layers=4,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        with pytest.raises(ValueError, match="Expected 4 KV caches"):
            conn.register_kv_caches(_make_kv_caches(2, 4, 64, 16))

    def test_dtype_mismatch_raises(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        with pytest.raises(ValueError, match="dtype"):
            conn.register_kv_caches(_make_kv_caches(1, 4, 64, 16, dtype=torch.bfloat16))

    def test_head_dim_mismatch_raises(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        with pytest.raises(ValueError, match="head_dim"):
            conn.register_kv_caches(_make_kv_caches(1, 4, 128, 16))

    def test_empty_list_raises(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        with pytest.raises(ValueError, match="non-empty"):
            conn.register_kv_caches([])

    def test_initialize_kvcaches_ptr_forwards(self):
        conn = RBLNConnector(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        kv_caches = _make_kv_caches(2, 4, 64, 16)
        conn.initialize_kvcaches_ptr(ordered_kv_caches=kv_caches)
        assert len(conn._kv_caches) == 2


class TestSlotGrouping:
    def test_contiguous_single_block(self):
        conn = _make_registered_connector(block_size=16)
        groups = conn._group_slots_by_block(torch.arange(16, dtype=torch.long))
        assert list(groups.keys()) == [0]
        assert len(groups[0]) == 16

    def test_cross_block_boundary(self):
        conn = _make_registered_connector(block_size=16)
        groups = conn._group_slots_by_block(torch.arange(14, 18, dtype=torch.long))
        assert sorted(groups.keys()) == [0, 1]

    def test_scattered_slots(self):
        conn = _make_registered_connector(block_size=16)
        slots = torch.tensor([0, 32, 64], dtype=torch.long)
        groups = conn._group_slots_by_block(slots)
        assert sorted(groups.keys()) == [0, 2, 4]


class TestNotReadyErrors:
    def test_from_gpu_without_register_raises(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(16), dtype=torch.float16)
        with pytest.raises(RuntimeError, match="not registered"):
            conn.from_gpu(mem, 0, 16, slot_mapping=_make_slot_mapping(16))

    def test_to_gpu_without_register_raises(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(16), dtype=torch.float16)
        with pytest.raises(RuntimeError, match="not registered"):
            conn.to_gpu(mem, 0, 16, slot_mapping=_make_slot_mapping(16))

    def test_missing_slot_mapping_raises(self):
        conn = _make_registered_connector()
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(16), dtype=torch.float16)
        with pytest.raises(AssertionError, match="slot_mapping"):
            conn.from_gpu(mem, 0, 16)

    def test_none_tensor_raises(self):
        conn = _make_registered_connector()
        mem = MagicMock()
        mem.tensor = None
        with pytest.raises(AssertionError):
            conn.from_gpu(mem, 0, 16, slot_mapping=_make_slot_mapping(16))


class TestRoundTripCorrectness:
    """End-to-end correctness: H2D then D2H should preserve the data."""

    @staticmethod
    def _writeable_kv_caches(
        num_layers: int = 2,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        block_size: int = 16,
        num_blocks: int = 4,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[RBLNConnector, list[torch.Tensor]]:
        conn = RBLNConnector(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
        )
        kv_caches = _make_kv_caches(
            num_layers, num_kv_heads, head_dim, block_size, num_blocks, dtype
        )
        conn.register_kv_caches(kv_caches)
        return conn, kv_caches

    def test_full_block_round_trip(self):
        conn, _ = self._writeable_kv_caches()
        shape = conn.get_shape(16)
        original = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(16)

        write = MagicMock()
        write.tensor = original.clone()
        conn.to_gpu(write, 0, 16, slot_mapping=slot_mapping)

        read = MagicMock()
        read.tensor = torch.zeros(shape, dtype=torch.float16)
        conn.from_gpu(read, 0, 16, slot_mapping=slot_mapping)

        assert torch.allclose(original, read.tensor, atol=1e-3)

    def test_partial_block_round_trip(self):
        conn, _ = self._writeable_kv_caches(block_size=16)
        num_tokens = 5
        shape = conn.get_shape(num_tokens)
        original = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(num_tokens)

        write = MagicMock()
        write.tensor = original.clone()
        conn.to_gpu(write, 0, num_tokens, slot_mapping=slot_mapping)

        read = MagicMock()
        read.tensor = torch.zeros(shape, dtype=torch.float16)
        conn.from_gpu(read, 0, num_tokens, slot_mapping=slot_mapping)

        assert torch.allclose(original, read.tensor, atol=1e-3)

    def test_round_trip_at_nonzero_block_offset(self):
        """Critical case: chunk starts mid-block (block_size=256, chunk=64)."""
        conn, _ = self._writeable_kv_caches(block_size=256, num_blocks=4)
        chunk_size = 64
        shape = conn.get_shape(chunk_size)
        original = torch.randn(shape, dtype=torch.float16)

        # Second chunk inside block 0: slots 64-127
        slot_mapping = _make_slot_mapping(chunk_size, offset=chunk_size)

        write = MagicMock()
        write.tensor = original.clone()
        conn.to_gpu(write, 0, chunk_size, slot_mapping=slot_mapping)

        read = MagicMock()
        read.tensor = torch.zeros(shape, dtype=torch.float16)
        conn.from_gpu(read, 0, chunk_size, slot_mapping=slot_mapping)

        assert torch.allclose(original, read.tensor, atol=1e-3)

    def test_multiple_chunks_in_one_block(self):
        conn, _ = self._writeable_kv_caches(block_size=256, num_blocks=4)
        chunk_size = 64
        num_chunks = 3
        shape = conn.get_shape(chunk_size)

        chunk_data: list[torch.Tensor] = []
        for i in range(num_chunks):
            data = torch.randn(shape, dtype=torch.float16)
            chunk_data.append(data)
            mem = MagicMock()
            mem.tensor = data.clone()
            slots = _make_slot_mapping(chunk_size, offset=i * chunk_size)
            conn.to_gpu(mem, 0, chunk_size, slot_mapping=slots)

        for i in range(num_chunks):
            mem = MagicMock()
            mem.tensor = torch.zeros(shape, dtype=torch.float16)
            slots = _make_slot_mapping(chunk_size, offset=i * chunk_size)
            conn.from_gpu(mem, 0, chunk_size, slot_mapping=slots)
            assert torch.allclose(chunk_data[i], mem.tensor, atol=1e-3), (
                f"Round-trip failed for chunk {i}"
            )

    def test_partial_block_does_not_touch_other_slots(self):
        """A partial-block H2D must preserve neighbouring slots."""
        conn, kv_caches = self._writeable_kv_caches(block_size=16, num_blocks=2)

        # Pre-populate the device cache with sentinel values.
        sentinel = 7.0
        for kv in kv_caches:
            kv.fill_(sentinel)

        # Write only 5 tokens (partial block) starting at slot 0.
        shape = conn.get_shape(5)
        write = MagicMock()
        write.tensor = torch.randn(shape, dtype=torch.float16)
        conn.to_gpu(write, 0, 5, slot_mapping=_make_slot_mapping(5))

        # Slots 5..15 of block 0 must still be the sentinel.
        for kv in kv_caches:
            untouched = kv[:, 0, :, 0, 5:, :]
            assert torch.all(untouched == sentinel), (
                "partial H2D should not overwrite untouched slots"
            )


class TestFallbackFreeGather:
    """Verify the K/V-split contiguous-slice gather path in ``from_gpu``.

    Each unique block contributes a ``[H, n, D]`` slice copied directly
    from ``kv[k, blk, :, 0, off0:off0+n, :]`` into the chunk-layout dst.
    These tests exercise:
      - scattered (non-contiguous) block indices,
      - partial blocks where only a slot subset is requested,
      - chunks crossing a block boundary (cross-block correctness).
    """

    @staticmethod
    def _prefilled_kv_caches(
        num_layers: int = 2,
        num_kv_heads: int = 2,
        head_dim: int = 8,
        block_size: int = 16,
        num_blocks: int = 8,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[RBLNConnector, list[torch.Tensor]]:
        conn = RBLNConnector(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
        )
        kv_caches = _make_kv_caches(
            num_layers, num_kv_heads, head_dim, block_size, num_blocks, dtype
        )
        # Encode (k, layer, block, slot) into the value so we can verify
        # exact byte-level provenance post-gather.
        for layer_idx, kv in enumerate(kv_caches):
            for k in range(2):
                for blk in range(num_blocks):
                    for slot in range(block_size):
                        kv[k, blk, :, 0, slot, :] = (
                            k * 1_000_000 + layer_idx * 10_000 + blk * 100 + slot
                        )
        conn.register_kv_caches(kv_caches)
        return conn, kv_caches

    def test_scattered_blocks_gather_correctness(self):
        conn, kv_caches = self._prefilled_kv_caches()
        # Three tokens from three non-contiguous blocks.
        block_size = 16
        slot_mapping = torch.tensor(
            [3, block_size + 5, 5 * block_size + 1], dtype=torch.long
        )
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(3), dtype=torch.float16)
        conn.from_gpu(mem, 0, 3, slot_mapping=slot_mapping)

        # Output is in block-grouped order (matches existing contract
        # consumed by ``to_gpu``): block 0, block 1, block 5.
        got = mem.tensor.reshape(2, 2, 2, 3, 8)  # [K, L, H, T, D]
        for tok_idx, (blk, off) in enumerate([(0, 3), (1, 5), (5, 1)]):
            for k in range(2):
                for layer_idx in range(2):
                    expected = kv_caches[layer_idx][k, blk, :, 0, off, :]
                    assert torch.allclose(got[k, layer_idx, :, tok_idx, :], expected), (
                        f"mismatch at tok={tok_idx} (blk={blk}, off={off}) "
                        f"k={k} layer={layer_idx}"
                    )

    def test_cross_block_boundary_gather_correctness(self):
        conn, kv_caches = self._prefilled_kv_caches(block_size=16, num_blocks=4)
        # Chunk straddles block 0 / block 1.
        slot_mapping = torch.arange(14, 18, dtype=torch.long)
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(4), dtype=torch.float16)
        conn.from_gpu(mem, 0, 4, slot_mapping=slot_mapping)

        got = mem.tensor.reshape(2, 2, 2, 4, 8)
        # Block-grouped order: block 0 (slots 14, 15) then block 1 (0, 1).
        expected_order = [(0, 14), (0, 15), (1, 0), (1, 1)]
        for tok_idx, (blk, off) in enumerate(expected_order):
            for k in range(2):
                for layer_idx in range(2):
                    expected = kv_caches[layer_idx][k, blk, :, 0, off, :]
                    assert torch.allclose(got[k, layer_idx, :, tok_idx, :], expected), (
                        f"mismatch at tok={tok_idx}"
                    )

    def test_multi_block_partial_gather_correctness(self):
        """Two unique blocks, partial slot subset on each.

        Verifies the slice gather copies exactly the requested slots
        (and only those) from each block into the chunk layout.
        """
        conn, kv_caches = self._prefilled_kv_caches(block_size=16, num_blocks=8)
        # Six tokens across two unique blocks (0 and 3): slots 0-2 and 48-50.
        slot_mapping = torch.tensor([0, 1, 2, 48, 49, 50], dtype=torch.long)
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(6), dtype=torch.float16)
        conn.from_gpu(mem, 0, 6, slot_mapping=slot_mapping)

        got = mem.tensor.reshape(2, 2, 2, 6, 8)  # [K, L, H, T, D]
        expected_order = [(0, 0), (0, 1), (0, 2), (3, 0), (3, 1), (3, 2)]
        for tok_idx, (blk, off) in enumerate(expected_order):
            for k in range(2):
                for layer_idx in range(2):
                    expected = kv_caches[layer_idx][k, blk, :, 0, off, :]
                    assert torch.allclose(got[k, layer_idx, :, tok_idx, :], expected), (
                        f"mismatch at tok={tok_idx} (blk={blk}, off={off}) "
                        f"k={k} layer={layer_idx}"
                    )


class TestBatchedOperations:
    def test_batched_from_gpu_processes_each_chunk(self):
        conn = _make_registered_connector(num_blocks=8, block_size=16)
        shape = conn.get_shape(16)
        mems = [MagicMock() for _ in range(3)]
        for m in mems:
            m.tensor = torch.zeros(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(48)

        spy = MagicMock(wraps=conn.from_gpu)
        conn.from_gpu = spy
        conn.batched_from_gpu(
            mems, [0, 16, 32], [16, 32, 48], slot_mapping=slot_mapping
        )
        # batched_from_gpu issues one D2H (from_gpu) call per memory_obj.
        assert spy.call_count == 3

    def test_batched_to_gpu_none_inputs_is_noop(self):
        conn = _make_registered_connector()
        spy = MagicMock(wraps=conn.to_gpu)
        conn.to_gpu = spy
        conn.batched_to_gpu(None, None, None)
        assert spy.call_count == 0


class TestDtypeHandling:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_get_shape_dtype_independent(self, dtype):
        conn = RBLNConnector(
            num_layers=4,
            num_kv_heads=8,
            head_dim=128,
            block_size=16,
            dtype=dtype,
        )
        assert conn.get_shape(32) == torch.Size([2, 4, 8, 32, 128])

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_round_trip_with_dtype(self, dtype):
        conn = RBLNConnector(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=dtype,
        )
        conn.register_kv_caches(_make_kv_caches(2, 4, 64, 16, dtype=dtype))

        shape = conn.get_shape(16)
        original = torch.randn(shape).to(dtype)

        write = MagicMock()
        write.tensor = original.clone()
        conn.to_gpu(write, 0, 16, slot_mapping=_make_slot_mapping(16))

        read = MagicMock()
        read.tensor = torch.zeros(shape, dtype=dtype)
        conn.from_gpu(read, 0, 16, slot_mapping=_make_slot_mapping(16))

        assert torch.allclose(original, read.tensor, atol=1e-3)


def _make_mla_kv_caches(
    num_layers: int,
    head_size: int,
    block_size: int,
    num_blocks: int = 4,
    dtype: torch.dtype = torch.float16,
) -> list[torch.Tensor]:
    """MLA layers: ``[num_blocks, block_size, head_size]`` (one latent, no K/V)."""
    return [
        torch.zeros(num_blocks, block_size, head_size, dtype=dtype)
        for _ in range(num_layers)
    ]


class SimulatedHostMLAConnector(RBLNConnector):
    """Exercise cross-device routing with CPU storage, without testing DMA.

    Only the routing device is simulated. Binding and transfer completion
    require separate tests on physical RBLN tensors.
    """

    def register_kv_caches(
        self,
        ordered_kv_caches: list[torch.Tensor],
        layer_group_indices: list[int] | None = None,
        aux_kv_caches: list[list[torch.Tensor]] | None = None,
        block_kv_caches: list[list[tuple[torch.Tensor, int]]] | None = None,
    ) -> None:
        super().register_kv_caches(
            ordered_kv_caches, layer_group_indices, aux_kv_caches, block_kv_caches
        )
        self._device = torch.device("meta")


class TestMLA:
    """MLA models: 3-D latent cache, chunks with a leading extent of 1."""

    @staticmethod
    def _registered(
        num_layers: int = 2,
        head_size: int = 8,
        block_size: int = 16,
        num_blocks: int = 8,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[RBLNConnector, list[torch.Tensor]]:
        conn = RBLNConnector(
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=head_size,
            block_size=block_size,
            dtype=dtype,
            use_mla=True,
        )
        kv_caches = _make_mla_kv_caches(
            num_layers, head_size, block_size, num_blocks, dtype
        )
        conn.register_kv_caches(kv_caches)
        return conn, kv_caches

    def test_get_shape_has_single_leading_dim(self):
        conn, _ = self._registered(num_layers=4, head_size=576)
        assert conn.get_shape(64) == torch.Size([1, 4, 1, 64, 576])

    @pytest.mark.parametrize(
        "slots",
        [
            list(range(16)),
            [14, 15, 16, 17],
            [0, 5, 9],
            [20, 21, 2, 3, 18, 19],
            [0, 5, 0, 18, 5],
        ],
        ids=["full", "crossing", "scattered", "interleaved", "repeated"],
    )
    def test_gather_preserves_token_order_without_intermediates(
        self, monkeypatch: pytest.MonkeyPatch, slots: list[int]
    ) -> None:
        conn, caches = self._registered()
        for cache in caches:
            cache.copy_(torch.randn_like(cache))
        mapping = torch.tensor(slots)
        destination = torch.empty(conn.get_shape(len(slots)), dtype=conn.dtype)
        expected = [cache.view(-1, 8)[mapping].clone() for cache in caches]
        intermediate = MagicMock(side_effect=AssertionError("Unexpected intermediate"))
        monkeypatch.setattr(torch, "empty", intermediate)
        monkeypatch.setattr(torch, "stack", intermediate)
        monkeypatch.setattr(torch, "cat", intermediate)
        for _ in range(2):
            conn.from_gpu(
                MagicMock(tensor=destination), 0, len(slots), slot_mapping=mapping
            )
            for layer, reference in enumerate(expected):
                assert torch.equal(destination[0, layer, 0], reference)
        conn.register_kv_caches(caches)
        conn.from_gpu(
            MagicMock(tensor=destination), 0, len(slots), slot_mapping=mapping
        )
        intermediate.assert_not_called()

    def test_gather_refuses_reclaimed_slots(self) -> None:
        conn, _ = self._registered()
        destination = torch.empty(conn.get_shape(2), dtype=conn.dtype)
        with pytest.raises(ValueError, match="reclaimed"):
            conn.from_gpu(
                MagicMock(tensor=destination), 0, 2, slot_mapping=torch.tensor([-1, 2])
            )

    @pytest.mark.parametrize(
        "slots",
        [list(range(16)), [2, 3, 4, 5], [14, 15, 16, 17]],
        ids=["full", "partial", "crossing"],
    )
    def test_host_restore_skips_upload_staging(
        self, monkeypatch: pytest.MonkeyPatch, slots: list[int]
    ) -> None:
        conn = SimulatedHostMLAConnector(
            num_layers=2,
            num_kv_heads=1,
            head_dim=8,
            block_size=16,
            dtype=torch.float16,
            use_mla=True,
        )
        caches = _make_mla_kv_caches(2, 8, 16, 8)
        conn.register_kv_caches(caches)
        mapping = torch.tensor(slots)
        source = torch.randn(conn.get_shape(len(slots)), dtype=conn.dtype)
        staging = MagicMock(side_effect=AssertionError("Unexpected staging allocation"))
        monkeypatch.setattr(torch, "empty", staging)
        conn.to_gpu(MagicMock(tensor=source), 0, len(slots), slot_mapping=mapping)
        staging.assert_not_called()
        for layer, cache in enumerate(caches):
            assert torch.equal(cache.view(-1, 8)[mapping], source[0, layer, 0])
            untouched = torch.ones(cache.numel() // 8, dtype=torch.bool)
            untouched[mapping] = False
            assert torch.count_nonzero(cache.view(-1, 8)[untouched]) == 0

    def test_caches_are_registered_as_is(self):
        conn, kv_caches = self._registered(num_blocks=4)
        assert conn._kv_caches[0] is kv_caches[0]
        assert conn._kv_caches[0].shape == torch.Size([4, 16, 8])

    def test_wrong_ndim_raises(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=1,
            head_dim=8,
            block_size=16,
            dtype=torch.float16,
            use_mla=True,
        )
        with pytest.raises(ValueError, match="Expected 3-D KV caches"):
            conn.register_kv_caches(_make_kv_caches(1, 1, 8, 16))

    def test_non_mla_connector_rejects_mla_caches(self):
        conn = RBLNConnector(
            num_layers=1,
            num_kv_heads=1,
            head_dim=8,
            block_size=16,
            dtype=torch.float16,
        )
        with pytest.raises(ValueError, match="Expected 6-D KV caches"):
            conn.register_kv_caches(_make_mla_kv_caches(1, 8, 16))

    @pytest.mark.parametrize("num_tokens", [16, 5, 32])
    def test_round_trip(self, num_tokens):
        conn, _ = self._registered(block_size=16, num_blocks=8)
        shape = conn.get_shape(num_tokens)
        original = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(num_tokens)

        write = MagicMock()
        write.tensor = original.clone()
        conn.to_gpu(write, 0, num_tokens, slot_mapping=slot_mapping)

        read = MagicMock()
        read.tensor = torch.zeros(shape, dtype=torch.float16)
        conn.from_gpu(read, 0, num_tokens, slot_mapping=slot_mapping)

        assert torch.allclose(original, read.tensor, atol=1e-3)

    def test_scattered_gather_reads_the_right_slots(self):
        conn, kv_caches = self._registered(block_size=16, num_blocks=8)
        for layer_idx, kv in enumerate(kv_caches):
            for blk in range(8):
                for slot in range(16):
                    kv[blk, slot, :] = layer_idx * 10_000 + blk * 100 + slot

        # Three tokens from three non-contiguous blocks (scattered offsets).
        slot_mapping = torch.tensor([3, 16 + 5, 5 * 16 + 1], dtype=torch.long)
        mem = MagicMock()
        mem.tensor = torch.zeros(conn.get_shape(3), dtype=torch.float16)
        conn.from_gpu(mem, 0, 3, slot_mapping=slot_mapping)

        got = mem.tensor.reshape(1, 2, 1, 3, 8)  # [1, L, H=1, T, D]
        for tok_idx, (blk, off) in enumerate([(0, 3), (1, 5), (5, 1)]):
            for layer_idx in range(2):
                expected = kv_caches[layer_idx][blk, off, :]
                assert torch.allclose(got[0, layer_idx, 0, tok_idx, :], expected), (
                    f"mismatch at tok={tok_idx} (blk={blk}, off={off}) "
                    f"layer={layer_idx}"
                )

    def test_scatter_writes_the_right_slots(self):
        conn, kv_caches = self._registered(block_size=16, num_blocks=8)
        num_tokens = 3
        chunk = torch.randn(conn.get_shape(num_tokens), dtype=torch.float16)
        slot_mapping = torch.tensor([3, 16 + 5, 5 * 16 + 1], dtype=torch.long)

        mem = MagicMock()
        mem.tensor = chunk
        conn.to_gpu(mem, 0, num_tokens, slot_mapping=slot_mapping)

        src = chunk.reshape(1, 2, 1, num_tokens, 8)
        for tok_idx, (blk, off) in enumerate([(0, 3), (1, 5), (5, 1)]):
            for layer_idx in range(2):
                assert torch.allclose(
                    kv_caches[layer_idx][blk, off, :], src[0, layer_idx, 0, tok_idx, :]
                )
        # Untouched slots stay zero.
        assert torch.all(kv_caches[0][0, 4, :] == 0)

    def test_noncontiguous_offsets_within_one_block(self):
        """Scattered offsets inside a single block: the index_select branch.

        The other scatter/gather tests use one slot per block, which the
        contiguity check treats as contiguous — only this shape reaches the
        MLA ``index_select`` / per-slot scatter code.
        """
        conn, kv_caches = self._registered(block_size=16, num_blocks=4)
        offsets = [0, 5, 9]
        slot_mapping = torch.tensor(offsets, dtype=torch.long)
        chunk = torch.randn(conn.get_shape(len(offsets)), dtype=torch.float16)

        write = MagicMock()
        write.tensor = chunk
        conn.to_gpu(write, 0, len(offsets), slot_mapping=slot_mapping)

        src = chunk.reshape(1, 2, 1, len(offsets), 8)
        for tok_idx, off in enumerate(offsets):
            for layer_idx in range(2):
                assert torch.allclose(
                    kv_caches[layer_idx][0, off, :], src[0, layer_idx, 0, tok_idx, :]
                )

        read = MagicMock()
        read.tensor = torch.zeros_like(chunk)
        conn.from_gpu(read, 0, len(offsets), slot_mapping=slot_mapping)
        assert torch.allclose(chunk, read.tensor, atol=1e-3)


class TestBlocksFirstLayout:
    """vLLM-RBLN's default allocation puts num_blocks ahead of the K/V axis.

    RBLN-SW/vllm-rbln#1114 made ``[num_blocks, 2, H, 1, S, D]`` what the
    rbln_custom_ops kernels read, and that is the default kernel path. The
    connector has to move the same bytes as it does for the K/V-first cache
    the rbln_triton_ops path still allocates.
    """

    @staticmethod
    def _blocks_first(
        num_layers: int = 2,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        block_size: int = 16,
        num_blocks: int = 4,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[RBLNConnector, list[torch.Tensor]]:
        conn = RBLNConnector(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
        )
        kv_caches = [
            torch.zeros(
                num_blocks, 2, num_kv_heads, 1, block_size, head_dim, dtype=dtype
            )
            for _ in range(num_layers)
        ]
        conn.register_kv_caches(kv_caches)
        return conn, kv_caches

    def test_full_block_round_trip(self):
        conn, _ = self._blocks_first()
        shape = conn.get_shape(16)
        original = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(16)

        write = MagicMock()
        write.tensor = original.clone()
        conn.to_gpu(write, 0, 16, slot_mapping=slot_mapping)

        read = MagicMock()
        read.tensor = torch.zeros(shape, dtype=torch.float16)
        conn.from_gpu(read, 0, 16, slot_mapping=slot_mapping)

        assert torch.allclose(original, read.tensor, atol=1e-3)

    def test_partial_block_round_trip(self):
        conn, _ = self._blocks_first()
        num_tokens = 5
        shape = conn.get_shape(num_tokens)
        original = torch.randn(shape, dtype=torch.float16)
        slot_mapping = _make_slot_mapping(num_tokens, offset=16 * 2 + 3)

        write = MagicMock()
        write.tensor = original.clone()
        conn.to_gpu(write, 0, num_tokens, slot_mapping=slot_mapping)

        read = MagicMock()
        read.tensor = torch.zeros(shape, dtype=torch.float16)
        conn.from_gpu(read, 0, num_tokens, slot_mapping=slot_mapping)

        assert torch.allclose(original, read.tensor, atol=1e-3)

    def test_the_staging_buffer_is_matched_to_the_allocation(self, monkeypatch):
        """``set_device_layout_like`` reads the allocation, not our view.

        It requires a reference covering its whole storage contiguously from
        offset 0, and rejects one that does not -- which is what a blocks-first
        cache's transposed view is. The staging path only runs when the chunk
        and the cache are on different devices, so it is unreachable from the
        public API on a CPU-only host; this drives it directly with the RBLN
        runtime stubbed out.
        """
        conn, kv_caches = self._blocks_first()
        seen: dict[str, torch.Tensor] = {}
        parent = types.ModuleType("torch_rbln")
        memory = types.ModuleType("torch_rbln.memory")
        memory.set_device_layout_like = lambda target, ref: seen.update(ref=ref)
        parent.memory = memory
        monkeypatch.setitem(sys.modules, "torch_rbln", parent)
        monkeypatch.setitem(sys.modules, "torch_rbln.memory", memory)

        conn._staging_views(16)

        ref = seen["ref"]
        assert ref.is_contiguous()
        assert ref.storage_offset() == 0
        assert ref.data_ptr() == kv_caches[0].data_ptr()

    def test_writes_land_where_the_kernels_read(self):
        """The bytes have to reach the allocated tensor's own K/V axis.

        A round trip alone would pass even if both directions agreed on a
        wrong axis, so read the caller's blocks-first tensor directly.
        """
        conn, kv_caches = self._blocks_first(num_layers=1, num_kv_heads=1, head_dim=2)
        shape = conn.get_shape(16)
        original = torch.randn(shape, dtype=torch.float16)

        write = MagicMock()
        write.tensor = original.clone()
        conn.to_gpu(write, 0, 16, slot_mapping=_make_slot_mapping(16, offset=16))

        # chunk [2, L, H, T, D] -> block 1 of [NB, 2, H, 1, S, D].
        k, v = original[0, 0, 0], original[1, 0, 0]
        assert torch.equal(kv_caches[0][1, 0, 0, 0], k)
        assert torch.equal(kv_caches[0][1, 1, 0, 0], v)


class TestHeterogeneousKVHeads:
    """A speculative draft with its own KV head count (eagle3 MHA heads over a
    GQA target: MiniMax-M2.7 has 8, its draft 48) lands as extra layers in the
    same KV cache. The chunk then holds one region per head group, and the
    transfer must gather/scatter each group from its own layers."""

    GROUPS = [(2, 4), (1, 12)]  # (layers, kv_heads): target x2, draft x1
    HEAD_DIM, BLOCK_SIZE, NUM_BLOCKS = 64, 16, 4

    def _connector(self) -> tuple[RBLNConnector, list[torch.Tensor]]:
        conn = RBLNConnector(
            num_layers=3,
            num_kv_heads=4,
            head_dim=self.HEAD_DIM,
            block_size=self.BLOCK_SIZE,
            dtype=torch.float16,
            kv_head_groups=self.GROUPS,
        )
        kv_caches = [
            torch.zeros(
                2,
                self.NUM_BLOCKS,
                h,
                1,
                self.BLOCK_SIZE,
                self.HEAD_DIM,
                dtype=torch.float16,
            )
            for n, h in self.GROUPS
            for _ in range(n)
        ]
        conn.register_kv_caches(kv_caches)
        return conn, kv_caches

    @staticmethod
    def _memory_obj(shapes: list[torch.Size]) -> TensorMemoryObj:
        """A real multi-group ``TensorMemoryObj`` laid out as upstream does."""
        dtypes = [torch.float16] * len(shapes)
        nbytes = sum(s.numel() * 2 for s in shapes)
        meta = MemoryObjMetadata(
            shape=shapes[0],
            dtype=torch.float16,
            address=0,
            phy_size=nbytes,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.KV_2LTD,
            shapes=shapes,
            dtypes=dtypes,
        )
        return TensorMemoryObj(torch.zeros(nbytes, dtype=torch.uint8), meta, None)

    def test_get_shapes_one_region_per_group(self):
        conn, _ = self._connector()
        assert conn.get_shapes(16) == [
            torch.Size([2, 2, 4, 16, 64]),
            torch.Size([2, 1, 12, 16, 64]),
        ]
        with pytest.raises(ValueError, match="use get_shapes"):
            conn.get_shape(16)

    @pytest.mark.parametrize("num_tokens,offset", [(16, 0), (5, 3), (24, 8)])
    def test_round_trip_moves_every_group(self, num_tokens, offset):
        conn, kv_caches = self._connector()
        shapes = conn.get_shapes(num_tokens)
        write = self._memory_obj(shapes)
        originals = []
        for i, shape in enumerate(shapes):
            t = write.get_tensor(i)
            t.copy_(torch.randn(shape, dtype=torch.float16))
            originals.append(t.clone())
        slot_mapping = _make_slot_mapping(num_tokens, offset)

        conn.to_gpu(write, 0, num_tokens, slot_mapping=slot_mapping)
        # The draft layer really received its 12 heads (not the target's 4).
        assert not torch.all(kv_caches[2] == 0)
        read = self._memory_obj(shapes)
        conn.from_gpu(read, 0, num_tokens, slot_mapping=slot_mapping)

        for i, original in enumerate(originals):
            assert torch.equal(original, read.get_tensor(i))

    def test_register_rejects_groups_the_chunk_was_not_sized_for(self):
        conn = RBLNConnector(
            num_layers=3,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        kv_caches = _make_kv_caches(2, 4, 64, 16) + _make_kv_caches(1, 12, 64, 16)
        with pytest.raises(ValueError, match=r"\[\(2, 4\), \(1, 12\)\]"):
            conn.register_kv_caches(kv_caches)

    def test_register_rejects_a_foreign_head_dim(self):
        conn = RBLNConnector(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            block_size=16,
            dtype=torch.float16,
        )
        with pytest.raises(ValueError, match="head_dim 32"):
            conn.register_kv_caches(_make_kv_caches(2, 4, 32, 16))

    def test_groups_must_cover_every_layer(self):
        with pytest.raises(ValueError, match="cover 2 layers, expected 3"):
            RBLNConnector(
                num_layers=3,
                num_kv_heads=4,
                head_dim=64,
                block_size=16,
                dtype=torch.float16,
                kv_head_groups=[(1, 4), (1, 12)],
            )

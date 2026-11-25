# SPDX-License-Identifier: Apache-2.0
"""Memory leak detection tests for RBLNConnector (device-tensor mode)."""

# Standard
from unittest.mock import MagicMock
import gc
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector


def _get_rss_mb() -> float:
    """Get current process RSS in MB (Linux only)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def _make_connector(num_layers: int = 2, block_size: int = 16) -> RBLNConnector:
    """Build a connector with CPU-resident KV caches that mimic RBLN layout."""
    conn = RBLNConnector(
        num_layers=num_layers,
        num_kv_heads=4,
        head_dim=64,
        block_size=block_size,
        dtype=torch.float16,
    )
    num_blocks = 4
    kv_caches = [
        torch.zeros(2, num_blocks, 4, 1, block_size, 64, dtype=torch.float16)
        for _ in range(num_layers)
    ]
    conn.register_kv_caches(kv_caches)
    return conn


def _make_slot_mapping(num_tokens: int) -> torch.Tensor:
    return torch.arange(num_tokens, dtype=torch.long)


class TestRepeatedCallsNoLeak:
    """Repeated transfer calls should not accumulate tensors or grow RSS."""

    def test_repeated_from_gpu_no_tensor_accumulation(self):
        conn = _make_connector()
        num_tokens = 16
        shape = conn.get_shape(num_tokens)
        slot_mapping = _make_slot_mapping(num_tokens)

        gc.collect()
        before_count = len(
            [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
        )

        for _ in range(100):
            mem = MagicMock()
            mem.tensor = torch.zeros(shape, dtype=torch.float16)
            conn.from_gpu(mem, 0, num_tokens, slot_mapping=slot_mapping)

        gc.collect()
        after_count = len(
            [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
        )
        growth = after_count - before_count
        assert growth < 50, f"Tensor count grew by {growth}, possible leak"

    def test_repeated_to_gpu_no_tensor_accumulation(self):
        conn = _make_connector()
        num_tokens = 16
        shape = conn.get_shape(num_tokens)
        slot_mapping = _make_slot_mapping(num_tokens)

        gc.collect()
        before_count = len(
            [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
        )

        for _ in range(100):
            mem = MagicMock()
            mem.tensor = torch.randn(shape, dtype=torch.float16)
            conn.to_gpu(mem, 0, num_tokens, slot_mapping=slot_mapping)

        gc.collect()
        after_count = len(
            [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
        )
        growth = after_count - before_count
        assert growth < 50, f"Tensor count grew by {growth}, possible leak"

    def test_repeated_ops_stable_rss(self):
        conn = _make_connector()
        num_tokens = 32
        shape = conn.get_shape(num_tokens)
        slot_mapping = _make_slot_mapping(num_tokens)

        for _ in range(10):
            mem = MagicMock()
            mem.tensor = torch.zeros(shape, dtype=torch.float16)
            conn.from_gpu(mem, 0, num_tokens, slot_mapping=slot_mapping)
            conn.to_gpu(mem, 0, num_tokens, slot_mapping=slot_mapping)

        gc.collect()
        rss_before = _get_rss_mb()
        if rss_before == 0:
            pytest.skip("Cannot read RSS on this platform")

        for _ in range(500):
            mem = MagicMock()
            mem.tensor = torch.zeros(shape, dtype=torch.float16)
            conn.from_gpu(mem, 0, num_tokens, slot_mapping=slot_mapping)
            conn.to_gpu(mem, 0, num_tokens, slot_mapping=slot_mapping)

        gc.collect()
        rss_after = _get_rss_mb()
        growth_mb = rss_after - rss_before
        assert growth_mb < 50, f"RSS grew by {growth_mb:.1f} MB, possible leak"

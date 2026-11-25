# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the RBLN connector tests.

``RBLNConnector`` moves KV with plain torch ops, so CPU tensors stand in for
the device KV caches; no NPU is needed. The modules under test still import
``torch_rbln`` and ``vllm``, which come from the RBLN SDK rather than from
this package's dependencies, so the directory is skipped without them.
"""

# Standard
from unittest.mock import MagicMock
import importlib.util

# Third Party
import pytest
import torch

_RUNTIME_DEPS = ("torch_rbln", "vllm")

if any(importlib.util.find_spec(name) is None for name in _RUNTIME_DEPS):
    collect_ignore_glob = ["*.py"]


@pytest.fixture
def rbln_params():
    """Default RBLN connector parameters."""
    return {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "block_size": 16,
        "dtype": torch.float16,
    }


@pytest.fixture
def kv_cache_names():
    """KV cache name strings for 32 layers."""
    return [f"model.layers.{i}.self_attn.kv_cache" for i in range(32)]


@pytest.fixture
def make_rbln_kv_caches():
    """Factory: build a list of RBLN-shaped KV cache tensors.

    Layout per layer: ``[2, num_blocks, num_kv_heads, 1, block_size, head_dim]``.
    Tensors live on CPU; in real use they live on ``device='rbln'``.
    """

    def _create(
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int = 4,
        dtype: torch.dtype = torch.float16,
    ) -> list[torch.Tensor]:
        return [
            torch.zeros(
                2, num_blocks, num_kv_heads, 1, block_size, head_dim, dtype=dtype
            )
            for _ in range(num_layers)
        ]

    return _create


@pytest.fixture
def mock_memory_obj():
    """Factory for creating mock MemoryObj with a real tensor."""

    def _create(shape, dtype=torch.float16):
        obj = MagicMock()
        obj.tensor = torch.randn(shape, dtype=dtype)
        return obj

    return _create

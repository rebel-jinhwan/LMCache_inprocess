# SPDX-License-Identifier: Apache-2.0
"""Tests for register_kv_caches with RBLN attention backend output.

The RBLN attention backend produces 6D KV cache tensors of shape
``[2, num_blocks, num_kv_heads, 1, block_size, head_size]``.  Upstream
LMCache stores caches as 5D ``[2, num_blocks, block_size, num_heads,
head_size]``.  ``RBLNLMCacheConnectorV1Impl.register_kv_caches`` must:

1. Normalize 6D → 5D for upstream's bookkeeping (squeeze + permute).
2. Hand the original 6D tensors to :class:`RBLNConnector` in layer order
   so device-tensor transfers see the native RBLN layout.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
from lmcache.inprocess.rbln.integration.vllm.cache_layout import CacheLayout, KVRegion
from lmcache.inprocess.rbln.integration.vllm.connector import (
    RBLNLMCacheConnectorV1Impl,
)
from lmcache.inprocess.v1.metadata import LMCacheMetadata

# ── helpers ──────────────────────────────────────────────────────────


def _rbln_get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[int, ...]:
    """Mirrors vllm-rbln RBLNFlashAttentionBackend.get_kv_cache_shape."""
    return (2, num_blocks, num_kv_heads, 1, block_size, head_size)


def _make_rbln_kv_caches(
    num_layers: int = 28,
    num_blocks: int = 8,
    block_size: int = 1024,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    shape = _rbln_get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
    return {
        f"model.layers.{i}.self_attn.kv_cache": torch.zeros(shape, dtype=dtype)
        for i in range(num_layers)
    }


def _make_impl_for_register(
    num_layers: int = 28,
    num_kv_heads: int = 8,
    head_size: int = 128,
    chunk_size: int = 256,
    kv_dtype: torch.dtype = torch.float16,
) -> RBLNLMCacheConnectorV1Impl:
    """Build a minimally-mocked impl that exercises the real register path.

    Skips the heavy ``__init__`` (which needs vLLM config + LMCache config +
    rebel runtime) but keeps upstream ``register_kv_caches`` and metadata
    intact, plus a mock ``gpu_connector`` to capture downstream calls.
    """
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = False
    impl.config = SimpleNamespace(extra_config={})
    impl.kv_caches = {}
    impl._cache_layout = CacheLayout(
        kv_regions=(
            KVRegion(
                num_kv_heads,
                kv_dtype,
                tuple(
                    f"model.layers.{i}.self_attn.kv_cache" for i in range(num_layers)
                ),
            ),
        ),
        aux_regions=(),
    )

    kv_shape = (num_layers, 2, chunk_size, num_kv_heads, head_size)
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=False,
        role="worker",
        served_model_name="test",
        chunk_size=chunk_size,
    )

    # First Party
    from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector

    manager = MagicMock()
    manager.lmcache_engine.metadata = metadata
    # spec=RBLNConnector so ``RBLNLMCacheConnectorV1Impl.register_kv_caches``
    # routes the ordered KV caches through this mock.
    manager.lmcache_engine.gpu_connector = MagicMock(spec=RBLNConnector)
    impl._manager = manager
    return impl


class TestRegisterRBLNKVCaches:
    """Cover the 6D shape that vllm-rbln actually produces."""

    def test_register_does_not_raise(self):
        impl = _make_impl_for_register()
        impl.register_kv_caches(_make_rbln_kv_caches())

    def test_metadata_single_group(self):
        # lmcache 0.4.6+ builds ``kv_layer_groups_manager`` in the GPU connector
        # (``_initialize_kv_cache_pointers``); our device connector transfers
        # per-layer via ``kv_shape`` and never builds it, so it stays ``None``
        # and metadata falls back to a single group derived from ``kv_shape``.
        impl = _make_impl_for_register(num_layers=28)
        impl.register_kv_caches(_make_rbln_kv_caches(num_layers=28))
        md = impl.lmcache_engine.metadata
        assert md.kv_layer_groups_manager is None
        assert md.get_num_groups() == 1
        assert md.kv_shape[0] == 28  # num_layers

    def test_hidden_dim_size_correct(self):
        impl = _make_impl_for_register(num_kv_heads=8, head_size=128)
        impl.register_kv_caches(_make_rbln_kv_caches(num_kv_heads=8, head_size=128))
        # hidden dim = num_kv_heads * head_size, surfaced via get_shapes fallback
        assert impl.lmcache_engine.metadata.get_shapes()[0][-1] == 8 * 128

    def test_get_shapes_matches_connector(self):
        num_layers, num_kv_heads, head_size = 28, 8, 128
        chunk_size = 256
        impl = _make_impl_for_register(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            chunk_size=chunk_size,
        )
        impl.register_kv_caches(
            _make_rbln_kv_caches(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
            )
        )
        shapes = impl.lmcache_engine.metadata.get_shapes(chunk_size)
        assert shapes[0] == torch.Size(
            [2, num_layers, chunk_size, num_kv_heads * head_size]
        )

    def test_ordered_kv_caches_passed_to_connector(self):
        """RBLNConnector receives the original 6D tensors in layer order."""
        impl = _make_impl_for_register(num_layers=4)
        kv_caches = _make_rbln_kv_caches(num_layers=4)
        impl.register_kv_caches(kv_caches)

        gpu_connector = impl.lmcache_engine.gpu_connector
        gpu_connector.register_kv_caches.assert_called_once()
        passed = gpu_connector.register_kv_caches.call_args[0][0]
        assert len(passed) == 4
        for tensor in passed:
            assert tensor.dim() == 6
            assert tensor.shape[3] == 1
        # Order: layer 0, 1, 2, 3
        for i, tensor in enumerate(passed):
            expected = kv_caches[f"model.layers.{i}.self_attn.kv_cache"]
            assert tensor.data_ptr() == expected.data_ptr()

    def test_normalized_shape_is_5d(self):
        impl = _make_impl_for_register()
        impl.register_kv_caches(_make_rbln_kv_caches(num_blocks=8, block_size=1024))
        for tensor in impl.kv_caches.values():
            assert tensor.dim() == 5
            assert tensor.shape == torch.Size([2, 8, 1024, 8, 128])


class TestRegisterStandardKVCaches:
    """5D tensors should pass through normalization unchanged."""

    def test_standard_5d_passthrough(self):
        impl = _make_impl_for_register(num_layers=4)
        kv_caches = {
            f"model.layers.{i}.self_attn.kv_cache": torch.zeros(
                2, 8, 16, 8, 128, dtype=torch.float16
            )
            for i in range(4)
        }
        impl.register_kv_caches(kv_caches)
        for tensor in impl.kv_caches.values():
            assert tensor.dim() == 5
            assert tensor.shape == torch.Size([2, 8, 16, 8, 128])
        assert impl.lmcache_engine.metadata.get_shapes()[0][-1] == 8 * 128


class TestRegisterEdgeCases:
    def test_heterogeneous_head_configs_forwarded(self):
        """Heterogeneous per-layer KV shapes are forwarded to the connector
        in order, unchanged.

        Multi-group bookkeeping is lmcache's ``KVLayerGroupsManager`` (built
        in the GPU connector). Our device connector transfers each layer's
        tensor directly and does not group them, so we verify the handoff
        rather than group structure.
        """
        impl = _make_impl_for_register(num_layers=4)
        kv_caches = {}
        for i in range(2):
            kv_caches[f"model.layers.{i}.self_attn.kv_cache"] = torch.zeros(
                *_rbln_get_kv_cache_shape(8, 1024, 8, 128), dtype=torch.float16
            )
        for i in range(2, 4):
            kv_caches[f"model.layers.{i}.self_attn.kv_cache"] = torch.zeros(
                *_rbln_get_kv_cache_shape(8, 1024, 4, 256), dtype=torch.float16
            )
        impl.register_kv_caches(kv_caches)

        passed = impl.lmcache_engine.gpu_connector.register_kv_caches.call_args[0][0]
        assert len(passed) == 4
        # dim 2 = num_kv_heads, dim 5 = head_size (RBLN 6D layout)
        assert [t.shape[2] for t in passed] == [8, 8, 4, 4]
        assert [t.shape[5] for t in passed] == [128, 128, 256, 256]

    def test_bfloat16_dtype_preserved(self):
        impl = _make_impl_for_register(num_layers=4, kv_dtype=torch.bfloat16)
        impl.register_kv_caches(
            _make_rbln_kv_caches(num_layers=4, dtype=torch.bfloat16)
        )
        # dtype surfaced via get_dtypes fallback (legacy kv_dtype)
        assert impl.lmcache_engine.metadata.get_dtypes()[0] == torch.bfloat16


class TestRegisterBlocksFirstKVCaches:
    """vLLM-RBLN's default allocation puts num_blocks ahead of the K/V axis.

    RBLN-SW/vllm-rbln#1114. Neither upstream's 5D bookkeeping nor the RBLN
    connector reads that layout, so both sides have to receive the K/V-first
    view instead.
    """

    @staticmethod
    def _blocks_first(
        num_layers: int = 4,
        num_blocks: int = 8,
        block_size: int = 1024,
        num_kv_heads: int = 8,
        head_size: int = 128,
    ) -> dict[str, torch.Tensor]:
        return {
            f"model.layers.{i}.self_attn.kv_cache": torch.zeros(
                num_blocks,
                2,
                num_kv_heads,
                1,
                block_size,
                head_size,
                dtype=torch.float16,
            )
            for i in range(num_layers)
        }

    def test_upstream_gets_the_kv_first_5d_view(self):
        impl = _make_impl_for_register(num_layers=4)

        impl.register_kv_caches(self._blocks_first())

        for tensor in impl.kv_caches.values():
            assert tensor.shape == torch.Size([2, 8, 1024, 8, 128])

    def test_the_connector_gets_the_allocation_not_a_view(self):
        """The connector has to receive what vLLM allocated, untouched.

        It lays its staging buffer out like the cache, and the RBLN runtime
        requires that reference to cover its whole storage contiguously from
        offset 0. A K/V-first view of a blocks-first cache does not, and the
        worker dies on its first retrieve. The connector transposes for its
        own indexing; that is its business, not this layer's.
        """
        impl = _make_impl_for_register(num_layers=4)
        kv_caches = self._blocks_first()

        impl.register_kv_caches(kv_caches)

        ordered = impl.lmcache_engine.gpu_connector.register_kv_caches.call_args[0][0]
        for i, tensor in enumerate(ordered):
            assert tensor is kv_caches[f"model.layers.{i}.self_attn.kv_cache"]
            assert tensor.is_contiguous()
            assert tensor.storage_offset() == 0

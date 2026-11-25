# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the RBLN NPU connector.

``RBLNConnector`` moves KV with plain torch ops, so CPU tensors stand in for
the ``device='rbln'`` KV caches vLLM-RBLN allocates: the 6-D attention cache
``[2, num_blocks, num_kv_heads, 1, block_size, head_dim]`` and the 3-D MLA
latent cache ``[num_blocks, block_size, head_size]``.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1 import gpu_connector
from lmcache.inprocess.v1.gpu_connector.rbln_connector import RBLNConnector
from lmcache.utils import EngineType

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8
BLOCK_SIZE = 16
NUM_BLOCKS = 4


def _metadata(**kwargs) -> SimpleNamespace:
    """The fields ``from_metadata`` reads from the RBLN service factory's
    metadata (``RBLNLMCacheMetadata`` in lmcache-rbln)."""
    fields = dict(
        kv_dtype=torch.float16,
        kv_shape=(NUM_LAYERS, 2, 256, NUM_KV_HEADS, HEAD_DIM),
        use_mla=False,
        local_worker_id=0,
        block_size=BLOCK_SIZE,
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _config(**kwargs) -> SimpleNamespace:
    fields = dict(
        use_layerwise=False,
        enable_blending=False,
        use_gpu_connector_v3=False,
        save_unfull_chunk=False,
        enable_pd=False,
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _chunk(tensor: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(tensor=tensor)


def _registered(use_mla: bool = False) -> tuple[RBLNConnector, list[torch.Tensor]]:
    connector = RBLNConnector.from_metadata(
        _metadata(
            use_mla=use_mla,
            kv_shape=(NUM_LAYERS, 1, 256, 1, HEAD_DIM)
            if use_mla
            else (NUM_LAYERS, 2, 256, NUM_KV_HEADS, HEAD_DIM),
        )
    )
    shape = (
        (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM)
        if use_mla
        else (2, NUM_BLOCKS, NUM_KV_HEADS, 1, BLOCK_SIZE, HEAD_DIM)
    )
    kv_caches = [torch.zeros(shape, dtype=torch.float16) for _ in range(NUM_LAYERS)]
    connector.register_kv_caches(kv_caches)
    return connector, kv_caches


class TestCreateGPUConnector:
    @pytest.fixture
    def rbln_device(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        torch_dev = MagicMock()
        monkeypatch.setattr(gpu_connector, "torch_device_type", "rbln")
        monkeypatch.setattr(gpu_connector, "torch_dev", torch_dev)
        return torch_dev

    @pytest.mark.parametrize("save_unfull_chunk", [False, True])
    def test_dispatches_rbln(
        self, rbln_device: MagicMock, save_unfull_chunk: bool
    ) -> None:
        connector = gpu_connector.CreateGPUConnector(
            _config(save_unfull_chunk=save_unfull_chunk), _metadata(), EngineType.VLLM
        )
        assert isinstance(connector, RBLNConnector)
        # Unfull chunks break the allocator's object grid.
        assert connector.slot_registration is not save_unfull_chunk
        # vLLM-RBLN already bound the worker's device.
        rbln_device.set_device.assert_not_called()

    def test_rejects_layerwise(self, rbln_device: MagicMock) -> None:
        with pytest.raises(ValueError, match="use_layerwise"):
            gpu_connector.CreateGPUConnector(
                _config(use_layerwise=True), _metadata(), EngineType.VLLM
            )


class TestFromMetadata:
    def test_single_group_comes_from_kv_shape(self) -> None:
        connector = RBLNConnector.from_metadata(_metadata())
        assert connector.num_layers == NUM_LAYERS
        assert connector.num_kv_heads == NUM_KV_HEADS
        assert connector.head_dim == HEAD_DIM
        assert connector.block_size == BLOCK_SIZE
        assert connector.kv_head_groups == [(NUM_LAYERS, NUM_KV_HEADS)]

    def test_sized_head_groups_override_kv_shape(self) -> None:
        groups = [(3, 4, 256), (1, 2, 512)]
        connector = RBLNConnector.from_metadata(_metadata(kv_head_groups=groups))
        assert (connector.num_layers, connector.num_kv_heads) == (4, 4)
        assert connector.head_dim == 256
        assert connector.kv_head_groups == groups

    def test_kv_head_size_and_buffer_dtype(self) -> None:
        connector = RBLNConnector.from_metadata(
            _metadata(
                use_mla=True,
                kv_head_groups=[(4, 1)],
                kv_head_size=512,
                kv_buffer_dtype=torch.int8,
                kv_dtype=torch.float8_e4m3fn,
            )
        )
        assert (connector.num_layers, connector.head_dim) == (4, 512)
        assert connector.dtype == torch.int8
        assert connector.use_mla

    def test_requires_block_size(self) -> None:
        with pytest.raises(ValueError, match="block size"):
            RBLNConnector.from_metadata(_metadata(block_size=None))


@pytest.mark.parametrize("use_mla", [False, True])
@pytest.mark.parametrize(
    "slots",
    [
        list(range(BLOCK_SIZE)),  # one full block
        list(range(3, 11)),  # part of a block
        [*range(40, 48), *range(5, 13)],  # scattered across blocks
    ],
    ids=["full", "partial", "scattered"],
)
def test_round_trip(use_mla: bool, slots: list[int]) -> None:
    connector, kv_caches = _registered(use_mla)
    sentinel = 7.0
    for kv in kv_caches:
        kv.fill_(sentinel)
    num_tokens = len(slots)
    slot_mapping = torch.tensor(slots, dtype=torch.long)
    original = torch.randn(connector.get_shape(num_tokens), dtype=torch.float16)

    connector.to_gpu(_chunk(original.clone()), 0, num_tokens, slot_mapping=slot_mapping)
    read = _chunk(torch.zeros_like(original))
    connector.from_gpu(read, 0, num_tokens, slot_mapping=slot_mapping)

    assert torch.equal(read.tensor, original)
    # Slots the chunk does not cover keep their contents.
    written = torch.zeros(NUM_BLOCKS * BLOCK_SIZE, dtype=torch.bool)
    written[slot_mapping] = True
    for kv in kv_caches:
        token_major = kv if use_mla else kv.permute(1, 4, 0, 2, 3, 5)
        per_token = token_major.reshape(NUM_BLOCKS * BLOCK_SIZE, -1)
        assert torch.all(per_token[~written] == sentinel)


def test_host_buffers_are_unregistered_by_default() -> None:
    connector, _ = _registered()
    assert connector.register_host_buffer(torch.zeros(16)) is False


@pytest.mark.parametrize("direction", ["from_gpu", "to_gpu"])
def test_transfers_offer_each_host_chunk_for_registration(
    monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    """``register_host_buffer`` is the hook a deployment replaces to register
    its host memory; every transferred host chunk goes through it."""
    connector, _ = _registered()
    seen: list[int] = []
    monkeypatch.setattr(
        connector,
        "register_host_buffer",
        lambda buffer, allocation=None: seen.append(buffer.data_ptr()) or False,
    )
    chunk = _chunk(torch.zeros(connector.get_shape(8), dtype=torch.float16))
    # Partial blocks skip the full-block gather's own registration check.
    getattr(connector, direction)(chunk, 0, 8, slot_mapping=torch.arange(8))
    assert seen == [chunk.tensor.data_ptr()]

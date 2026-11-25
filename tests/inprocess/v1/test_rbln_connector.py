# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN NPU connector.

``RBLNConnector`` moves KV with plain torch ops, so CPU tensors stand in for
the ``device='rbln'`` KV caches vLLM-RBLN allocates: the 6-D attention cache
``[2, num_blocks, num_kv_heads, 1, block_size, head_dim]`` and the 3-D MLA
latent cache ``[num_blocks, block_size, head_size]``. Where torch-rbln sees an
NPU (the RBLN Buildkite pipeline), the round trip also runs on real
``device='rbln'`` caches, which takes the torch-rbln staging path.
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


def _rbln_device_count() -> int:
    """Return how many NPUs torch-rbln sees, or 0 without torch-rbln."""
    rbln = getattr(torch, "rbln", None)
    return rbln.device_count() if rbln is not None else 0


KV_DEVICES = [
    "cpu",
    pytest.param(
        "rbln",
        marks=pytest.mark.skipif(
            _rbln_device_count() == 0, reason="needs torch-rbln and an RBLN NPU"
        ),
    ),
]


def _metadata(use_mla: bool = False) -> SimpleNamespace:
    """The ``LMCacheMetadata`` fields ``from_metadata`` reads."""
    return SimpleNamespace(
        kv_dtype=torch.float16,
        kv_shape=(NUM_LAYERS, 1, 256, 1, HEAD_DIM)
        if use_mla
        else (NUM_LAYERS, 2, 256, NUM_KV_HEADS, HEAD_DIM),
        use_mla=use_mla,
        local_worker_id=0,
    )


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


def _kv_cache(shape: tuple[int, ...], fill: float, device: str) -> torch.Tensor:
    """A KV cache filled with ``fill``, allocated on ``device``.

    On RBLN, ``torch.empty`` and ``torch.zeros`` only reserve the tensor: the
    runtime places it on the NPU later. vLLM-RBLN's caches are bound to device
    memory, and ``set_device_layout_like`` needs a bound reference, so the
    cache is bound before the values are uploaded.
    """
    host = torch.full(shape, fill, dtype=torch.float16)
    if device == "cpu":
        return host
    # Third Party
    from torch_rbln.memory import bind_device_memory

    cache = torch.empty(shape, dtype=torch.float16, device=device)
    bind_device_memory(cache)
    cache.copy_(host)
    return cache


def _kv_caches(
    use_mla: bool = False,
    fill: float = 0.0,
    blocks_first: bool = False,
    device: str = "cpu",
) -> list[torch.Tensor]:
    if use_mla:
        shape: tuple[int, ...] = (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM)
    elif blocks_first:
        shape = (NUM_BLOCKS, 2, NUM_KV_HEADS, 1, BLOCK_SIZE, HEAD_DIM)
    else:
        shape = (2, NUM_BLOCKS, NUM_KV_HEADS, 1, BLOCK_SIZE, HEAD_DIM)
    return [_kv_cache(shape, fill, device) for _ in range(NUM_LAYERS)]


class TestCreateGPUConnector:
    @pytest.fixture
    def rbln_device(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        torch_dev = MagicMock()
        monkeypatch.setattr(gpu_connector, "torch_device_type", "rbln")
        monkeypatch.setattr(gpu_connector, "torch_dev", torch_dev)
        return torch_dev

    def test_dispatches_rbln(self, rbln_device: MagicMock) -> None:
        connector = gpu_connector.CreateGPUConnector(
            _config(), _metadata(), EngineType.VLLM
        )
        assert isinstance(connector, RBLNConnector)
        # vLLM-RBLN already bound the worker's device.
        rbln_device.set_device.assert_not_called()

    def test_rejects_layerwise(self, rbln_device: MagicMock) -> None:
        with pytest.raises(ValueError, match="use_layerwise"):
            gpu_connector.CreateGPUConnector(
                _config(use_layerwise=True), _metadata(), EngineType.VLLM
            )


@pytest.mark.parametrize("use_mla", [False, True])
def test_from_metadata(use_mla: bool) -> None:
    connector = RBLNConnector.from_metadata(_metadata(use_mla))
    heads = 1 if use_mla else NUM_KV_HEADS
    assert connector.get_shape(5) == (2 - use_mla, NUM_LAYERS, heads, 5, HEAD_DIM)
    assert connector.block_size is None


@pytest.mark.parametrize("blocks_first", [False, True])
def test_registration_reads_block_size(blocks_first: bool) -> None:
    connector = RBLNConnector.from_metadata(_metadata())
    connector.register_kv_caches(_kv_caches(blocks_first=blocks_first))
    assert connector.block_size == BLOCK_SIZE


def test_registration_rejects_other_geometry() -> None:
    connector = RBLNConnector.from_metadata(_metadata())
    caches = [
        torch.zeros(2, NUM_BLOCKS, NUM_KV_HEADS + 1, 1, BLOCK_SIZE, HEAD_DIM)
        for _ in range(NUM_LAYERS)
    ]
    with pytest.raises(ValueError, match="KV cache 0"):
        connector.register_kv_caches(caches)


def test_transfer_needs_caches() -> None:
    connector = RBLNConnector.from_metadata(_metadata())
    chunk = _chunk(torch.zeros(connector.get_shape(4), dtype=torch.float16))
    with pytest.raises(RuntimeError, match="not registered"):
        connector.to_gpu(chunk, 0, 4, slot_mapping=torch.arange(4))


def test_rejects_negative_slots() -> None:
    connector = RBLNConnector.from_metadata(_metadata())
    chunk = _chunk(torch.zeros(connector.get_shape(2), dtype=torch.float16))
    with pytest.raises(ValueError, match="negative slot"):
        connector.from_gpu(
            chunk, 0, 2, slot_mapping=torch.tensor([3, -1]), kvcaches=_kv_caches()
        )


@pytest.mark.parametrize("device", KV_DEVICES)
@pytest.mark.parametrize("use_mla", [False, True])
@pytest.mark.parametrize(
    "slots",
    [
        list(range(BLOCK_SIZE)),  # one full block
        list(range(3, 11)),  # part of a block
        [*range(40, 48), *range(5, 13)],  # scattered across blocks
        [*range(2 * BLOCK_SIZE, 3 * BLOCK_SIZE), *range(BLOCK_SIZE)],  # two blocks
    ],
    ids=["full", "partial", "scattered", "two-blocks"],
)
def test_round_trip(use_mla: bool, slots: list[int], device: str) -> None:
    """The vLLM adapter never registers the caches; each transfer passes them
    as ``kvcaches``."""
    sentinel = 7.0
    kv_caches = _kv_caches(use_mla, fill=sentinel, device=device)
    connector = RBLNConnector.from_metadata(_metadata(use_mla))
    num_tokens = len(slots)
    slot_mapping = torch.tensor(slots, dtype=torch.long)
    original = torch.randn(connector.get_shape(num_tokens), dtype=torch.float16)

    connector.to_gpu(
        _chunk(original.clone()),
        0,
        num_tokens,
        slot_mapping=slot_mapping,
        kvcaches=kv_caches,
    )
    read = _chunk(torch.zeros_like(original))
    connector.from_gpu(
        read, 0, num_tokens, slot_mapping=slot_mapping, kvcaches=kv_caches
    )

    assert torch.equal(read.tensor, original)
    # Slots the chunk does not cover keep their contents.
    written = torch.zeros(NUM_BLOCKS * BLOCK_SIZE, dtype=torch.bool)
    written[slot_mapping] = True
    for kv in kv_caches:
        kv = kv.cpu()
        token_major = kv if use_mla else kv.permute(1, 4, 0, 2, 3, 5)
        per_token = token_major.reshape(NUM_BLOCKS * BLOCK_SIZE, -1)
        assert torch.all(per_token[~written] == sentinel)

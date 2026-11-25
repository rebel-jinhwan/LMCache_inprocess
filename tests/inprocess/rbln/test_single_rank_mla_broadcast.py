# SPDX-License-Identifier: Apache-2.0
"""A single-rank MLA engine restores from host chunks, not from device copies.

LMCache turns ``save_only_first_rank`` on for every MLA model. On retrieve the
leader copies each chunk into a fresh device tensor for a TP broadcast and
restores the KV cache from that copy. With one rank nobody receives the
broadcast; ``lmcache.inprocess.rbln`` patches ``_broadcast_or_receive_memory_objs`` so
the host chunks themselves become the substitutes (rebellions-sw/lmcache-rbln
#310). These tests drive upstream's real ``LMCacheEngine.retrieve`` so they
also pin the upstream contract the patch relies on.
"""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.rbln import _patches
from lmcache.inprocess.v1.cache_engine import LMCacheEngine

NUM_TOKENS = 8


def _chunk(index: int) -> MagicMock:
    """A host memory object whose device copy is observable."""
    mo = MagicMock(name=f"chunk{index}")
    mo.raw_tensor = MagicMock(name=f"raw{index}")
    mo.raw_tensor.to.side_effect = lambda device: MagicMock(name=f"copy{index}")
    mo.metadata.to_dict.return_value = {}
    mo.get_size.return_value = 1024
    mo.is_pinned = False
    return mo


def _engine(world_size: int, chunks: list[MagicMock]) -> Any:
    """An ``LMCacheEngine`` with only what ``retrieve()`` touches."""
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.metadata = SimpleNamespace(
        world_size=world_size, worker_id=0, first_rank=0, is_first_rank=lambda: True
    )
    engine.save_only_first_rank = True
    engine.async_loading = False
    engine.remove_after_retrieve = False
    engine.retrieve_locations = None
    engine.broadcast_fn = MagicMock(name="broadcast_fn")
    engine.broadcast_object_fn = MagicMock(name="broadcast_object_fn")
    engine.broadcast_stream = MagicMock(name="broadcast_stream")
    engine.gpu_connector = MagicMock(name="gpu_connector", spec=["batched_to_gpu"])
    engine.storage_manager = MagicMock(name="storage_manager")
    engine._leader_gpu_substitute_objs = []
    engine.is_healthy = lambda: True
    engine._is_passive = lambda: False
    engine._is_sync_pd_backend = lambda: False
    engine._log_kvcache_for_check = lambda *a, **k: None
    stats = MagicMock(name="retrieve_stats")
    stats.time_to_retrieve.return_value = 0.001
    engine.stats_monitor = MagicMock(name="stats_monitor")
    engine.stats_monitor.on_retrieve_request.return_value = stats

    def process_tokens(tokens, mask, ret_mask, **kwargs):  # noqa: ANN001, ANN202
        step = NUM_TOKENS // len(chunks)
        out = [("key", mo, i * step, (i + 1) * step) for i, mo in enumerate(chunks)]
        ret_mask[:] = True
        return out, sum(mo.get_size() for mo in chunks)

    engine._process_tokens_internal = process_tokens
    return engine


@pytest.fixture
def no_device_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """``retrieve()`` enters ``torch_dev.stream(...)``; no device in unit tests."""
    monkeypatch.setattr(sys.modules[LMCacheEngine.__module__], "torch_dev", MagicMock())


def _restored_from(engine: Any) -> list[Any]:
    engine.gpu_connector.batched_to_gpu.assert_called_once()
    return list(engine.gpu_connector.batched_to_gpu.call_args.args[0])


def test_the_patch_is_installed() -> None:
    assert (
        LMCacheEngine._broadcast_or_receive_memory_objs
        is _patches._patched_broadcast_or_receive_memory_objs
    )


@pytest.mark.usefixtures("no_device_stream")
def test_single_rank_restores_from_the_host_chunks() -> None:
    chunks = [_chunk(0), _chunk(1)]
    engine = _engine(world_size=1, chunks=chunks)

    ret_mask = LMCacheEngine.retrieve(engine, torch.arange(NUM_TOKENS))

    assert bool(ret_mask.all())
    restored = _restored_from(engine)
    assert all(a is b for a, b in zip(restored, chunks, strict=True))
    for mo in chunks:
        mo.raw_tensor.to.assert_not_called()
        mo.ref_count_down.assert_called_once()  # released once, as upstream does
    engine.broadcast_fn.assert_not_called()
    engine.broadcast_object_fn.assert_not_called()
    assert engine._leader_gpu_substitute_objs == []  # retrieve()'s finally


@pytest.mark.usefixtures("no_device_stream")
def test_upstream_restores_from_device_copies_without_the_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract the patch relies on. If upstream stops restoring from
    ``_leader_gpu_substitute_objs``, this fails: revisit the patch."""
    monkeypatch.setattr(
        LMCacheEngine,
        "_broadcast_or_receive_memory_objs",
        _patches._original_broadcast_or_receive_memory_objs,
    )
    chunks = [_chunk(0), _chunk(1)]
    engine = _engine(world_size=1, chunks=chunks)

    LMCacheEngine.retrieve(engine, torch.arange(NUM_TOKENS))

    restored = _restored_from(engine)
    assert not any(a is b for a, b in zip(restored, chunks, strict=True))
    for mo in chunks:
        mo.raw_tensor.to.assert_called_once()
    assert engine.broadcast_fn.call_count == len(chunks)


@pytest.mark.parametrize("world_size", [2, 4])
def test_more_than_one_rank_keeps_the_broadcast(
    monkeypatch: pytest.MonkeyPatch, world_size: int
) -> None:
    original = MagicMock(name="original")
    monkeypatch.setattr(
        _patches, "_original_broadcast_or_receive_memory_objs", original
    )
    engine = _engine(world_size=world_size, chunks=[_chunk(0)])
    chunks = [("key", _chunk(0), 0, NUM_TOKENS)]
    ret_mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)

    LMCacheEngine._broadcast_or_receive_memory_objs(engine, chunks, ret_mask)

    original.assert_called_once_with(engine, chunks, ret_mask)

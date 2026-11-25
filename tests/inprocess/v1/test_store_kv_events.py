# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the ``BlockStored`` events ``LMCacheEngine.store`` emits.

Each stored chunk's event must carry exactly that chunk's tokens and name the
chunk right before it as its parent: a cache-aware router (e.g. llm-d) keys a
prefix chain by those parents and reads a missing parent as the start of a
prompt.
"""

# Standard
from collections.abc import Iterator

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.inprocess.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.inprocess.v1.metadata import LMCacheMetadata

CHUNK = 16
ENGINE = "test_store_kv_events"


@pytest.fixture
def engine() -> Iterator[LMCacheEngine]:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK,
        local_cpu=True,
        max_local_cpu_size=0.05,
        enable_kv_events=True,
    )
    kv_shape = (2, 2, CHUNK, 2, 8)
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=kv_shape,
        chunk_size=CHUNK,
    )
    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE,
        config,
        metadata,
        MockGPUConnector(kv_shape=kv_shape),
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )
    try:
        engine.post_init()
        yield engine
    finally:
        LMCacheEngineBuilder.destroy(ENGINE)


def _chunk_hashes(engine: LMCacheEngine, tokens: torch.Tensor) -> list[int]:
    return [
        chunk_hash
        for _, _, chunk_hash in engine.token_database.process_tokens(
            tokens, make_key=False
        )
    ]


def test_events_carry_each_chunks_own_tokens(engine: LMCacheEngine) -> None:
    tokens = torch.arange(3 * CHUNK)
    engine.store(tokens)
    events = list(engine.get_kv_events())
    assert [e.token_ids for e in events] == [
        tokens[i * CHUNK : (i + 1) * CHUNK].tolist() for i in range(3)
    ]
    assert [e.block_size for e in events] == [CHUNK] * 3


def test_parents_chain_the_chunks(engine: LMCacheEngine) -> None:
    tokens = torch.arange(3 * CHUNK)
    hashes = _chunk_hashes(engine, tokens)
    engine.store(tokens)
    events = list(engine.get_kv_events())
    assert [e.block_hashes for e in events] == [[h] for h in hashes]
    assert [e.parent_block_hash for e in events] == [None, *hashes[:-1]]


def test_masked_prefix_still_names_its_parent(engine: LMCacheEngine) -> None:
    """A store that skips an already-cached prefix links its first chunk to
    the prefix's last chunk, not to 0."""
    tokens = torch.arange(3 * CHUNK)
    hashes = _chunk_hashes(engine, tokens)
    mask = torch.ones(3 * CHUNK, dtype=torch.bool)
    mask[:CHUNK] = False
    engine.store(tokens, mask=mask)
    events = list(engine.get_kv_events())
    assert [e.block_hashes for e in events] == [[h] for h in hashes[1:]]
    assert [e.parent_block_hash for e in events] == hashes[:2]
    assert events[0].token_ids == tokens[CHUNK : 2 * CHUNK].tolist()

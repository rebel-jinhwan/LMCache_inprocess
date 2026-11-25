# SPDX-License-Identifier: Apache-2.0
"""LMCacheConnectorV1Dynamic bridges LMCache's store events to vLLM.

Workers report the chunks they stored; vLLM's ``KVOutputAggregator`` merges
the reports, one vote per reporting worker; the scheduler connector keeps the
events every voting worker agreed on and hands them to vLLM's KV event
publisher through ``take_events``.
"""

# Standard
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_events import BlockRemoved, BlockStored  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.utils import (  # noqa: E402
    KVOutputAggregator,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402

# First Party
from lmcache.inprocess.integration.vllm.kv_events import (  # noqa: E402
    ConsensusKVEvents,
)
from lmcache.inprocess.integration.vllm.lmcache_connector_v1 import (  # noqa: E402
    LMCacheConnectorV1Dynamic,
)


def _stored(chunk_hash: int, parent: Optional[int] = None) -> SimpleNamespace:
    """An LMCache ``CacheStoreEvent`` for one chunk."""
    return SimpleNamespace(
        block_hashes=[chunk_hash],
        parent_block_hash=parent,
        token_ids=[chunk_hash, chunk_hash + 1],
        block_size=2,
        lora_id=None,
        medium="cpu",
        lora_name=None,
    )


def _block(chunk_hash: int, parent: Optional[int] = None) -> BlockStored:
    return BlockStored(
        block_hashes=[chunk_hash],
        parent_block_hash=parent,
        token_ids=[chunk_hash, chunk_hash + 1],
        block_size=2,
        lora_id=None,
        medium="cpu",
        lora_name=None,
    )


class _Engine:
    """The LMCacheEngine surface the worker connector reads."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        save_only_first_rank: bool = False,
        first_rank: bool = True,
    ) -> None:
        self.kv_events_enabled = enabled
        self.save_only_first_rank = save_only_first_rank
        self.metadata = SimpleNamespace(is_first_rank=lambda: first_rank)
        self.queue: list[SimpleNamespace] = []

    def get_kv_events(self) -> list[SimpleNamespace]:
        events, self.queue = self.queue, []
        return events


def _connector(engine: Optional[_Engine] = None) -> LMCacheConnectorV1Dynamic:
    # Neither side's hooks touch the engine-building init.
    connector = object.__new__(LMCacheConnectorV1Dynamic)
    connector._lmcache_engine = SimpleNamespace(lmcache_engine=engine)
    connector._kv_cache_events = None
    return connector


def _step(scheduler: LMCacheConnectorV1Dynamic, workers: list[_Engine]) -> None:
    """Run one step's worker reports through vLLM's aggregator to the scheduler."""
    outputs = [
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_connector_output=KVConnectorOutput(
                kv_cache_events=_connector(w).get_kv_connector_kv_cache_events()
            ),
        )
        for w in workers
    ]
    merged = KVOutputAggregator(expected_finished_count=len(workers)).aggregate(
        outputs, output_rank=0
    )
    assert merged.kv_connector_output is not None
    scheduler.update_connector_output(merged.kv_connector_output)


def test_store_reported_by_every_rank_is_published_once() -> None:
    ranks = [_Engine(), _Engine()]
    for rank in ranks:
        rank.queue = [_stored(1), _stored(2, parent=1)]
    scheduler = _connector()

    _step(scheduler, ranks)

    assert list(scheduler.take_events()) == [_block(1), _block(2, parent=1)]
    assert list(scheduler.take_events()) == []


def test_store_missing_on_one_rank_is_not_published() -> None:
    # Rank 1 stored nothing this step (it failed, or skipped the chunk). Its
    # empty batch is a vote, so the instance does not advertise the chunk.
    ranks = [_Engine(), _Engine()]
    ranks[0].queue = [_stored(1)]
    scheduler = _connector()

    _step(scheduler, ranks)

    assert list(scheduler.take_events()) == []


def test_passive_mla_rank_does_not_vote() -> None:
    # Under save_only_first_rank only the first rank stores; the others hold
    # the same MLA KV and must not veto its stores.
    ranks = [
        _Engine(save_only_first_rank=True, first_rank=True),
        _Engine(save_only_first_rank=True, first_rank=False),
    ]
    ranks[0].queue = [_stored(1)]
    scheduler = _connector()

    _step(scheduler, ranks)

    assert list(scheduler.take_events()) == [_block(1)]


def test_prefix_stored_twice_in_a_step_counts_as_one_vote() -> None:
    # Two requests of one step store the same prefix on rank 0 only; the
    # duplicate must not stand in for rank 1's vote.
    ranks = [_Engine(), _Engine()]
    ranks[0].queue = [_stored(1), _stored(1)]
    scheduler = _connector()

    _step(scheduler, ranks)

    assert list(scheduler.take_events()) == []


def test_steps_before_take_events_are_not_merged_as_votes() -> None:
    ranks = [_Engine(), _Engine()]
    scheduler = _connector()
    ranks[0].queue = [_stored(1)]
    ranks[1].queue = [_stored(1)]
    _step(scheduler, ranks)
    # Rank 0 alone stores chunk 2 in the next step; the first step's votes
    # must not make it look agreed.
    ranks[0].queue = [_stored(2)]
    _step(scheduler, ranks)

    assert list(scheduler.take_events()) == [_block(1)]


def test_disabled_events_drain_the_queue_without_voting() -> None:
    engine = _Engine(enabled=False)
    engine.queue = [_stored(1)]

    assert _connector(engine).get_kv_connector_kv_cache_events() is None
    assert engine.queue == []


def test_worker_without_engine_reports_nothing() -> None:
    assert _connector(None).get_kv_connector_kv_cache_events() is None


def test_foreign_events_are_ignored() -> None:
    scheduler = _connector()
    scheduler.update_connector_output(KVConnectorOutput(kv_cache_events=MagicMock()))
    assert list(scheduler.take_events()) == []


def test_removal_from_any_rank_is_published_and_drops_its_store() -> None:
    removed = BlockRemoved(block_hashes=[1], medium="cpu")
    events = ConsensusKVEvents()
    events.add_events([_block(1), _block(2)])
    events.increment_workers()
    events.add_events([_block(1), _block(2), removed])

    assert sorted(map(repr, events.aggregate().get_all_events())) == sorted(
        map(repr, [_block(2), removed])
    )
    assert events.get_number_of_workers() == 1


@pytest.mark.parametrize("bad", [0, -1])
def test_worker_counts_must_be_positive(bad: int) -> None:
    with pytest.raises(ValueError):
        ConsensusKVEvents(num_workers=bad)
    with pytest.raises(ValueError):
        ConsensusKVEvents().increment_workers(bad)

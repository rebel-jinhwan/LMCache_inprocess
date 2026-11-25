# SPDX-License-Identifier: Apache-2.0
"""Rank consensus for the KV cache events LMCache reports to vLLM.

Every worker that stores KV reports one batch per step, including an empty
batch when it stored nothing: vLLM's ``KVOutputAggregator`` counts each
reporting worker as one vote. A store is advertised only once every voting
worker reports it, since a router that sends a prefix to this instance needs
all of its ranks (every TP shard, every PP stage) to hold the chunk. A removal
from any worker invalidates the chunk for the whole instance.
"""

# Standard
from collections import Counter

# Third Party
from vllm.distributed.kv_events import (
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
)


class ConsensusKVEvents(KVConnectorKVEvents):
    """Intersect store events and union removal events across workers.

    A worker deduplicates its own batch before reporting it, so a count of an
    event is the number of workers that reported it. :meth:`aggregate` folds
    one step's votes; events of distinct steps must not be merged as votes.
    """

    def __init__(self, num_workers: int = 1) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self._num_workers = num_workers
        self._events: Counter[KVCacheEvent] = Counter()

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._events.update(events)

    def aggregate(self) -> "ConsensusKVEvents":
        removed = {
            (event.medium, block_hash)
            for event in self._events
            if isinstance(event, BlockRemoved)
            for block_hash in event.block_hashes
        }
        events: list[KVCacheEvent] = []
        for event, count in self._events.items():
            if isinstance(event, BlockRemoved):
                events.append(event)
            elif count == self._num_workers:
                # A store of a chunk that some worker dropped in the same step
                # is not held by the whole instance.
                if isinstance(event, BlockStored) and any(
                    (event.medium, block_hash) in removed
                    for block_hash in event.block_hashes
                ):
                    continue
                events.append(event)
        self._events = Counter(events)
        self._num_workers = 1
        return self

    def increment_workers(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self._num_workers += count

    def get_all_events(self) -> list[KVCacheEvent]:
        return list(self._events.elements())

    def get_number_of_workers(self) -> int:
        return self._num_workers

    def clear_events(self) -> None:
        self._events.clear()
        self._num_workers = 1

    def __repr__(self) -> str:
        return (
            f"<ConsensusKVEvents workers={self._num_workers} "
            f"events={len(self._events)}>"
        )

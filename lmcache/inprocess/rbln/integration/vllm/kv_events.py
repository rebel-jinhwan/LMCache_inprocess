# SPDX-License-Identifier: Apache-2.0
"""Rank consensus for external KV cache residency.

Every storing worker contributes one batch, including empty batches. Stores
require consensus; a removal from any worker invalidates the pod's full KV.
The vLLM executor merges batches through the public LMCacheKVEvents interface.
"""

# Standard
from collections import Counter

# Third Party
from typing_extensions import override
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheKVEvents,
)


class RBLNKVEvents(LMCacheKVEvents):
    """Intersect store votes and union removals across storing ranks."""

    def __init__(self, num_workers: int = 1) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self._num_workers = num_workers
        self._events: Counter[KVCacheEvent] = Counter()

    @override
    def add_events(self, events: list[KVCacheEvent]) -> None:
        # Each worker deduplicates its own batch before the executor merges it.
        self._events.update(events)

    @override
    def aggregate(self) -> "RBLNKVEvents":
        removed = {
            (event.medium, block_hash)
            for event in self._events
            if isinstance(event, BlockRemoved)
            for block_hash in event.block_hashes
        }
        events = []
        for event, count in self._events.items():
            if isinstance(event, BlockRemoved):
                events.append(event)
            elif count == self._num_workers:
                if isinstance(event, BlockStored) and any(
                    (event.medium, block_hash) in removed
                    for block_hash in event.block_hashes
                ):
                    continue
                events.append(event)
        self._events = Counter(events)
        self._num_workers = 1
        return self

    @override
    def increment_workers(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self._num_workers += count

    @override
    def get_all_events(self) -> list[KVCacheEvent]:
        return list(self._events.elements())

    @override
    def get_number_of_workers(self) -> int:
        return self._num_workers

    @override
    def clear_events(self) -> None:
        self._events.clear()
        self._num_workers = 1

    def __repr__(self) -> str:
        return f"<RBLNKVEvents workers={self._num_workers} events={len(self._events)}>"

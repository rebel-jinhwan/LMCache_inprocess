# SPDX-License-Identifier: Apache-2.0
"""Exercise worker -> vLLM executor -> scheduler event consensus."""

# Standard
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import MagicMock

# vLLM pickles worker outputs on their way to the executor; the test does too.
import pickle  # noqa: TID251

# Third Party
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
import pytest
import torch

# First Party
from lmcache.inprocess.rbln.integration.vllm.connector import RBLNLMCacheConnectorV1
from lmcache.inprocess.rbln.integration.vllm.kv_events import RBLNKVEvents
from lmcache.utils import CacheStoreEvent


def _stored(block_hash: int = 7, medium: str = "rds") -> BlockStored:
    return BlockStored(
        block_hashes=[block_hash],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
        lora_id=None,
        medium=medium,
        lora_name=None,
    )


def _batch(events: list[KVCacheEvent]) -> RBLNKVEvents:
    batch = RBLNKVEvents()
    batch.add_events(list(dict.fromkeys(events)))
    # Worker output is serialized on its way to the executor.
    return pickle.loads(pickle.dumps(batch))


def _executor(batches: Sequence[RBLNKVEvents | None]) -> KVConnectorOutput:
    outputs = [
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_connector_output=KVConnectorOutput(kv_cache_events=batch),
        )
        for batch in batches
    ]
    output = KVOutputAggregator(expected_finished_count=len(outputs)).aggregate(outputs)
    assert output is not None
    assert output.kv_connector_output is not None
    return output.kv_connector_output


def _scheduler() -> RBLNLMCacheConnectorV1:
    scheduler = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
    scheduler._kv_cache_events = None
    return scheduler


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_store_requires_every_storing_rank(workers: int) -> None:
    event = _stored()
    scheduler = _scheduler()
    scheduler.update_connector_output(
        _executor([_batch([event] * 3) for _ in range(workers)])
    )
    assert list(scheduler.take_events()) == [event]
    assert list(scheduler.take_events()) == []


@pytest.mark.parametrize("missing_rank", range(4))
def test_empty_rank_prevents_partial_cache_advertisement(missing_rank: int) -> None:
    scheduler = _scheduler()
    batches = [_batch([] if rank == missing_rank else [_stored()]) for rank in range(4)]
    scheduler.update_connector_output(_executor(batches))
    assert list(scheduler.take_events()) == []


@pytest.mark.parametrize("failed_rank", range(4))
def test_removal_on_any_stage_retracts_same_step_store(failed_rank: int) -> None:
    removed = BlockRemoved(block_hashes=[7], medium="rds")
    scheduler = _scheduler()
    batches = [
        _batch([_stored(), removed] if rank == failed_rank else [_stored()])
        for rank in range(4)
    ]
    scheduler.update_connector_output(_executor(batches))
    assert list(scheduler.take_events()) == [removed]


def test_single_stage_eviction_is_published_without_other_votes() -> None:
    removed = BlockRemoved(block_hashes=[7], medium="rds")
    scheduler = _scheduler()
    scheduler.update_connector_output(
        _executor([_batch([]), _batch([removed]), _batch([]), _batch([])])
    )
    assert list(scheduler.take_events()) == [removed]


def test_removal_is_scoped_to_its_medium() -> None:
    cpu = _stored(medium="cpu")
    removed = BlockRemoved(block_hashes=[7], medium="rds")
    scheduler = _scheduler()
    scheduler.update_connector_output(
        _executor([_batch([cpu, removed]), _batch([cpu])])
    )
    assert list(scheduler.take_events()) == [cpu, removed]


def test_consensus_does_not_intersect_different_executor_steps() -> None:
    scheduler = _scheduler()
    first, second = _stored(7), _stored(8)
    scheduler.update_connector_output(_executor([_batch([first]), _batch([first])]))
    scheduler.update_connector_output(_executor([_batch([second]), _batch([second])]))
    assert list(scheduler.take_events()) == [first, second]


def test_later_restore_can_follow_an_eviction() -> None:
    scheduler = _scheduler()
    removed, stored = BlockRemoved(block_hashes=[7], medium="rds"), _stored()
    scheduler.update_connector_output(_executor([_batch([removed]), _batch([])]))
    scheduler.update_connector_output(_executor([_batch([stored]), _batch([stored])]))
    assert list(scheduler.take_events()) == [removed, stored]


@pytest.mark.parametrize("passive", [False, True])
def test_worker_reports_empty_votes_except_passive_mla(passive: bool) -> None:
    engine = MagicMock(kv_events_enabled=True, save_only_first_rank=passive)
    engine.metadata.is_first_rank.return_value = False
    engine.get_kv_events.return_value = []
    backend = MagicMock()
    type(backend).__name__ = "RDSBackend"
    backend.take_evicted_hashes.return_value = []
    engine.storage_manager.storage_backends = {"RDSBackend": backend}
    worker = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
    worker._lmcache_engine = MagicMock(lmcache_engine=engine)
    batch = worker.get_kv_connector_kv_cache_events()
    assert (batch is None) is passive
    if batch is not None:
        assert batch.get_all_events() == []
        assert batch.get_number_of_workers() == 1
    engine.get_kv_events.assert_called_once()
    backend.take_evicted_hashes.assert_called_once()


def test_passive_mla_rank_is_not_a_missing_store_vote() -> None:
    scheduler = _scheduler()
    scheduler.update_connector_output(_executor([_batch([_stored()]), None]))
    assert list(scheduler.take_events()) == [_stored()]


def test_worker_failure_on_nonfirst_stage_reaches_scheduler() -> None:
    engine = MagicMock(kv_events_enabled=True, save_only_first_rank=False)
    event = CacheStoreEvent(
        block_hashes=[7],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
        lora_id=None,
        medium="rds",
        lora_name=None,
    )
    engine.get_kv_events.return_value = [event]
    backend = MagicMock()
    type(backend).__name__ = "RDSBackend"
    backend.take_evicted_hashes.return_value = [7]
    engine.storage_manager.storage_backends = {"RDSBackend": backend}
    worker = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
    worker._lmcache_engine = MagicMock(lmcache_engine=engine)
    batch = worker.get_kv_connector_kv_cache_events()
    scheduler = _scheduler()
    scheduler.update_connector_output(_executor([_batch([_stored()]), batch]))
    assert list(scheduler.take_events()) == [
        BlockRemoved(block_hashes=[7], medium="rds")
    ]


def test_disabled_events_do_not_create_empty_transport_batches() -> None:
    worker = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
    engine = MagicMock(kv_events_enabled=False)
    engine.get_kv_events.return_value = []
    engine.storage_manager.storage_backends = {}
    worker._lmcache_engine = MagicMock(lmcache_engine=engine)
    assert worker.get_kv_connector_kv_cache_events() is None


def test_event_batch_clear_and_repeated_aggregation() -> None:
    events = _batch([_stored()])
    events.aggregate().aggregate()
    assert events.get_all_events() == [_stored()]
    events.clear_events()
    assert events.get_all_events() == []
    assert events.get_number_of_workers() == 1


@pytest.mark.parametrize(
    "case", ["success", "failed", "evicted", "duplicate", "passive"]
)
@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_speculative_save_reaches_consensus_without_another_step(
    monkeypatch: pytest.MonkeyPatch, case: str, pp_size: int
) -> None:
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.worker import kv_connector_model_runner_mixin as mixin
    from vllm_rbln.v1.worker import rbln_model_runner as runner_module

    store = CacheStoreEvent(
        block_hashes=[7],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
        lora_id=None,
        medium="rds",
        lora_name=None,
    )
    pending = [store] if case == "duplicate" else []
    order: list[str] = []

    def drain_stores():
        events = pending[:]
        pending.clear()
        return events

    engine = MagicMock(kv_events_enabled=True, save_only_first_rank=case == "passive")
    engine.metadata.is_first_rank.return_value = False
    engine.get_kv_events.side_effect = drain_stores
    backend = MagicMock()
    type(backend).__name__ = "RDSBackend"
    backend.take_evicted_hashes.side_effect = [[7] if case == "evicted" else [], []]
    engine.storage_manager.storage_backends = {"RDSBackend": backend}
    worker = RBLNLMCacheConnectorV1.__new__(RBLNLMCacheConnectorV1)
    worker._lmcache_engine = MagicMock(lmcache_engine=engine)

    def save():
        order.append("save")
        assert order[-2] == "draft"
        if case not in ("failed", "passive"):
            pending.append(store)

    connector = MagicMock(spec=KVConnectorBase_V1)
    connector.wait_for_save.side_effect = save
    connector.get_kv_connector_kv_cache_events.side_effect = (
        worker.get_kv_connector_kv_cache_events
    )
    connector.get_finished.return_value = (None, None)
    connector.get_block_ids_with_load_errors.return_value = set()
    connector.get_kv_connector_stats.return_value = None
    connector.build_connector_worker_meta.return_value = None
    for module in (mixin, runner_module):
        monkeypatch.setattr(module, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(module, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(mixin, "get_forward_context", lambda: None)
    scheduled = SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids=set(),
        total_num_scheduled_tokens=1,
        has_sync_kv_loads=False,
    )
    with mixin.KVConnectorModelRunnerMixin._get_kv_connector_output(
        scheduled, defer_finalize=True
    ) as output:
        pass
    connector.wait_for_save.assert_not_called()

    runner = SimpleNamespace(
        execute_model_state=runner_module.ExecuteModelState(
            scheduled, None, None, SimpleNamespace(max_seq_len=1), None, None, None
        ),
        kv_connector_output=output,
        # vllm-rbln hands it to ModelRunnerOutput beside kv_connector_output.
        ec_connector_output=None,
        speculative_config=SimpleNamespace(use_eagle=lambda: False),
        # Only a strict KV producer skips the after-bookkeeping drafts.
        is_strict_kv_producer=False,
        use_async_scheduling=False,
        _sample=lambda *a: SamplerOutput(torch.tensor([[7]]), None),
        _bookkeeping_sync=lambda *a: (0, None, [], {}, [], {}, []),
        propose_draft_token_ids=lambda *a: order.append("draft"),
        finalize_kv_connector=mixin.KVConnectorModelRunnerMixin.finalize_kv_connector,
        input_batch=SimpleNamespace(sampling_metadata=None),
        num_spec_tokens=1,
        effective_drafter_max_model_len=100,
    )
    final = runner_module.RBLNModelRunner.sample_tokens(runner, None)
    last_batch = pickle.loads(pickle.dumps(final.kv_connector_output.kv_cache_events))
    scheduler = _scheduler()
    scheduler.update_connector_output(
        _executor([_batch([_stored()]) for _ in range(pp_size - 1)] + [last_batch])
    )
    expected = [] if case in ("failed", "passive") else [_stored()]
    if case == "passive" and pp_size > 1:
        expected = [_stored()]
    if case == "evicted":
        expected = [BlockRemoved(block_hashes=[7], medium="rds")]
    assert list(scheduler.take_events()) == expected
    assert list(scheduler.take_events()) == []
    assert order == ["draft", "save"]
    assert pending == []
    connector.clear_connector_metadata.assert_called_once()

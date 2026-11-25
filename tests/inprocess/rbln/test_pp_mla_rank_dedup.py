# SPDX-License-Identifier: Apache-2.0
"""MLA cross-rank cache dedup must be off under pipeline parallel.

Regression coverage for rebellions-sw/lmcache-rbln#189: with ``pp>1`` and
an MLA model, LMCache's ``save_only_first_rank`` left every rank but
global rank 0 passive -- storing nothing and waiting for a broadcast over
the (single-member, under ``tp=1``) TP group that never arrives -- so
``retrieve`` returned 0 tokens on ranks 1..N-1 while the scheduler's
lookup, answered by rank 0, reported a full prefix hit. vLLM then failed
the request under ``failure_policy=fail``.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.rbln.integration.vllm.vllm_v1_adapter import (
    _align_mla_rank_dedup_with_pp,
)
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.inprocess.v1.token_database import ChunkedTokenDatabase

# Local
from .vllm_deployment import Deployment, build_metadata


def _make_config(**extra_config) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults()
    if extra_config:
        config.extra_config = dict(extra_config)
    return config


def _make_parallel_config(pp_size: int, tp_size: int = 1, rank: int = 0) -> MagicMock:
    parallel_config = MagicMock()
    parallel_config.pipeline_parallel_size = pp_size
    parallel_config.tensor_parallel_size = tp_size
    parallel_config.world_size = pp_size * tp_size
    parallel_config.rank = rank
    return parallel_config


def _make_metadata(config: LMCacheEngineConfig, rank: int, world_size: int):
    """Metadata as ``RBLNServiceFactory`` builds it, for one MLA rank."""
    return LMCacheMetadata(
        model_name="skt/A.X-K2",
        world_size=world_size,
        local_world_size=world_size,
        worker_id=rank,
        local_worker_id=rank,
        kv_dtype=torch.bfloat16,
        kv_shape=(8, 1, config.chunk_size, 1, 576),
        use_mla=True,
        role="worker",
        chunk_size=config.chunk_size,
    )


class TestAlignMlaRankDedupWithPp:
    def test_pp1_is_left_alone(self):
        """Without PP the MLA dedup is sound -- don't touch it."""
        config = _make_config()

        _align_mla_rank_dedup_with_pp(config, _make_parallel_config(1, tp_size=8), True)

        assert config.extra_config is None
        assert config.lookup_server_worker_ids is None

    def test_non_mla_is_left_alone(self):
        """The three knobs are no-ops without MLA; leave the config clean."""
        config = _make_config()

        _align_mla_rank_dedup_with_pp(config, _make_parallel_config(8), False)

        assert config.extra_config is None
        assert config.lookup_server_worker_ids is None

    def test_mla_pp_disables_first_rank_only_storage(self):
        config = _make_config()

        _align_mla_rank_dedup_with_pp(config, _make_parallel_config(4), True)

        assert config.get_extra_config_value("save_only_first_rank") is False
        assert config.get_extra_config_value("remote_enable_mla_worker_id_as0") is False

    def test_mla_pp_fans_lookup_out_to_every_rank(self):
        """A late PP stage must be able to veto a prefix hit.

        Stages store as their own forward retires, not in lockstep like TP
        ranks, so asking only rank 0 (upstream's MLA default) can report a
        hit for a chunk stage 3 has not written yet. The lookup client
        takes ``min()`` over the ranks it asks.
        """
        config = _make_config()

        _align_mla_rank_dedup_with_pp(config, _make_parallel_config(4, tp_size=2), True)

        assert config.lookup_server_worker_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert config.get_lookup_server_worker_ids(True, 8) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_explicit_opt_in_is_overridden(self):
        """An explicit ``save_only_first_rank=True`` is broken under PP too."""
        config = _make_config(save_only_first_rank=True)

        _align_mla_rank_dedup_with_pp(config, _make_parallel_config(4), True)

        assert config.get_extra_config_value("save_only_first_rank") is False

    def test_other_extra_config_entries_survive(self):
        config = _make_config(save_only_first_rank=True, some_other_knob="keep me")

        _align_mla_rank_dedup_with_pp(config, _make_parallel_config(4), True)

        assert config.get_extra_config_value("some_other_knob") == "keep me"


class TestUpstreamReadsTheAlignedConfig:
    """The knobs only matter through the upstream code that reads them."""

    @pytest.mark.parametrize("rank", range(4))
    def test_no_rank_is_passive_under_pp(self, rank):
        """``save_only_first_rank`` is what makes a rank passive.

        ``LMCacheEngine`` and ``ChunkedTokenDatabase`` both derive it from
        ``config.get_extra_config_value("save_only_first_rank", use_mla)
        and use_mla``; a passive rank (the flag on, and not first rank)
        neither stores nor retrieves. Reading it back off the token
        database exercises upstream's own resolution of our override.
        """
        config = _make_config()
        _align_mla_rank_dedup_with_pp(config, _make_parallel_config(4), True)
        metadata = _make_metadata(config, rank=rank, world_size=4)

        token_database = ChunkedTokenDatabase(config, metadata)

        assert token_database.save_only_first_rank is False

    def test_without_the_fix_ranks_1_to_3_are_passive(self):
        """Pin the upstream default this guard exists to undo."""
        config = _make_config()
        metadata = _make_metadata(config, rank=1, world_size=4)

        token_database = ChunkedTokenDatabase(config, metadata)

        assert token_database.save_only_first_rank is True
        assert metadata.is_first_rank() is False

    def test_each_stage_keys_its_own_chunk(self):
        """Every stage caches different layers, so keys must not collide."""
        config = _make_config()
        _align_mla_rank_dedup_with_pp(config, _make_parallel_config(4), True)
        tokens = torch.arange(config.chunk_size * 2)

        keys_per_rank = []
        for rank in range(4):
            metadata = _make_metadata(config, rank=rank, world_size=4)
            token_database = ChunkedTokenDatabase(config, metadata)
            keys_per_rank.append(
                [key.to_string() for _, _, key in token_database.process_tokens(tokens)]
            )

        flat = [key for keys in keys_per_rank for key in keys]
        assert len(flat) == len(set(flat)), "PP stages share a cache key"


class TestFactoryAppliesTheGuard:
    """The guard has to run on both roles -- the scheduler's lookup client
    builds its keys from the same config as the workers' engines."""

    @staticmethod
    def _axk2(use_mla: bool) -> Deployment:
        return Deployment(
            model="skt/A.X-K2",
            num_layers=32,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            use_mla=use_mla,
            block_size=2048,
            pp_size=4,
        )

    @pytest.mark.parametrize("role", ["scheduler", "worker"])
    def test_metadata_creation_aligns_the_config(self, role):
        config = _make_config()
        build_metadata(self._axk2(use_mla=True), role, config=config)

        assert config.get_extra_config_value("save_only_first_rank") is False
        assert config.lookup_server_worker_ids == [0, 1, 2, 3]

    def test_non_mla_pp_model_is_untouched(self):
        config = _make_config()
        build_metadata(self._axk2(use_mla=False), config=config)

        assert config.extra_config is None
        assert config.lookup_server_worker_ids is None

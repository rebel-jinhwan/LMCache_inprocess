# SPDX-License-Identifier: Apache-2.0
"""MLA's cross-rank cache dedup is off under pipeline parallelism.

Each PP stage caches only its own layers, so no rank may store, look up or
broadcast on behalf of the others.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.inprocess.integration.vllm.utils import align_mla_rank_dedup_with_pp
from lmcache.inprocess.v1.config import LMCacheEngineConfig


def _parallel(pp: int, tp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(pipeline_parallel_size=pp, world_size=pp * tp)


def _dedup(config: LMCacheEngineConfig, world_size: int) -> tuple:
    return (
        config.get_extra_config_value("save_only_first_rank", True),
        config.get_extra_config_value("remote_enable_mla_worker_id_as0", True),
        config.get_lookup_server_worker_ids(True, world_size),
    )


@pytest.mark.parametrize("pp, tp", [(2, 1), (4, 1), (2, 2)])
def test_pp_turns_mla_dedup_off(pp: int, tp: int) -> None:
    config = LMCacheEngineConfig.from_defaults()
    align_mla_rank_dedup_with_pp(config, _parallel(pp, tp), use_mla=True)
    assert _dedup(config, pp * tp) == (False, False, list(range(pp * tp)))


@pytest.mark.parametrize("pp, use_mla", [(1, True), (2, False)])
def test_other_deployments_keep_the_defaults(pp: int, use_mla: bool) -> None:
    config = LMCacheEngineConfig.from_defaults()
    align_mla_rank_dedup_with_pp(config, _parallel(pp, tp=2), use_mla=use_mla)
    assert _dedup(config, pp * 2) == (True, True, [0])

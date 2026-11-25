# SPDX-License-Identifier: Apache-2.0
"""An EAGLE-family draft is counted on the last pipeline stage only.

vLLM builds the whole drafter on the last PP rank, so no other stage
registers any of its KV caches.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.inprocess.integration.vllm.utils import calculate_stage_draft_layers


class _DraftModelConfig:
    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers

    def get_total_num_hidden_layers(self) -> int:
        return self.num_layers

    def get_num_layers(self, parallel_config: SimpleNamespace) -> int:
        # A split over the *target*'s stages, which is what
        # calculate_draft_layers reads. vLLM's get_pp_indices places the
        # remainder differently, but any split puts layers off the last stage.
        pp = parallel_config.pipeline_parallel_size
        pp_rank = (parallel_config.rank // parallel_config.tensor_parallel_size) % pp
        base, extra = divmod(self.num_layers, pp)
        return base + (1 if pp_rank < extra else 0)


class _BrokenDraftModelConfig:
    def get_total_num_hidden_layers(self) -> int:
        raise RuntimeError("no hf_config")

    def get_num_layers(self, parallel_config: SimpleNamespace) -> int:
        raise RuntimeError("no hf_config")


def _vllm_config(
    method: str | None,
    num_draft_layers: int = 1,
    *,
    pp: int = 1,
    tp: int = 1,
    rank: int = 0,
    draft_model_config=None,
) -> SimpleNamespace:
    speculative_config = None
    if method is not None:
        speculative_config = SimpleNamespace(
            method=method,
            use_eagle=lambda: method in ("eagle", "eagle3", "mtp", "dflash"),
            draft_model_config=draft_model_config
            or _DraftModelConfig(num_draft_layers),
        )
    return SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            tensor_parallel_size=tp,
            world_size=pp * tp,
            rank=rank,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )


@pytest.mark.parametrize("method", ["eagle", "eagle3", "mtp"])
def test_without_pp_counts_the_whole_draft(method: str) -> None:
    assert calculate_stage_draft_layers(_vllm_config(method, 3)) == 3


@pytest.mark.parametrize("method", ["eagle", "mtp"])
def test_one_layer_draft_over_four_stages_sits_on_the_last(method: str) -> None:
    counts = [
        calculate_stage_draft_layers(_vllm_config(method, 1, pp=4, rank=r))
        for r in range(4)
    ]
    assert counts == [0, 0, 0, 1]


def test_mtp_at_pp8_is_all_on_the_last_stage() -> None:
    # DeepSeek-V3.2's single MTP layer at pp=8 must not be split.
    counts = [
        calculate_stage_draft_layers(_vllm_config("mtp", 1, pp=8, rank=r))
        for r in range(8)
    ]
    assert counts == [0] * 7 + [1]


def test_multi_layer_draft_is_not_split_across_stages() -> None:
    counts = [
        calculate_stage_draft_layers(_vllm_config("eagle", 3, pp=2, rank=r))
        for r in range(2)
    ]
    assert counts == [0, 3]


def test_stage_follows_vllm_dp_pp_tp_rank_layout() -> None:
    # pp=2, tp=2: ranks 0-1 are stage 0, ranks 2-3 are stage 1; rank 4 is
    # the next DP replica's stage 0.
    counts = [
        calculate_stage_draft_layers(_vllm_config("eagle", 2, pp=2, tp=2, rank=r))
        for r in range(6)
    ]
    assert counts == [0, 0, 2, 2, 0, 0]


def test_unreadable_draft_falls_back_to_one_layer_on_the_last_stage() -> None:
    broken = _BrokenDraftModelConfig()
    counts = [
        calculate_stage_draft_layers(
            _vllm_config("eagle", pp=2, rank=r, draft_model_config=broken)
        )
        for r in range(2)
    ]
    assert counts == [0, 1]


@pytest.mark.parametrize("method", [None, "ngram"])
def test_non_eagle_drafts_are_unchanged(method: str | None) -> None:
    assert calculate_stage_draft_layers(_vllm_config(method, pp=4, rank=3)) == 0

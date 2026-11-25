# SPDX-License-Identifier: Apache-2.0
"""``CacheLayout`` reads the chunk layout trunk derived from config keys and
cache names, from the per-layer vLLM specs alone.

Every expected layout is what trunk's ``RBLNServiceFactory`` built for the
same model on a worker, so a divergence from the shipped chunk fails here.
"""

# Standard
from collections.abc import Callable
from dataclasses import fields, replace
import random

# Third Party
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
)
from vllm_rbln.v1.kv_cache import RBLNMambaSpec, RBLNSlidingWindowSpec
import pytest
import torch

# First Party
from lmcache.inprocess.rbln.integration.vllm.cache_layout import CacheLayout, KVRegion

# Local
from . import kv_spec_fixtures as fx

FIXTURES: dict[str, Callable[[], fx.KVSpecFixture]] = {
    "DSA fp8": lambda: fx.deepseek_v32("fp8"),
    "DSA bf16": lambda: fx.deepseek_v32("bf16"),
    "DSA fp8 + MTP draft with its indexer": lambda: fx.deepseek_v32("fp8", draft="mtp"),
    "MSA bf16": lambda: fx.minimax_m3("bf16"),
    "MSA fp8": lambda: fx.minimax_m3("fp8"),
    "MSA PP stage of layers 30-59": lambda: fx.minimax_m3("bf16", range(30, 60)),
    "MSA PP stage 0, dense layers 0-2": lambda: fx.minimax_m3("bf16", range(3)),
    "Qwen3": fx.qwen3,
    "Qwen3 + Eagle3 draft of 4 heads": fx.qwen3_eagle3,
    "Qwen3 + Eagle3 draft of 8 heads": lambda: fx.qwen3_eagle3(8),
    "gpt-oss": fx.gpt_oss,
    "Gemma4": fx.gemma4,
}


# What trunk's ``RBLNServiceFactory`` built on a worker for each fixture, from
# config keys and cache names, before workers read ``CacheLayout`` (c81f8d7).
# Trunk refused Gemma4, so it has no entry.
TRUNK_GROUPS: dict[
    str, tuple[list[tuple[int, int]], list[tuple[int, int, torch.dtype]]]
] = {
    "DSA fp8": (
        [(61, 1)],
        [(61, 128, torch.float8_e4m3fn), (61, 1, torch.bfloat16)],
    ),
    "DSA bf16": ([(61, 1)], [(61, 128, torch.bfloat16)]),
    "DSA fp8 + MTP draft with its indexer": (
        [(62, 1)],
        [(62, 128, torch.float8_e4m3fn), (62, 1, torch.bfloat16)],
    ),
    "MSA bf16": ([(60, 4)], [(57, 128, torch.bfloat16)]),
    "MSA fp8": ([(60, 4)], [(57, 128, torch.uint8)]),
    "MSA PP stage of layers 30-59": ([(30, 4)], [(30, 128, torch.bfloat16)]),
    "MSA PP stage 0, dense layers 0-2": ([(3, 4)], []),
    "Qwen3": ([(36, 8)], []),
    "Qwen3 + Eagle3 draft of 4 heads": ([(36, 8), (1, 4)], []),
    "Qwen3 + Eagle3 draft of 8 heads": ([(37, 8)], []),
    "gpt-oss": ([(24, 8)], []),
}

KV_DTYPES: dict[str, torch.dtype] = {
    "DSA fp8": torch.uint8,
    "DSA fp8 + MTP draft with its indexer": torch.uint8,
    "MSA fp8": torch.uint8,
}


@pytest.mark.parametrize("name", TRUNK_GROUPS)
def test_layout_equals_what_trunk_built_on_a_worker(name: str) -> None:
    layout = CacheLayout.from_kv_cache_config(FIXTURES[name]().kv_cache_config())
    assert (layout.kv_head_groups, layout.aux_groups) == TRUNK_GROUPS[name]
    assert {r.dtype for r in layout.kv_regions} == {KV_DTYPES.get(name, torch.bfloat16)}


@pytest.mark.parametrize("name", FIXTURES)
def test_regions_name_every_cache_once_in_layer_order(name: str) -> None:
    model = FIXTURES[name]()
    layout = CacheLayout.from_kv_cache_config(model.kv_cache_config())
    regions = [r.layer_names for r in layout.kv_regions] + [
        r.layer_names for r in layout.aux_regions
    ]
    assert sorted(n for names in regions for n in names) == sorted(model.specs)
    registered = list(model.specs)
    for names in regions:
        assert list(names) == sorted(names, key=registered.index)


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_spec_order_does_not_change_the_layout(name: str, seed: int) -> None:
    model = FIXTURES[name]()
    items = list(model.specs.items())
    random.Random(seed).shuffle(items)
    shuffled = CacheLayout.from_kv_cache_config(model.kv_cache_config(dict(items)))
    assert shuffled == CacheLayout.from_kv_cache_config(model.kv_cache_config())


def test_layers_with_one_head_count_form_one_region_in_first_seen_order() -> None:
    # RBLNConnector.register_kv_caches groups an 8, 4, 8 head pattern this way.
    specs = dict(fx._qwen3_layer(i, h) for i, h in enumerate([8, 4, 8]))
    layout = CacheLayout.from_kv_cache_config(_one_group(specs))
    assert layout.kv_regions == (
        KVRegion(
            8,
            torch.bfloat16,
            ("model.layers.0.self_attn.attn", "model.layers.2.self_attn.attn"),
            128,
        ),
        KVRegion(4, torch.bfloat16, ("model.layers.1.self_attn.attn",), 128),
    )
    assert layout.kv_head_groups == [(2, 8), (1, 4)]


def _one_group(specs: dict[str, KVCacheSpec]) -> KVCacheConfig:
    """Groups that ``get_kv_cache_groups`` would refuse or regroup."""
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([n], s) for n, s in specs.items()],
    )


def test_mla_families_with_equal_pages_are_refused() -> None:
    # Without a K/V cache the widest MLA family is the latent; a tie has none.
    specs = dict(fx.deepseek_v32("bf16").specs)
    key = "model.layers.5.self_attn.indexer.k_cache"
    specs[key] = replace(specs[key], head_size=576, dtype=torch.float16)
    with pytest.raises(
        ValueError, match=r"model\.layers\.\d+\.self_attn.* equal pages"
    ):
        CacheLayout.from_kv_cache_config(fx.worker_kv_cache_config(specs))


def test_kv_region_with_another_dtype_is_refused() -> None:
    specs = dict(fx.qwen3().specs)
    key = "model.layers.3.self_attn.attn"
    specs[key] = replace(specs[key], dtype=torch.float16)
    with pytest.raises(ValueError, match=r"KV region of 8 heads mixes .*layers\.3\."):
        CacheLayout.from_kv_cache_config(_one_group(specs))


@pytest.mark.parametrize(
    "edit", [{"head_size": 64}, {"dtype": torch.float16}], ids=["head", "dtype"]
)
def test_an_aux_cache_of_another_spec_forms_its_own_region(
    edit: dict[str, object],
) -> None:
    specs = dict(fx.minimax_m3("bf16").specs)
    key = "model.layers.7.self_attn.attn.indexer"
    specs[key] = replace(specs[key], **edit)
    layout = CacheLayout.from_kv_cache_config(_one_group(specs))
    assert [r.layer_names for r in layout.aux_regions][1:] == [(key,)]
    assert key not in layout.aux_regions[0].layer_names


def test_rbln_sliding_window_is_refused() -> None:
    specs = dict(fx.gpt_oss().specs)
    key = "model.layers.0.attn.attn"
    specs[key] = RBLNSlidingWindowSpec(
        block_size=fx.BLOCK_SIZE,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    with pytest.raises(
        ValueError, match=r"layers\.0\.attn\.attn .* is a RBLNSlidingWindowSpec"
    ):
        CacheLayout.from_kv_cache_config(_one_group(specs))


def test_names_without_one_layer_index_keep_vllm_group_order() -> None:
    # GLM-5.3 names its caches attn.<slot>, which extract_layer_index asserts on.
    specs = dict(fx.qwen3().specs)
    specs["model.layers.2.self_attn.attn.1"] = specs.pop(
        "model.layers.2.self_attn.attn"
    )
    layout = CacheLayout.from_kv_cache_config(_one_group(specs))
    assert layout.kv_regions[0].layer_names == tuple(specs)


def _glm_layout(fixture: fx.GlmFixture) -> CacheLayout:
    return CacheLayout.from_kv_cache_config(fixture.kv_cache_config, fixture.aux_specs)


def _short(name: str) -> str:
    return name.removeprefix("model.layers.").replace(".self_attn.attn", "")


def test_glm_chunk_is_latent_then_index_key_scale_kda_state_and_conv() -> None:
    layout = _glm_layout(fx.glm53(4))
    assert layout.kv_regions == (
        KVRegion(1, torch.uint8, ("model.layers.3.self_attn.attn.0",), 512),
    )
    assert layout.aux_regions == ()
    assert [
        ([_short(c.name) for c in r.caches], r.block_shape, r.dtype, r.checkpoint)
        for r in layout.block_regions
    ] == [
        (["3.1"], (1024, 128), torch.float8_e4m3fn, False),
        (["3.2"], (1024,), torch.float16, False),
        (["0.0", "1.0", "2.0"], (4096, 512), torch.uint8, True),
        (["0.1", "1.1", "2.1"], (4, 3, 128, 64), torch.bfloat16, True),
    ]
    assert layout.unstored == ("model.layers.3.self_attn.attn.3",)


@pytest.mark.parametrize("layers, drafts", [(4, 0), (8, 0), (16, 0), (4, 1), (4, 2)])
def test_glm_layout_places_every_registered_cache_once(
    layers: int, drafts: int
) -> None:
    fixture = fx.glm53(layers, drafts)
    layout = _glm_layout(fixture)
    latents = [n for r in layout.kv_regions for n in r.layer_names]
    stored = latents + [c.name for r in layout.block_regions for c in r.caches]
    assert len(stored) == len(set(stored))
    assert layout.cache_names == set(fixture.registered)
    for region in layout.block_regions:
        for cache in region.caches:
            shape, dtype = fixture.registered[cache.name]
            assert (torch.Size(shape).numel(), dtype) == (
                torch.Size(region.block_shape).numel(),
                region.dtype,
            ), cache.name
    # Only the tail rings and the MTP drafts are scratch.
    unstored = {n: fixture.aux_specs[n].prefix_cacheable for n in layout.unstored}
    assert unstored and not any(unstored.values())
    dsa = layers // 4
    assert len(latents) == dsa
    assert len(layout.unstored) == dsa + (layers - dsa if drafts else 0)


def test_glm_mtp_drafts_and_tail_rings_are_not_stored() -> None:
    fixture = fx.glm53(8, drafts=1)
    layout = _glm_layout(fixture)
    assert set(layout.unstored) == {
        n for n, a in fixture.aux_specs.items() if not a.prefix_cacheable
    }
    assert [_short(n) for n in layout.unstored] == [
        "0.2",
        "1.2",
        "2.2",
        "3.3",
        "4.2",
        "5.2",
        "6.2",
        "7.3",
    ]
    assert len(layout.block_regions) == 4


def test_glm_without_prefix_caching_keeps_the_chunk_layout() -> None:
    # One running state a request ("none") moves as the block a checkpoint
    # ("align") does, so both engines read each other's chunks.
    align = _glm_layout(fx.glm53(8))
    none = _glm_layout(fx.glm53(8, mamba_cache_mode="none"))
    assert none.kv_head_groups == align.kv_head_groups
    assert none.aux_groups == align.aux_groups
    assert none.block_groups == align.block_groups


def test_kda_layers_alone_without_prefix_caching_are_refused() -> None:
    # Only DSA layers keep one row per token; a KDA-only model reads a chunk's
    # tokens from a block, and a running state's block spans max_model_len.
    fixture = fx.glm53(3, mamba_cache_mode="none")
    with pytest.raises(ValueError, match="with prefix caching off no block"):
        CacheLayout.from_kv_cache_config(fixture.kv_cache_config, fixture.aux_specs)


def test_glm_aux_caches_follow_their_owners_group_and_order() -> None:
    layout = _glm_layout(fx.glm53(8))
    keys, scales, states, convs = layout.block_regions
    assert [c.group for c in convs.caches] == [c.group for c in states.caches]
    assert [_short(c.name)[:-2] for c in convs.caches] == [
        _short(c.name)[:-2] for c in states.caches
    ]
    assert [c.group for c in scales.caches] == [c.group for c in keys.caches] == [0, 0]


@pytest.mark.parametrize("seed", range(3))
def test_glm_aux_spec_order_does_not_change_the_layout(seed: int) -> None:
    fixture = fx.glm53(16)
    items = list(fixture.aux_specs.items())
    random.Random(seed).shuffle(items)
    assert CacheLayout.from_kv_cache_config(
        fixture.kv_cache_config, dict(items)
    ) == _glm_layout(fixture)


def _glm_groups(fixture: fx.GlmFixture, **edit: KVCacheSpec) -> KVCacheConfig:
    groups = []
    for group in fixture.kv_cache_config.kv_cache_groups:
        name = group.layer_names[0]
        spec = edit.get(name, group.kv_cache_spec)
        groups.append(KVCacheGroupSpec(list(group.layer_names), spec))
    return KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups)


def test_glm_mamba_group_not_in_align_mode_is_refused() -> None:
    fixture = fx.glm53(4)
    state = fixture.kv_cache_config.kv_cache_groups[1]
    spec = replace(state.kv_cache_spec, mamba_cache_mode="all")
    with pytest.raises(ValueError, match=r"layers\.0\.self_attn\.attn\.0 .*'all'"):
        CacheLayout.from_kv_cache_config(
            _glm_groups(fixture, **{state.layer_names[0]: spec}), fixture.aux_specs
        )


def test_upstream_mamba_spec_is_refused() -> None:
    # Only vllm-rbln's RBLNMambaSpec (GLM-5.3's KDA state) is verified.
    fixture = fx.glm53(4)
    state = fixture.kv_cache_config.kv_cache_groups[1]
    spec = MambaSpec(
        **{f.name: getattr(state.kv_cache_spec, f.name) for f in fields(MambaSpec)}
    )
    with pytest.raises(ValueError, match=r"attn\.0 .* is a MambaSpec"):
        CacheLayout.from_kv_cache_config(
            _glm_groups(fixture, **{state.layer_names[0]: spec}), fixture.aux_specs
        )


def test_glm_aux_without_a_grouped_owner_is_refused() -> None:
    fixture = fx.glm53(4)
    aux = dict(fixture.aux_specs)
    aux["model.layers.3.self_attn.attn.2"] = replace(
        aux["model.layers.3.self_attn.attn.2"], owner="model.layers.9.self_attn.attn.1"
    )
    with pytest.raises(ValueError, match=r"attn\.2: owner .*layers\.9.* no KV cache"):
        CacheLayout.from_kv_cache_config(fixture.kv_cache_config, aux)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_glm_cut_to_kda_layers_is_block_regions_only(layers: int) -> None:
    # VLLM_RBLN_NUM_HIDDEN_LAYERS below 4 leaves GLM-5.3 without a DSA layer.
    layout = _glm_layout(fx.glm53(layers))
    assert (layout.kv_regions, layout.aux_regions) == ((), ())
    states, convs = layout.block_regions
    assert states.block_shape == (4096, 512) and states.checkpoint
    assert len(states.caches) == len(convs.caches) == layers


def test_block_regions_without_a_row_per_token_first_are_refused() -> None:
    state = RBLNMambaSpec(
        block_size=fx.GLM_BLOCK_SIZE,
        shapes=((4, 64),),
        dtypes=(torch.uint8,),
        mamba_cache_mode="align",
    )
    with pytest.raises(ValueError, match="no row per token of a"):
        CacheLayout.from_kv_cache_config(
            _one_group({"model.layers.0.self_attn.attn.0": state})
        )

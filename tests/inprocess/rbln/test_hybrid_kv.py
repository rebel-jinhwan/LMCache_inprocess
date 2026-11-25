# SPDX-License-Identifier: Apache-2.0
"""HMA must preserve group addresses through metadata and physical transfers."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    supports_hma,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
import pytest
import torch

# First Party
from lmcache.inprocess.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    SaveSpec,
)
from lmcache.inprocess.rbln.integration.vllm.cache_layout import CacheLayout, KVRegion
from lmcache.inprocess.rbln.integration.vllm.connector import (
    RBLNLMCacheConnectorMetadata,
    RBLNLMCacheConnectorV1,
    RBLNLMCacheConnectorV1Impl,
)
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector
from lmcache.inprocess.rbln.v1.rbln_connector.kv_ops import (
    kernel_block_slabs,
    slot_runs,
)
from lmcache.inprocess.v1.config import LMCacheEngineConfig

# Local
from . import kv_spec_fixtures as fx

CONNECTOR = "lmcache.inprocess.rbln.integration.vllm.connector"


def _token_major(chunk, kernel_sizes):
    """``chunk`` with each layer plane put back in token order.

    Grouped chunks keep each plane kernel-block-major: chunk token ``t`` of a
    kernel-``k`` layer sits at slab ``t // k``, column ``t % k``. The
    physical-coordinate oracles below index tokens, so they read through this.
    """
    out = chunk.clone()
    for layer, kernel in enumerate(kernel_sizes):
        for half in range(chunk.shape[0]):
            slabs = kernel_block_slabs(chunk[half, layer], kernel)
            for t in range(chunk.shape[3]):
                out[half, layer, :, t] = slabs[t // kernel][:, t % kernel]
    return out


def test_hma_completion_dispatch():
    assert supports_hma(RBLNLMCacheConnectorV1)
    connector = object.__new__(RBLNLMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()
    connector._lmcache_engine.request_finished.return_value = (False, {"done": 1})
    connector._lmcache_engine._request_blocks = {"request": object()}
    connector._lookup_req_ids = {"request"}
    connector._async_lookup_req_ids = {"request"}
    request = SimpleNamespace(request_id="request")
    assert connector.request_finished_all_groups(request, ([3], [7])) == (
        False,
        {"done": 1},
    )
    connector._lmcache_engine.request_finished.assert_called_once_with(request, [])
    assert not connector._lmcache_engine._request_blocks
    assert not connector._lookup_req_ids
    assert not connector._async_lookup_req_ids


@pytest.mark.parametrize("local_tokens", [0, 8])
def test_recompute_bypasses_lookup_but_keeps_resume_load_spec(local_tokens):
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = True
    impl.load_specs = {}
    impl._manager = MagicMock()
    impl.kv_role = "kv_both"
    impl._block_size = impl._lmcache_chunk_size = 8
    impl._max_tokens_per_load = 0
    impl.config = LMCacheEngineConfig(chunk_size=4)
    impl.lookup_client.lookup_cache.return_value = 16
    request = SimpleNamespace(
        request_id="req", num_tokens=24, skip_reading_prefix_cache=True
    )

    assert impl.get_num_new_matched_tokens(request, local_tokens) == 0

    impl.lookup_client.lookup_cache.assert_not_called()
    assert impl.load_specs["req"] == LoadSpec(local_tokens, 0, False)


@pytest.mark.parametrize("role", ["kv_both", "kv_producer"])
def test_failed_load_blocks_saves_until_scheduler_reset(role, monkeypatch):
    # Patch where the inherited upstream impl looks it up, wherever that lives.
    monkeypatch.setattr(
        sys.modules[LMCacheConnectorV1Impl.__module__],
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = True
    impl._failed_load_req_ids = set()
    impl._block_size = impl._lmcache_chunk_size = 8
    spec = FullAttentionSpec(
        block_size=8, num_kv_heads=1, head_size=2, dtype=torch.float32
    )
    impl._kv_cache_groups = [KVCacheGroupSpec([f"l{g}"], spec) for g in range(2)]
    impl.kv_caches = {"layer": torch.zeros(1)}
    impl._manager = MagicMock()
    impl.lmcache_engine.retrieve.return_value = torch.zeros(8, dtype=torch.bool)
    impl._stats_monitor = MagicMock()
    impl.device = torch.device("cpu")
    impl.use_layerwise = impl.enable_blending = False
    impl.async_loading = False
    impl._invalid_block_ids = set()
    impl.kv_role = role
    impl._is_last_pp_rank = True
    impl.config = LMCacheEngineConfig(chunk_size=4)
    slots = torch.stack((torch.arange(8, 24), torch.arange(24, 40)), dim=1)
    metadata = RBLNLMCacheConnectorMetadata(
        [
            ReqMeta(
                "failed",
                list(range(16)),
                slots,
                True,
                SaveSpec(8, True),
                LoadSpec(0, 8, True),
            ),
            ReqMeta("shared", list(range(16)), slots, True, SaveSpec(8, True)),
        ]
    )
    impl._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)

    impl.start_load_kv(SimpleNamespace(attn_metadata={}))
    impl.wait_for_save()
    impl.lmcache_engine.store.assert_not_called()
    # lmcache >= 0.5.6 also unpins right after a blocking retrieve; the cleanup
    # is idempotent, so check which requests were released rather than how often.
    unpinned = {c.args[0] for c in impl.lmcache_engine.lookup_unpin.call_args_list}
    assert unpinned == {"failed", "shared"}

    # A queued step may still consume the failed prefix before scheduler reset.
    impl._invalid_block_ids.clear()
    metadata.requests[0].load_spec = None
    impl.start_load_kv(SimpleNamespace(attn_metadata={}))
    impl.wait_for_save()
    impl.lmcache_engine.store.assert_not_called()

    metadata.reset_req_ids = {"failed", "shared"}
    impl.start_load_kv(SimpleNamespace(attn_metadata={}))
    impl.wait_for_save()
    assert impl.lmcache_engine.store.call_count == 2


@pytest.mark.parametrize("blocks_first", [False, True])
@pytest.mark.parametrize("chunk_size", [4, 16])
def test_hybrid_transfer_roundtrip(blocks_first, chunk_size):
    # Manager blocks contain 16 tokens; SWA splits each into four kernel blocks.
    connector = RBLNConnector(2, 2, 4, 16, torch.float32)
    caches = [
        torch.full((2, 8, 2, 1, 16, 4), -1.0),
        torch.full((2, 32, 2, 1, 4, 4), -1.0),
    ]
    if blocks_first:
        caches = [cache.transpose(0, 1).contiguous() for cache in caches]
    connector.register_kv_caches(caches, layer_group_indices=[1, 0])
    slots = torch.stack((torch.arange(48, 80), torch.arange(16, 48)), dim=1)
    expected = torch.arange(2 * 2 * 2 * 32 * 4, dtype=torch.float32).reshape(
        2, 2, 2, 32, 4
    )
    logical = torch.empty_like(expected)
    for start in range(0, 32, chunk_size):
        end = start + chunk_size
        source = SimpleNamespace(tensor=expected[:, :, :, start:end].contiguous())
        connector.to_gpu(source, start, end, slot_mapping=slots)
        logical[:, :, :, start:end] = _token_major(source.tensor, [16, 4])

    # Independent physical-coordinate oracle catches a shared wrong mapping.
    for layer, (group, kernel_size) in enumerate([(1, 16), (0, 4)]):
        cache = connector._kv_caches[layer]
        for token in range(32):
            block, offset = divmod(int(slots[token, group]), kernel_size)
            assert torch.equal(
                cache[:, block, :, 0, offset], logical[:, layer, :, token]
            )
        assert torch.all(cache[:, 0] == -1)

    for start in range(0, 32, chunk_size):
        end = start + chunk_size
        dest = SimpleNamespace(tensor=torch.empty(2, 2, 2, chunk_size, 4))
        connector.from_gpu(dest, start, end, slot_mapping=slots)
        assert torch.equal(dest.tensor, expected[:, :, :, start:end])


def _reference_runs(slots, kernel_size):
    # The per-layer walk the helper replaced, kept as the oracle.
    runs = []
    token = 0
    while token < len(slots):
        slot = slots[token]
        if slot < 0:
            token += 1
            continue
        block, offset = divmod(slot, kernel_size)
        stop = token + 1
        while (
            stop < len(slots)
            and offset + stop - token < kernel_size
            and slots[stop] == slot + stop - token
            and stop % kernel_size
        ):
            stop += 1
        runs.append((block, offset, token, stop))
        token = stop
    return runs


@pytest.mark.parametrize("kernel_size", [1, 3, 4, 16])
def test_slot_runs_match_the_per_token_walk(kernel_size):
    generator = torch.Generator().manual_seed(192)
    for _ in range(64):
        length = int(torch.randint(0, 40, (1,), generator=generator))
        slots = []
        cursor = int(torch.randint(0, 64, (1,), generator=generator))
        for _ in range(length):
            roll = int(torch.randint(0, 8, (1,), generator=generator))
            if roll == 0:
                slots.append(-1)
            elif roll == 1:
                cursor = int(torch.randint(0, 64, (1,), generator=generator))
                slots.append(cursor)
            else:
                slots.append(cursor)
            cursor += 1
        assert slot_runs(slots, kernel_size) == _reference_runs(slots, kernel_size)
    assert slot_runs([], 4) == []
    assert slot_runs([-1, -1], 4) == []
    assert slot_runs([6, 7, 8, 9], 4) == [(1, 2, 0, 2), (2, 0, 2, 4)]


def test_grouped_copy_walks_the_chunk_once_per_group_and_kernel_size(monkeypatch):
    connector = RBLNConnector(
        3, 2, 4, 16, torch.float32, kv_head_groups=[(2, 2), (1, 4)]
    )
    connector.register_kv_caches(
        [
            torch.zeros(2, 8, 2, 1, 16, 4),
            torch.zeros(2, 8, 4, 1, 16, 4),
            torch.zeros(2, 32, 2, 1, 4, 4),
        ],
        layer_group_indices=[1, 1, 0],
    )
    slots = torch.stack((torch.arange(48, 64), torch.arange(16, 32)), dim=1)
    source = [torch.ones(shape) for shape in connector.get_shapes(16)]
    real_foreach = torch._foreach_copy_
    calls: list[int] = []

    def copy_contiguous_heads(dsts, srcs):
        # Strided multi-head runs can pull an entire compiled cache to host.
        assert all(t.is_contiguous() for t in (*dsts, *srcs))
        calls.append(len(dsts))
        return real_foreach(dsts, srcs)

    monkeypatch.setattr(torch, "_foreach_copy_", copy_contiguous_heads)
    with patch(
        "lmcache.inprocess.rbln.v1.rbln_connector.connector.slot_runs", wraps=slot_runs
    ) as runs:
        connector.to_gpu(
            SimpleNamespace(get_tensor=lambda i: source[i]), 0, 16, slot_mapping=slots
        )
    # Two layers share (group 1, kernel 16); the third is (group 0, kernel 4).
    assert sorted(call.args[1] for call in runs.call_args_list) == [4, 16]
    # Group 1 (layers 0 and 1) took slots 16..31: manager block 1. Group 0
    # (layer 2) took slots 48..63: kernel blocks 12..15.
    for cache in connector._kv_caches[:2]:
        assert torch.all(cache[:, 1] == 1)
        assert torch.all(cache[:, 0] == 0) and torch.all(cache[:, 2:] == 0)
    assert torch.all(connector._kv_caches[2][:, 12:16] == 1)
    assert torch.all(connector._kv_caches[2][:, :12] == 0)
    # Every layer goes out in one batch, as the store below does: a call per
    # layer cost ~11 ms per 256-token GPT-OSS chunk in fixed submit cost. K and
    # V of one block for each kernel-16 layer, of four for the kernel-4 one.
    assert calls == [2 + 2 + 2 * 4]

    calls.clear()
    connector._device = torch.device("meta")  # a host destination
    stored = [torch.zeros(shape) for shape in connector.get_shapes(16)]
    connector.from_gpu(
        SimpleNamespace(get_tensor=lambda i: stored[i]), 0, 16, slot_mapping=slots
    )
    assert calls == [2 + 2 + 2 * 4]
    assert all(torch.all(chunk == 1) for chunk in stored)


@pytest.mark.parametrize("device_resident", [False, True])
def test_grouped_download_gathers_into_contiguous_planes(monkeypatch, device_resident):
    # Manager blocks contain 16 tokens; SWA splits each into four kernel blocks.
    connector = RBLNConnector(2, 2, 4, 16, torch.float32)
    caches = [
        torch.full((2, 8, 2, 1, 16, 4), -1.0),
        torch.full((2, 32, 2, 1, 4, 4), -1.0),
    ]
    connector.register_kv_caches(caches, layer_group_indices=[1, 0])
    slots = torch.stack((torch.arange(48, 64), torch.arange(16, 32)), dim=1)
    expected = torch.arange(2 * 2 * 2 * 16 * 4, dtype=torch.float32).reshape(
        2, 2, 2, 16, 4
    )
    connector.to_gpu(
        SimpleNamespace(tensor=expected.clone()), 0, 16, slot_mapping=slots
    )

    # CPU caches stand in for the device. A different registered device
    # represents a host destination. A resident destination is gathered into
    # directly, with no intermediate stack or whole-K/V device copy.
    monkeypatch.delenv("LMCACHE_RBLN_SPLIT_KV_COPY", raising=False)
    if not device_resident:
        connector._device = torch.device("meta")
    dest = SimpleNamespace(tensor=torch.zeros(2, 2, 2, 16, 4))
    real_copy = torch.Tensor.copy_
    real_cat = torch.cat
    real_foreach = torch._foreach_copy_
    copies_into_dest = []
    gather_targets = []

    def cat_into_plane(tensors, dim, *, out):
        assert out.is_contiguous()
        gather_targets.append(out)
        return real_cat(tensors, dim=dim, out=out)

    def copy_contiguous_heads(dsts, srcs):
        assert all(t.is_contiguous() for t in (*dsts, *srcs))
        gather_targets.extend(dsts)
        return real_foreach(dsts, srcs)

    def spy_copy(self, src, *args, **kwargs):
        target = dest.tensor.untyped_storage().data_ptr()
        if self.untyped_storage().data_ptr() == target:
            copies_into_dest.append((tuple(self.shape), src.is_contiguous()))
        return real_copy(self, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", spy_copy)
    monkeypatch.setattr(torch, "cat", cat_into_plane)
    monkeypatch.setattr(torch, "_foreach_copy_", copy_contiguous_heads)
    monkeypatch.setattr(torch, "stack", lambda *a, **kw: pytest.fail("layer stack"))
    connector.from_gpu(dest, 0, 16, slot_mapping=slots)

    assert torch.equal(dest.tensor, expected)
    assert gather_targets
    target_ptrs = {t.untyped_storage().data_ptr() for t in gather_targets}
    if device_resident:
        assert target_ptrs == {dest.tensor.untyped_storage().data_ptr()}
        assert copies_into_dest == []
    else:
        assert dest.tensor.untyped_storage().data_ptr() not in target_ptrs
        assert copies_into_dest == [((2, 2, 2, 16, 4), True)]


def test_grouped_download_rejects_reclaimed_slots_before_gathering():
    connector = RBLNConnector(2, 1, 2, 8, torch.float32)
    connector.register_kv_caches(
        [torch.zeros(2, 4, 1, 1, 8, 2), torch.zeros(2, 16, 1, 1, 2, 2)],
        layer_group_indices=[0, 1],
    )
    connector._device = torch.device("meta")
    slots = torch.stack(
        (torch.arange(8, 16), torch.tensor([-1] * 4 + [12, 13, 14, 15])), dim=1
    )
    with pytest.raises(ValueError, match="reclaimed"):
        connector.from_gpu(
            SimpleNamespace(tensor=torch.empty(2, 2, 1, 8, 2)), 0, 8, slot_mapping=slots
        )


def test_hybrid_restore_skips_reclaimed_blocks():
    connector = RBLNConnector(2, 1, 2, 8, torch.float32)
    caches = [
        torch.full((2, 4, 1, 1, 8, 2), -1.0),
        torch.full((2, 16, 1, 1, 2, 2), -1.0),
    ]
    connector.register_kv_caches(caches, layer_group_indices=[0, 1])
    slots = torch.stack(
        (torch.arange(8, 16), torch.tensor([-1] * 4 + [12, 13, 14, 15])), dim=1
    )
    source = SimpleNamespace(
        tensor=torch.arange(64, dtype=torch.float32).reshape(2, 2, 1, 8, 2)
    )
    connector.to_gpu(source, 0, 8, slot_mapping=slots)
    assert torch.all(caches[1][:, :6] == -1)
    logical = _token_major(source.tensor, [8, 2])
    for token in range(4, 8):
        block, offset = divmod(int(slots[token, 1]), 2)
        assert torch.equal(caches[1][:, block, :, 0, offset], logical[:, 1, :, token])
    with pytest.raises(ValueError, match="reclaimed"):
        connector.from_gpu(
            SimpleNamespace(tensor=torch.empty_like(source.tensor)),
            0,
            8,
            slot_mapping=slots,
        )


def test_hybrid_transfer_with_interleaved_head_groups():
    connector = RBLNConnector(
        3, 2, 4, 16, torch.float32, kv_head_groups=[(2, 2), (1, 4)]
    )
    caches = [
        torch.full((2, 8, 2, 1, 16, 4), -1.0),
        torch.full((2, 8, 4, 1, 16, 4), -1.0),
        torch.full((2, 32, 2, 1, 4, 4), -1.0),
    ]
    connector.register_kv_caches(caches, layer_group_indices=[1, 1, 0])
    slots = torch.stack((torch.arange(48, 64), torch.arange(16, 32)), dim=1)
    expected = [
        torch.arange(shape.numel(), dtype=torch.float32).reshape(shape) + 1000 * i
        for i, shape in enumerate(connector.get_shapes(16))
    ]
    connector.to_gpu(
        SimpleNamespace(get_tensor=lambda i: expected[i]), 0, 16, slot_mapping=slots
    )
    logical = [_token_major(expected[0], [16, 4]), _token_major(expected[1], [16])]
    for layer, (region, plane, group, kernel) in enumerate(
        [(0, 0, 1, 16), (1, 0, 1, 16), (0, 1, 0, 4)]
    ):
        for token in range(16):
            block, offset = divmod(int(slots[token, group]), kernel)
            assert torch.equal(
                caches[layer][:, block, :, 0, offset],
                logical[region][:, plane, :, token],
            )
        assert torch.all(caches[layer][:, 0] == -1)
    restored = [torch.empty_like(chunk) for chunk in expected]
    connector.from_gpu(
        SimpleNamespace(get_tensor=lambda i: restored[i]), 0, 16, slot_mapping=slots
    )
    assert all(torch.equal(a, b) for a, b in zip(restored, expected, strict=True))


def test_hybrid_transfer_with_head_groups_of_several_head_sizes():
    """Gemma4: sliding layers of 4 heads x 2 and a full layer of 1 head x 4.

    The full layer's page is half the sliding one per token, so vLLM gives
    its group a block of 16 against the sliding 8, and, as on the device,
    the sliding layer of one group and the full layer view one storage
    (sliding kernel block 4: two per page). Two 8-token chunks fill both
    halves of one full block; nothing else in either storage changes.
    """
    connector = RBLNConnector(
        3, 4, 2, 8, torch.float32, kv_head_groups=[(2, 4, 2), (1, 1, 4)]
    )
    pages, page = 6, 128
    shared = torch.full((pages * page,), -1.0)
    alone = torch.full((pages * page,), -1.0)
    caches = [
        shared.view(2 * pages, 2, 4, 1, 4, 2),
        shared.view(pages, 2, 1, 1, 16, 4),
        alone.view(2 * pages, 2, 4, 1, 4, 2),
    ]
    connector.register_kv_caches(caches, layer_group_indices=[0, 1, 0])
    assert connector.get_shapes(8) == [
        torch.Size([2, 2, 4, 8, 2]),
        torch.Size([2, 1, 1, 8, 4]),
    ]
    # Pool pages: the sliding group gets 1 and 2, the full group 3 (16 tokens).
    sliding = torch.cat((torch.arange(8, 16), torch.arange(16, 24)))
    slots = torch.stack((sliding, torch.arange(48, 64)), dim=1)
    chunks = []
    for c in range(2):
        chunk = [
            torch.arange(s.numel(), dtype=torch.float32).reshape(s) + 1000 * (i + 2 * c)
            for i, s in enumerate(connector.get_shapes(8))
        ]
        connector.to_gpu(
            SimpleNamespace(get_tensor=lambda i, chunk=chunk: chunk[i]),
            8 * c,
            8 * c + 8,
            slot_mapping=slots,
        )
        chunks.append(chunk)
    for c, chunk in enumerate(chunks):
        logical = [_token_major(chunk[0], [4, 4]), _token_major(chunk[1], [16])]
        for layer, (region, plane, group, kernel) in enumerate(
            [(0, 0, 0, 4), (1, 0, 1, 16), (0, 1, 0, 4)]
        ):
            for token in range(8):
                block, offset = divmod(int(slots[8 * c + token, group]), kernel)
                assert torch.equal(
                    caches[layer][block, :, :, 0, offset],
                    logical[region][:, plane, :, token],
                )
    # 16 tokens of 16 sliding and 8 full elements each; the rest is untouched.
    assert int((shared != -1).sum()) == 16 * (16 + 8)
    assert int((alone != -1).sum()) == 16 * 16
    for c, chunk in enumerate(chunks):
        restored = [torch.empty_like(t) for t in chunk]
        connector.from_gpu(
            SimpleNamespace(get_tensor=lambda i, r=restored: r[i]),
            8 * c,
            8 * c + 8,
            slot_mapping=slots,
        )
        assert all(torch.equal(a, b) for a, b in zip(restored, chunk, strict=True))


def test_head_groups_are_all_sized_or_none():
    with pytest.raises(ValueError, match="mix sized and unsized"):
        RBLNConnector(3, 2, 4, 8, torch.float32, kv_head_groups=[(2, 2, 4), (1, 1)])


def test_head_groups_refuse_a_cache_of_another_head_size():
    connector = RBLNConnector(2, 2, 4, 8, torch.float32, kv_head_groups=[(2, 2, 4)])
    with pytest.raises(ValueError, match="head_dim 8, connector expects 4"):
        connector.register_kv_caches(
            [torch.zeros(2, 4, 2, 1, 8, 4), torch.zeros(2, 4, 2, 1, 8, 8)],
            layer_group_indices=[0, 0],
        )


def test_scheduler_metadata_observes_each_group_and_reclamation():
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = True
    spec_args = dict(block_size=8, num_kv_heads=1, head_size=2, dtype=torch.float32)
    impl._kv_cache_groups = [
        KVCacheGroupSpec(["model.layers.1.attn"], FullAttentionSpec(**spec_args)),
        KVCacheGroupSpec(
            ["model.layers.0.attn"], SlidingWindowSpec(**spec_args, sliding_window=2)
        ),
    ]
    impl._request_blocks = {}
    impl._save_blocked_req_ids = set()
    impl._manager = MagicMock()
    impl.config = SimpleNamespace(priority_limit=None, save_decode_cache=False)
    impl.kv_role = "kv_both"
    impl.force_skip_save = False
    impl._request_trackers = {}
    impl._unfinished_requests = {}
    impl._requests_priority = {}
    impl.load_specs = {}
    impl._block_size = 8
    impl._lmcache_chunk_size = 4
    impl._discard_partial_chunks = True
    request = SimpleNamespace(
        request_id="req", num_computed_tokens=0, all_token_ids=list(range(24))
    )
    groups = ([KVCacheBlock(3)], [KVCacheBlock(7)])
    impl.update_state_after_alloc(request, 0, KVCacheBlocks(groups))
    new_request = SimpleNamespace(
        req_id="req",
        prompt_token_ids=request.all_token_ids,
        block_ids=([3], [7]),
        num_computed_tokens=0,
        sampling_params=None,
        mm_features=[],
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        num_scheduled_tokens={"req": 8},
    )
    first = impl.build_connector_meta(output).requests[0]
    assert first.slot_mapping.tolist() == [[24 + i, 56 + i] for i in range(8)]
    assert new_request.block_ids == ([3], [7])

    # Simulate the real manager mutating the lists after the next allocation.
    groups[0].append(KVCacheBlock(4))
    groups[1][0] = KVCacheBlock(0, is_null=True)
    groups[1].append(KVCacheBlock(6))
    request.num_computed_tokens = 8
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["req"], new_block_ids=[([4], [6])], resumed_req_ids=set()
    )
    second = impl.build_connector_meta(output).requests[0]
    assert second.slot_mapping[:8].tolist() == [[24 + i, -1] for i in range(8)]
    assert second.slot_mapping[8:].tolist() == [[32 + i, 48 + i] for i in range(8)]
    assert second.save_spec.skip_leading_tokens == 8
    assert first.slot_mapping[:, 1].tolist() == list(range(56, 64))

    # A fresh allocation on resumption replaces the retained list references.
    replacement = KVCacheBlocks(([KVCacheBlock(2)], [KVCacheBlock(5)]))
    impl.update_state_after_alloc(request, 0, replacement)
    assert impl._request_blocks["req"] is replacement
    output.finished_req_ids = {"req"}
    output.scheduled_cached_reqs = CachedRequestData.make_empty()
    impl.build_connector_meta(output)
    assert not impl._request_blocks


def test_scheduler_slots_address_each_gemma4_group_at_its_own_block():
    """Gemma4's groups as vLLM builds them: five sliding groups of 2048-token
    blocks and a full group of 4096. A prompt of three cache blocks ends
    halfway into the full group's second block; sliding blocks below the
    window are reclaimed (null)."""
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = True
    impl._kv_cache_groups = fx.gemma4().kv_cache_config().kv_cache_groups
    sizes = [g.kv_cache_spec.block_size for g in impl._kv_cache_groups]
    assert sizes == [2048] * 5 + [4096]
    impl._request_blocks = {}
    impl._save_blocked_req_ids = set()
    impl._manager = MagicMock()
    impl.config = SimpleNamespace(priority_limit=None, save_decode_cache=False)
    impl.kv_role = "kv_both"
    impl.force_skip_save = False
    impl._request_trackers = {}
    impl._unfinished_requests = {}
    impl._requests_priority = {}
    impl.load_specs = {}
    impl._block_size = impl._lmcache_chunk_size = 2048
    impl._discard_partial_chunks = True
    tokens = 3 * 2048
    # Pool pages: sliding group g holds 10g+1..10g+3, its first one reclaimed;
    # the full group holds 60 and 61.
    ids = [[0, 10 * g + 2, 10 * g + 3] for g in range(5)] + [[60, 61]]
    groups = tuple(
        [KVCacheBlock(i, is_null=(g < 5 and k == 0)) for k, i in enumerate(b)]
        for g, b in enumerate(ids)
    )
    request = SimpleNamespace(
        request_id="req", num_computed_tokens=0, all_token_ids=list(range(tokens))
    )
    impl.update_state_after_alloc(request, 0, KVCacheBlocks(groups))
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="req",
                prompt_token_ids=request.all_token_ids,
                block_ids=tuple(ids),
                num_computed_tokens=0,
                sampling_params=None,
                mm_features=[],
            )
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        num_scheduled_tokens={"req": tokens},
    )
    slots = impl.build_connector_meta(output).requests[0].slot_mapping
    t = torch.arange(tokens)
    for g in range(5):
        want = torch.tensor(ids[g])[t // 2048] * 2048 + t % 2048
        want[t < 2048] = -1
        assert torch.equal(slots[:, g], want), g
    # The third cache block is the first half of full block 61.
    assert torch.equal(slots[:, 5], torch.tensor(ids[5])[t // 4096] * 4096 + t % 4096)
    assert slots[2 * 2048, 5] == 61 * 4096
    assert slots[-1, 5] == 61 * 4096 + 2047


def _grouped_impl():
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = True
    spec_args = dict(block_size=8, num_kv_heads=1, head_size=2, dtype=torch.float32)
    impl._kv_cache_groups = [
        KVCacheGroupSpec(["model.layers.1.attn"], FullAttentionSpec(**spec_args)),
        KVCacheGroupSpec(
            ["model.layers.0.attn"], SlidingWindowSpec(**spec_args, sliding_window=2)
        ),
    ]
    impl._request_blocks = {}
    impl._save_blocked_req_ids = set()
    impl._manager = MagicMock()
    impl.config = SimpleNamespace(priority_limit=None, save_decode_cache=False)
    impl.kv_role = "kv_both"
    impl.force_skip_save = False
    impl._request_trackers = {}
    impl._unfinished_requests = {}
    impl._requests_priority = {}
    impl.load_specs = {}
    impl._block_size = 8
    impl._lmcache_chunk_size = 4
    impl._discard_partial_chunks = True
    return impl


def test_saves_stop_for_good_once_the_window_passes_the_lmcache_prefix():
    impl = _grouped_impl()
    # A local prefix hit already reclaimed the first sliding-window block, and
    # LMCache holds none of the prompt (no load spec): the gap can never fill.
    request = SimpleNamespace(
        request_id="req", num_computed_tokens=8, all_token_ids=list(range(24))
    )
    groups = (
        [KVCacheBlock(3), KVCacheBlock(4)],
        [KVCacheBlock(0, is_null=True), KVCacheBlock(6)],
    )
    impl.update_state_after_alloc(request, 0, KVCacheBlocks(groups))
    new_request = SimpleNamespace(
        req_id="req",
        prompt_token_ids=request.all_token_ids,
        block_ids=([3, 4], [0, 6]),
        num_computed_tokens=8,
        sampling_params=None,
        mm_features=[],
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        num_scheduled_tokens={"req": 8},
    )
    first = impl.build_connector_meta(output).requests[0]
    assert first.slot_mapping[:8, 1].tolist() == [-1] * 8
    # The reclaimed slots are skipped for any role, and this role also stops.
    assert first.save_spec.skip_leading_tokens == 8
    assert not first.save_spec.can_save
    assert impl._save_blocked_req_ids == {"req"}

    # Upstream's tracker already counts the first 16 tokens as saved, so a
    # one-off flip would let the next chunk boundary save the unreachable tail.
    groups[0].append(KVCacheBlock(5))
    groups[1].append(KVCacheBlock(7))
    request.num_computed_tokens = 16
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["req"], new_block_ids=[([5], [7])], resumed_req_ids=set()
    )
    second = impl.build_connector_meta(output).requests[0]
    assert second.save_spec.skip_leading_tokens == 16
    assert not second.save_spec.can_save

    # A preemption resets the request; the resumed request looks up again.
    output.preempted_req_ids = {"req"}
    output.scheduled_cached_reqs = CachedRequestData.make_empty()
    impl.build_connector_meta(output)
    assert not impl._save_blocked_req_ids


def test_a_producer_resumes_at_the_first_complete_chunk_instead_of_stopping():
    # Upstream stores for a producer whatever can_save says and reads
    # skip_leading_tokens as-is, so the reclaimed slots must be skipped there.
    impl = _grouped_impl()
    impl.kv_role = "kv_producer"
    request = SimpleNamespace(
        request_id="req", num_computed_tokens=8, all_token_ids=list(range(24))
    )
    groups = (
        [KVCacheBlock(3), KVCacheBlock(4)],
        [KVCacheBlock(0, is_null=True), KVCacheBlock(6)],
    )
    impl.update_state_after_alloc(request, 0, KVCacheBlocks(groups))
    new_request = SimpleNamespace(
        req_id="req",
        prompt_token_ids=request.all_token_ids,
        block_ids=([3, 4], [0, 6]),
        num_computed_tokens=8,
        sampling_params=None,
        mm_features=[],
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        num_scheduled_tokens={"req": 8},
    )
    meta = impl.build_connector_meta(output).requests[0]
    assert meta.save_spec.skip_leading_tokens == 8
    assert meta.save_spec.can_save
    assert not impl._save_blocked_req_ids
    assert (meta.slot_mapping[meta.save_spec.skip_leading_tokens :] >= 0).all()


def test_a_window_behind_saved_chunks_keeps_saving():
    impl = _grouped_impl()
    request = SimpleNamespace(
        request_id="req", num_computed_tokens=0, all_token_ids=list(range(24))
    )
    groups = ([KVCacheBlock(3)], [KVCacheBlock(7)])
    impl.update_state_after_alloc(request, 0, KVCacheBlocks(groups))
    new_request = SimpleNamespace(
        req_id="req",
        prompt_token_ids=request.all_token_ids,
        block_ids=([3], [7]),
        num_computed_tokens=0,
        sampling_params=None,
        mm_features=[],
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        num_scheduled_tokens={"req": 8},
    )
    first = impl.build_connector_meta(output).requests[0]
    assert first.save_spec.can_save

    # The window moves past chunks this request already saved: no gap in
    # LMCache, so saving continues from where it left off.
    groups[0].append(KVCacheBlock(4))
    groups[1][0] = KVCacheBlock(0, is_null=True)
    groups[1].append(KVCacheBlock(6))
    request.num_computed_tokens = 8
    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["req"], new_block_ids=[([4], [6])], resumed_req_ids=set()
    )
    second = impl.build_connector_meta(output).requests[0]
    assert second.save_spec.skip_leading_tokens == 8
    assert second.save_spec.can_save
    assert not impl._save_blocked_req_ids


@pytest.mark.parametrize("full_block", [8, 16])
def test_failed_load_reports_all_live_group_blocks(full_block):
    """Each column is a block id of its own group's block size (Gemma4's full
    layers: twice the cache block)."""
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = True
    impl._failed_load_req_ids = set()
    impl._block_size = 8
    spec_args = dict(num_kv_heads=1, head_size=2, dtype=torch.float32)
    impl._kv_cache_groups = [
        KVCacheGroupSpec(
            ["model.layers.0.attn"],
            SlidingWindowSpec(block_size=8, sliding_window=2, **spec_args),
        ),
        KVCacheGroupSpec(
            ["model.layers.1.attn"],
            FullAttentionSpec(block_size=full_block, **spec_args),
        ),
    ]
    # Sliding block 1 (reclaimed for the first two tokens) and full block 5.
    full = 5 * full_block + torch.arange(4)
    slots = torch.stack((torch.tensor([-1, -1, 10, 11]), full), dim=1)
    failed = impl.record_failed_blocks(
        "req", torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), slots
    )
    assert failed == {1, 5}


def test_registration_maps_layer_order_to_group_order():
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = True
    impl.config = SimpleNamespace(extra_config={})
    impl._kv_cache_groups = [
        SimpleNamespace(layer_names=["model.layers.1.attn"]),
        SimpleNamespace(layer_names=["model.layers.0.attn"]),
    ]
    impl._cache_layout = CacheLayout(
        kv_regions=(
            KVRegion(1, torch.float32, ("model.layers.0.attn", "model.layers.1.attn")),
        ),
        aux_regions=(),
    )
    impl.kv_caches = {}
    impl._manager = MagicMock()
    device_connector = RBLNConnector(2, 1, 2, 8, torch.float32)
    impl._manager.lmcache_engine.gpu_connector = device_connector
    caches = {
        "model.layers.1.attn": torch.zeros(4, 2, 1, 1, 8, 2),
        "model.layers.0.attn": torch.zeros(16, 2, 1, 1, 2, 2),
    }
    impl.register_kv_caches(caches)
    assert device_connector._layer_group_indices == [1, 0]
    assert device_connector._layout_ref is caches["model.layers.0.attn"]
    assert [kv.shape[-2] for kv in device_connector._kv_caches] == [2, 8]


@pytest.mark.parametrize(
    "invalid", ["chunk", "block_size", "grown_first_group", "stateful_spec"]
)
def test_unsupported_hybrid_layout_rejected_before_services_start(invalid):
    class StatefulSlidingSpec(SlidingWindowSpec):
        pass

    spec_cls = StatefulSlidingSpec if invalid == "stateful_spec" else SlidingWindowSpec
    spec = spec_cls(
        # Not a multiple of the cache block; a multiple, but first.
        block_size={"block_size": 12, "grown_first_group": 16}.get(invalid, 8),
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        sliding_window=2,
    )
    config = LMCacheEngineConfig(chunk_size=16 if invalid == "chunk" else 4)
    parent = SimpleNamespace(
        _kv_cache_config=SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["model.layers.0.attn"], spec)]
        )
    )
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        cache_config=SimpleNamespace(block_size=8),
    )
    with (
        patch(
            f"{CONNECTOR}.lmcache_get_or_create_config",
            return_value=config,
        ),
        patch.object(RBLNLMCacheConnectorV1Impl, "_apply_extra_config"),
        patch(f"{CONNECTOR}.LMCacheManager") as manager,
    ):
        with pytest.raises(ValueError):
            RBLNLMCacheConnectorV1Impl(vllm_config, KVConnectorRole.SCHEDULER, parent)
        manager.assert_not_called()


@pytest.mark.parametrize(
    "hit, prompt, computed, expected",
    [(4, 24, 0, 0), (12, 24, 0, 8), (12, 24, 8, 0), (16, 16, 0, 15)],
)
def test_hybrid_lookup_requires_complete_manager_blocks(
    hit, prompt, computed, expected
):
    impl = object.__new__(RBLNLMCacheConnectorV1Impl)
    impl._use_grouped_slots = True
    impl._block_size = 8
    impl._lmcache_chunk_size = 8
    impl._max_tokens_per_load = 0
    impl.kv_role = "kv_both"
    impl.config = SimpleNamespace(min_retrieve_tokens=0)
    impl._manager = MagicMock()
    impl._manager.lookup_client.lookup_cache.return_value = hit
    impl.load_specs = {}
    request = SimpleNamespace(
        request_id="req", num_tokens=prompt, skip_reading_prefix_cache=False
    )
    assert impl.get_num_new_matched_tokens(request, computed) == expected
    assert impl.load_specs["req"].lmcache_cached_tokens == hit // 8 * 8


@pytest.mark.parametrize("tp, pp, dp", [(1, 1, 1), (1, 1, 2), (2, 1, 1), (1, 2, 1)])
def test_hybrid_init_preserves_chunks_and_checks_parallelism(tp, pp, dp):
    config = LMCacheEngineConfig(chunk_size=4)
    spec = SlidingWindowSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        sliding_window=2,
    )
    parent = SimpleNamespace(
        _kv_cache_config=SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["model.layers.0.attn"], spec)]
        )
    )
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu")),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
        ),
        cache_config=SimpleNamespace(block_size=8),
    )

    def init_state(self, role, vllm_config, config):
        self._block_size = vllm_config.cache_config.block_size
        self._lmcache_chunk_size = config.chunk_size
        self._discard_partial_chunks = False

    with (
        patch(
            f"{CONNECTOR}.lmcache_get_or_create_config",
            return_value=config,
        ),
        patch.object(RBLNLMCacheConnectorV1Impl, "_apply_extra_config"),
        patch.object(RBLNLMCacheConnectorV1Impl, "_init_connector_state", init_state),
        patch.object(RBLNLMCacheConnectorV1Impl, "_setup_metrics"),
        patch(f"{CONNECTOR}.RBLNServiceFactory"),
        patch(f"{CONNECTOR}.LMCacheManager") as manager,
    ):
        if tp > 1 or pp > 1:
            with pytest.raises(ValueError, match="HMA.*parallel_size=1"):
                RBLNLMCacheConnectorV1Impl(
                    vllm_config, KVConnectorRole.SCHEDULER, parent
                )
            manager.assert_not_called()
            return
        impl = RBLNLMCacheConnectorV1Impl(
            vllm_config, KVConnectorRole.SCHEDULER, parent
        )
    assert impl._lmcache_chunk_size == 8
    assert impl._discard_partial_chunks
    assert config.chunk_size == 4


def test_download_staging_reuses_capacity_and_refreshes_contents():
    connector = RBLNConnector(2, 2, 4, 16, torch.float32)
    caches = [torch.zeros(2, 8, 2, 1, 16, 4), torch.zeros(2, 32, 2, 1, 4, 4)]
    connector.register_kv_caches(caches, layer_group_indices=[0, 1])
    original = None
    for length, value in [(8, 1), (4, 2), (16, 3), (8, 4)]:
        for cache in caches:
            cache.fill_(value)
        slots = torch.stack((torch.arange(length), torch.arange(length)), dim=1)
        gathered = connector._gather_grouped(slots)
        assert gathered[0].shape == connector.get_shape(length)
        assert torch.all(gathered[0] == value)
        storage = gathered[0].untyped_storage()
        if original is not None and length <= original[1]:
            assert storage.data_ptr() == original[0]
        original = (storage.data_ptr(), storage.nbytes() // (2 * 2 * 2 * 4 * 4))


def _record_batches(monkeypatch, check=None):
    batches = []
    real_copy = torch._foreach_copy_

    def copy(dsts, srcs):
        if check is not None:
            check(dsts)
        batches.append(len(dsts))
        return real_copy(dsts, srcs)

    monkeypatch.setattr(torch, "_foreach_copy_", copy)
    return batches


def test_mla_upload_uses_host_rows_without_device_staging(monkeypatch):
    connector = RBLNConnector(2, 1, 4, 8, torch.float32, use_mla=True)
    caches = [torch.full((4, 8, 4), -1.0) for _ in range(2)]
    connector.register_kv_caches(caches)
    connector._device = torch.device("meta")
    monkeypatch.setattr(
        connector, "_staging_views", lambda n: pytest.fail("unnecessary MLA staging")
    )
    source = torch.arange(24, dtype=torch.float32).view(1, 2, 1, 3, 4)
    slots = torch.tensor([1, 2, 9])
    connector.to_gpu(SimpleNamespace(tensor=source), 0, 3, slot_mapping=slots)
    for layer, cache in enumerate(caches):
        flat = cache.view(-1, 4)
        assert torch.equal(flat[slots], source[0, layer, 0])
        untouched = torch.ones(32, dtype=torch.bool)
        untouched[slots] = False
        assert torch.all(flat[untouched] == -1)


def test_long_swa_download_preserves_data_across_many_kernel_blocks():
    connector = RBLNConnector(2, 2, 4, 16, torch.float32)
    caches = [torch.zeros(2, 8, 2, 1, 16, 4), torch.zeros(2, 32, 2, 1, 4, 4)]
    connector.register_kv_caches(caches, layer_group_indices=[0, 1])
    slots = torch.stack((torch.arange(64), torch.arange(64)), dim=1)
    expected = torch.arange(2 * 2 * 2 * 64 * 4, dtype=torch.float32).view(
        2, 2, 2, 64, 4
    )
    connector.to_gpu(SimpleNamespace(tensor=expected), 0, 64, slot_mapping=slots)
    connector._device = torch.device("meta")
    result = torch.empty_like(expected)
    connector.from_gpu(SimpleNamespace(tensor=result), 0, 64, slot_mapping=slots)
    assert torch.equal(result, expected)
    assert torch.all(caches[0][:, 4:] == 0)
    assert torch.all(caches[1][:, 16:] == 0)


def test_kernel_block_slabs_tile_the_plane_and_invert():
    plane = torch.arange(2 * 10 * 3, dtype=torch.float32).view(2, 10, 3)
    slabs = kernel_block_slabs(plane, 4)
    assert [tuple(s.shape) for s in slabs] == [(2, 4, 3), (2, 4, 3), (2, 2, 3)]
    # Consecutive, contiguous slabs that cover the plane exactly once.
    assert all(s.is_contiguous() for s in slabs)
    assert torch.equal(torch.cat([s.reshape(-1) for s in slabs]), plane.reshape(-1))
    # A kernel block covering the chunk is the plain plane, a divisor no tail.
    (whole,) = kernel_block_slabs(plane, 16)
    assert torch.equal(whole, plane)
    assert [tuple(s.shape) for s in kernel_block_slabs(plane, 5)] == [(2, 5, 3)] * 2


def test_slot_runs_end_at_chunk_kernel_blocks():
    # A chunk starting mid-block: slots 4..7 share a kernel block but cross
    # chunk kernel block 1 at token 4, so the run splits there.
    assert slot_runs(list(range(2, 10)), 4) == [
        (0, 2, 0, 2),
        (1, 0, 2, 4),
        (1, 2, 4, 6),
        (2, 0, 6, 8),
    ]


def _shared_pool_connector():
    """GPT-OSS in miniature: an SWA (kernel 4) / full (kernel 16) pair sharing
    one SWA-shaped, blocks-first allocation with 4 KV heads."""
    pool = torch.zeros(32, 2, 4, 1, 4, 2)
    caches = [pool, pool.view(8, 2, 4, 1, 16, 2)]
    connector = RBLNConnector(2, 4, 2, 16, torch.float32)
    connector.register_kv_caches(caches, layer_group_indices=[1, 0])
    # Full layer: manager block 1 (pool kernel blocks 4..7). SWA layer: kernel
    # blocks 8..11 (manager block 2).
    slots = torch.stack((torch.arange(16, 32), torch.arange(32, 48)), dim=1)
    return connector, pool, slots


def test_swa_kernel_blocks_move_as_one_copy_each(monkeypatch):
    connector, pool, slots = _shared_pool_connector()
    pool.copy_(torch.arange(pool.numel(), dtype=torch.float32).view_as(pool))
    connector._device = torch.device("meta")  # a host destination
    real_foreach = torch._foreach_copy_
    calls = []

    def record(dsts, srcs):
        calls.append([tuple(t.shape) for t in srcs])
        assert all(t.is_contiguous() for t in (*dsts, *srcs))
        return real_foreach(dsts, srcs)

    monkeypatch.setattr(torch, "_foreach_copy_", record)
    dest = SimpleNamespace(tensor=torch.empty(2, 2, 4, 16, 2))
    connector.from_gpu(dest, 0, 16, slot_mapping=slots)
    # One batch. SWA: one [H, 4, D] copy per kernel block and K/V; full: one
    # [H, 16, D] copy per K/V, however many physical mappings it spans (a D2D
    # copy may span up to 1024 since rebel_compiler#15445).
    (pieces,) = calls
    assert sorted(pieces) == sorted([(4, 4, 2)] * 8 + [(4, 16, 2)] * 2)
    # The SWA plane is the four cache kernel blocks, back to back; the full
    # plane is its manager block as is.
    swa, full = connector._kv_caches
    for half in range(2):
        for i in range(4):
            assert torch.equal(
                kernel_block_slabs(dest.tensor[half, 0], 4)[i], swa[half, 8 + i, :, 0]
            )
        assert torch.equal(dest.tensor[half, 1], full[half, 1, :, 0])

    # Restoring the chunk into a fresh pool writes exactly those blocks.
    fresh, fresh_pool, _ = _shared_pool_connector()
    fresh.to_gpu(dest, 0, 16, slot_mapping=slots)
    touched = torch.zeros(32, dtype=torch.bool)
    touched[4:12] = True
    assert torch.equal(fresh_pool[touched], pool[touched])
    assert torch.all(fresh_pool[~touched] == 0)


def test_grouped_pieces_pair_equal_shapes():
    connector, _, slots = _shared_pool_connector()
    caches, chunk_pieces = connector._grouped_pieces(
        [torch.empty(2, 2, 4, 16, 2)], slots.T.tolist()
    )
    assert caches
    for cache_piece, chunk_piece in zip(caches, chunk_pieces, strict=True):
        assert cache_piece.shape == chunk_piece.shape


@pytest.mark.parametrize("device_resident", [False, True])
def test_grouped_round_trip_matches_a_physical_address_oracle(device_resident):
    """Store and restore against element addresses computed from scratch.

    The chunk format is stored unversioned, so the general path is checked
    against the physical coordinates themselves. Kernel 4 and two heads.
    Group 0 starts mid-block (slots 2..11): partial runs only, one starting
    mid-slab (tokens 2..3) and one in a later slab (tokens 6..7). Group 1 is
    block aligned (slots 4..13): whole blocks, then the short last slab of a
    10-token chunk.
    """
    heads, kernel, dim, tokens, blocks = 2, 4, 3, 10, 8
    numel = 2 * blocks * heads * kernel * dim
    caches = [
        torch.arange(n * numel, (n + 1) * numel, dtype=torch.float32).view(
            2, blocks, heads, 1, kernel, dim
        )
        for n in range(2)
    ]
    connector = RBLNConnector(2, heads, dim, 16, torch.float32)
    connector.register_kv_caches(caches, layer_group_indices=[0, 1])
    slots = torch.stack((torch.arange(2, 12), torch.arange(4, 14)), dim=1)

    def cache_index(half, block, head, offset, d):
        return (((half * blocks + block) * heads + head) * kernel + offset) * dim + d

    def chunk_index(half, layer, head, token, d):
        # Slab ``token // kernel`` of the layer plane, ``width`` tokens wide.
        index, column = divmod(token, kernel)
        width = min(kernel, tokens - index * kernel)
        plane = (half * 2 + layer) * heads * tokens * dim
        slab = plane + index * kernel * heads * dim
        return slab + (head * width + column) * dim + d

    pairs = []  # (layer, cache element, chunk element)
    for layer in range(2):
        for token in range(tokens):
            block, offset = divmod(int(slots[token, layer]), kernel)
            for half in range(2):
                for head in range(heads):
                    for d in range(dim):
                        pairs.append(
                            (
                                layer,
                                cache_index(half, block, head, offset, d),
                                chunk_index(half, layer, head, token, d),
                            )
                        )
    chunk = torch.full((2, 2, heads, tokens, dim), -1.0)
    # The oracle places every chunk element exactly once.
    assert sorted(k for _, _, k in pairs) == list(range(chunk.numel()))

    if not device_resident:
        connector._device = torch.device("meta")
    connector.from_gpu(SimpleNamespace(tensor=chunk), 0, tokens, slot_mapping=slots)
    flat = chunk.view(-1)
    for layer, c, k in pairs:
        assert flat[k] == caches[layer].view(-1)[c]

    fresh = [torch.full_like(cache, -1.0) for cache in caches]
    restore = RBLNConnector(2, heads, dim, 16, torch.float32)
    restore.register_kv_caches(fresh, layer_group_indices=[0, 1])
    restore.to_gpu(SimpleNamespace(tensor=chunk), 0, tokens, slot_mapping=slots)
    written = [torch.zeros(numel, dtype=torch.bool) for _ in fresh]
    for layer, c, k in pairs:
        assert fresh[layer].view(-1)[c] == flat[k]
        written[layer][c] = True
    for cache, mask in zip(fresh, written, strict=True):
        assert torch.all(cache.view(-1)[~mask] == -1)


@pytest.mark.parametrize("kernel", [4, 5])
def test_kernel_block_slabs_refuse_a_non_contiguous_plane(kernel):
    # Slabs over a strided plane would not be the stored order, and a copy
    # would take the writes instead of the chunk. Kernel 5 divides the plane:
    # a plain view of it would succeed, so the check must not rely on one.
    plane = torch.zeros(10, 2, 3).transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        kernel_block_slabs(plane, kernel)


def test_mla_scatter_moves_scattered_slots_run_by_run(monkeypatch):
    connector = RBLNConnector(2, 1, 4, 8, torch.float32, use_mla=True)
    caches = [torch.full((4, 8, 4), -1.0) for _ in range(2)]
    connector.register_kv_caches(caches)
    # Offsets 1, 2 | 5 in block 0 (two runs), 1 in block 1.
    slots = torch.tensor([1, 2, 5, 9])
    source = torch.arange(32, dtype=torch.float32).view(1, 2, 1, 4, 4)
    batches = _record_batches(monkeypatch)
    connector.to_gpu(SimpleNamespace(tensor=source), 0, 4, slot_mapping=slots)
    # One batched call: three runs for each of the two layers.
    assert batches == [6]
    for layer, cache in enumerate(caches):
        assert torch.equal(cache.view(-1, 4)[slots], source[0, layer, 0])

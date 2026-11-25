# SPDX-License-Identifier: Apache-2.0
"""The H2D staging buffer takes the KV cache's device layout only when 6-D.

``set_device_layout_like`` is what keeps the staging -> KV scatter on the
device-native path for the 6-D layout (0.6 ms vs 4.5 ms per scatter on
Qwen3-1.7B). Under MLA's 3-D layout it does the opposite: the copy leaves
the device-native path and the runtime services every 288 KB scatter by
pulling the whole KV cache off all four chiplets to the host, so the cost
tracks cache size (0.19 ms without it; 1.7 s at a 1.3 GB cache, 41 s at
29.5 GB -- past vLLM's step window, which kills the engine).
"""

# Standard
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.rbln.v1.rbln_connector import RBLNConnector

NUM_LAYERS = 2
NUM_KV_HEADS = 4
HEAD_DIM = 64
BLOCK_SIZE = 16
NUM_BLOCKS = 4


def _make_connector(use_mla: bool) -> RBLNConnector:
    conn = RBLNConnector(
        num_layers=NUM_LAYERS,
        num_kv_heads=1 if use_mla else NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        dtype=torch.float16,
        use_mla=use_mla,
    )
    if use_mla:
        caches = [
            torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM, dtype=torch.float16)
            for _ in range(NUM_LAYERS)
        ]
    else:
        caches = [
            torch.zeros(
                2,
                NUM_BLOCKS,
                NUM_KV_HEADS,
                1,
                BLOCK_SIZE,
                HEAD_DIM,
                dtype=torch.float16,
            )
            for _ in range(NUM_LAYERS)
        ]
    conn.register_kv_caches(caches)
    return conn


@pytest.fixture
def spy_set_layout():
    with patch("torch_rbln.memory.set_device_layout_like") as spy:
        yield spy


def test_six_dim_staging_takes_the_kv_layout(spy_set_layout):
    conn = _make_connector(use_mla=False)

    conn._staging_views(BLOCK_SIZE)

    spy_set_layout.assert_called_once()
    buf, like = spy_set_layout.call_args[0]
    assert buf is conn._staging[0]
    assert like is conn._kv_caches[0]


def test_mla_staging_keeps_the_default_layout(spy_set_layout):
    conn = _make_connector(use_mla=True)

    conn._staging_views(BLOCK_SIZE)

    spy_set_layout.assert_not_called()


def test_mla_staging_view_still_has_the_chunk_shape(spy_set_layout):
    """Skipping the layout call must not change what the caller gets."""
    conn = _make_connector(use_mla=True)

    views = conn._staging_views(BLOCK_SIZE)

    assert len(views) == 1  # one KV head group
    assert views[0].shape == (1, NUM_LAYERS, 1, BLOCK_SIZE, HEAD_DIM)
    assert views[0].dtype == torch.float16


def test_staging_buffer_is_reused_across_calls(spy_set_layout):
    """The buffer is allocated once, so the layout call happens once too."""
    conn = _make_connector(use_mla=False)

    first = conn._staging_views(BLOCK_SIZE)
    second = conn._staging_views(BLOCK_SIZE)

    assert first[0].data_ptr() == second[0].data_ptr()
    spy_set_layout.assert_called_once()


def test_each_head_group_gets_its_own_laid_out_buffer(spy_set_layout):
    """A draft with another KV head count is a second head group: its staging
    region is a buffer of its own, starting at offset 0 like every region."""
    conn = RBLNConnector(
        num_layers=2,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        dtype=torch.float16,
        kv_head_groups=[(1, NUM_KV_HEADS), (1, 2 * NUM_KV_HEADS)],
    )
    conn.register_kv_caches(
        [
            torch.zeros(2, NUM_BLOCKS, h, 1, BLOCK_SIZE, HEAD_DIM, dtype=torch.float16)
            for h in (NUM_KV_HEADS, 2 * NUM_KV_HEADS)
        ]
    )

    views = conn._staging_views(BLOCK_SIZE)

    assert [v.shape[2] for v in views] == [NUM_KV_HEADS, 2 * NUM_KV_HEADS]
    assert [c.args for c in spy_set_layout.call_args_list] == [
        (conn._staging[0], conn._layout_ref),
        (conn._staging[1], conn._layout_ref),
    ]
    buffers = conn._staging
    assert [v.data_ptr() for v in views] == [b.data_ptr() for b in buffers]


# MiniMax-M3 keeps a key-only indexer cache ``[NB, BS, w]`` next to the 6-D
# K/V (#256). Its staging takes the indexer cache's own layout, so the scatter
# into it is the device-side copy the layout API promises between the two.
AUX_LAYERS = 2
AUX_WIDTH = 32


def _make_aux_connector(
    use_mla: bool,
    aux_dtype: torch.dtype,
    kv_dtype: torch.dtype = torch.float16,
    blocks_first: bool = False,
) -> RBLNConnector:
    conn = RBLNConnector(
        num_layers=NUM_LAYERS,
        num_kv_heads=1 if use_mla else NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        dtype=kv_dtype,
        use_mla=use_mla,
        aux_groups=[(AUX_LAYERS, AUX_WIDTH, aux_dtype)],
    )
    if use_mla:
        kv_shape: tuple[int, ...] = (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM)
    elif blocks_first:
        # vLLM-RBLN's default allocation; registered as a transposed view.
        kv_shape = (NUM_BLOCKS, 2, NUM_KV_HEADS, 1, BLOCK_SIZE, HEAD_DIM)
    else:
        kv_shape = (2, NUM_BLOCKS, NUM_KV_HEADS, 1, BLOCK_SIZE, HEAD_DIM)
    conn.register_kv_caches(
        [torch.zeros(kv_shape, dtype=kv_dtype) for _ in range(NUM_LAYERS)],
        aux_kv_caches=[
            [
                torch.zeros(NUM_BLOCKS, BLOCK_SIZE, AUX_WIDTH, dtype=aux_dtype)
                for _ in range(AUX_LAYERS)
            ]
        ],
    )
    return conn


@pytest.mark.parametrize(
    "blocks_first", [False, True], ids=["kv-first", "blocks-first"]
)
@pytest.mark.parametrize(
    "aux_dtype, kv_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.uint8, torch.uint8),
        (torch.bfloat16, torch.float16),
    ],
    ids=["KV16", "KV8 uint8 container", "bf16 indexer, fp16 K/V"],
)
def test_each_region_takes_the_layout_of_the_cache_it_fills(
    spy_set_layout, aux_dtype: torch.dtype, kv_dtype: torch.dtype, blocks_first: bool
):
    """The head buffer is laid out against the K/V allocation (not the possibly
    transposed view), the aux buffer against its own indexer cache -- in any
    dtype pairing, since each buffer shares its reference's dtype."""
    conn = _make_aux_connector(
        use_mla=False,
        aux_dtype=aux_dtype,
        kv_dtype=kv_dtype,
        blocks_first=blocks_first,
    )

    views = conn._staging_views(BLOCK_SIZE)

    head, aux = conn._staging
    assert [c.args for c in spy_set_layout.call_args_list] == [
        (head, conn._layout_ref),
        (aux, conn._aux_kv_caches[0][0]),
    ]
    assert views[1].shape == (1, AUX_LAYERS, 1, BLOCK_SIZE, AUX_WIDTH)
    assert views[1].dtype == aux_dtype


@pytest.mark.parametrize("aux_dtype", [torch.float16, torch.bfloat16])
def test_mla_aux_staging_keeps_the_default_layout(
    spy_set_layout, aux_dtype: torch.dtype
):
    """Under MLA no region takes a layout: the 3-D latent cache's is the #189
    host-shadow trap."""
    conn = _make_aux_connector(use_mla=True, aux_dtype=aux_dtype)

    conn._staging_views(BLOCK_SIZE)

    spy_set_layout.assert_not_called()


def test_aux_staging_takes_the_layout_again_when_it_grows(spy_set_layout):
    """A larger chunk reallocates the buffer; the new one needs the call too."""
    conn = _make_aux_connector(use_mla=False, aux_dtype=torch.float16)

    conn._staging_views(BLOCK_SIZE // 2)
    conn._staging_views(BLOCK_SIZE // 2)
    conn._staging_views(BLOCK_SIZE)
    grown = conn._staging[1]
    smaller = conn._staging_views(BLOCK_SIZE // 4)  # reuses the grown buffer

    aux_calls = [
        c.args[0]
        for c in spy_set_layout.call_args_list
        if c.args[1] is conn._aux_kv_caches[0][0]
    ]
    assert len(aux_calls) == 2
    assert aux_calls[-1] is grown
    assert conn._staging[1] is grown
    assert smaller[1].data_ptr() == grown.data_ptr()

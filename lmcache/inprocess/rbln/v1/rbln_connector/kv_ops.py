# SPDX-License-Identifier: Apache-2.0
"""6-D head-major KV gather/scatter kernels.

``RBLNConnector`` moves the native RBLN KV cache
(6-D ``[2, num_blocks, num_kv_heads, 1, block_size, head_dim]``, heads before
block tokens = HND) into a **head-major** chunk ``[2, L, H, T, D]`` and back,
with NO head<->token transpose.

vLLM-RBLN allocates that cache blocks-first (``[NB, 2, ...]``) on its default
kernel path; :func:`as_kv_first` puts the K/V axis back in front once, at
registration, so everything below reads the one layout.

MLA caches are 3-D (``[num_blocks, block_size, head_size]``: one latent vector
per token, no K/V pair and no head axis) and produce a ``[1, L, 1, T, D]``
chunk. Both kernels branch on that layout.

The functions here handle the whole-block case (every gathered block contributes
all ``block_size`` slots), which is the fast path of ``RBLNConnector``.
Partial-offset gathers stay in ``RBLNConnector`` because they arise only from
the token-precise offload slot_mapping; the pure layout arithmetic they share
(:func:`slot_runs`, :func:`kernel_block_slabs`) lives here.
"""

# Standard
from typing import List, Sequence
import os

# Third Party
import torch


def as_kv_first(kv_cache: torch.Tensor) -> torch.Tensor:
    """Return a ``[2, NB, NH, 1, BS, HD]`` view of a paged attention KV cache.

    vLLM-RBLN allocates ``num_blocks`` ahead of the K/V axis
    (``[NB, 2, NH, 1, BS, HD]``) on the rbln_custom_ops path, which is its
    default, and behind it on rbln_triton_ops (RBLN-SW/vllm-rbln#1114). Every
    gather/scatter site here indexes the K/V-first form, so a blocks-first
    tensor is transposed back once at registration -- stride-only, no copy,
    and the two axes it swaps are the outermost, so each indexed plane below
    is exactly as contiguous as before.

    Args:
        kv_cache: Per-layer KV tensor in either 6-D attention layout, or an
            MLA cache (3-D), which has no K/V axis and is returned unchanged.

    Returns:
        torch.Tensor: A K/V-first view, or ``kv_cache`` itself when it already
        is one. Idempotent, so callers may apply it more than once.

    Note:
        A two-block cache is shape-identical under both layouts and is read as
        K/V-first. vLLM allocates orders of magnitude more blocks than that.
    """
    if kv_cache.ndim == 6 and kv_cache.shape[1] == 2 and kv_cache.shape[0] != 2:
        return kv_cache.transpose(0, 1)
    return kv_cache


def split_kv_copy_enabled() -> bool:
    """Whether to copy K and V halves separately (``LMCACHE_RBLN_SPLIT_KV_COPY``).

    Writing ``dst.copy_(torch.stack([k, v]))`` in one shot can corrupt a half
    through the underlying v2v stack-pair copy on some RBLN geometries; the
    split copy makes each half byte-exact. Gated so both paths can be A/B'd.
    """
    return bool(os.environ.get("LMCACHE_RBLN_SPLIT_KV_COPY"))


def gather_full_blocks_headmajor(
    kv_caches: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    dst: torch.Tensor,
) -> None:
    """Gather whole paged blocks into a head-major chunk ``dst``.

    Args:
        kv_caches: Per-layer KV tensors, each 6-D
            ``[2, num_blocks, num_kv_heads, 1, block_size, head_dim]`` on device,
            or 3-D ``[num_blocks, block_size, head_size]`` under MLA.
        block_ids: Block indices to gather, in chunk-token order.
        dst: Destination chunk view shaped
            ``[2, L, H, len(block_ids) * block_size, D]``
            (``[1, L, 1, ...]`` for MLA). May be on device (D2D) or CPU (D2H).

    Reads are batched into a single ``torch._foreach_copy_``, mirroring
    :func:`scatter_headmajor_to_full_blocks`.
    """
    # One pair per (block, layer, half), straight from the paged KV — the
    # stack/cat form this replaces built a chunk-sized device tensor first, so
    # the payload moved twice. A multi-block chunk makes the destination views
    # strided and the batch falls to a sequential loop; splitting the head axis
    # would keep it, but at 8x the descriptors that measures slower (see the
    # commit). The single-block chunk the serve uses is contiguous either way.
    mla = kv_caches[0].ndim == 3
    block_size = kv_caches[0].shape[1 if mla else 4]
    dsts: List[torch.Tensor] = []
    srcs: List[torch.Tensor] = []
    for i, b in enumerate(block_ids):
        tok = slice(i * block_size, (i + 1) * block_size)
        for layer_idx, kv in enumerate(kv_caches):
            if mla:
                # kv: [NB, BS, HS], dst: [1, L, 1, T, HS]
                dsts.append(dst[0, layer_idx][0, tok, :])
                srcs.append(kv[b])
            else:
                for half in (0, 1):
                    dsts.append(dst[half, layer_idx][:, tok, :])
                    srcs.append(kv[half, b, :, 0, :])
    if dsts:
        torch._foreach_copy_(dsts, srcs)


def scatter_headmajor_to_full_blocks(
    kv_caches: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    src: torch.Tensor,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Scatter a head-major chunk ``src`` back into whole paged blocks.

    Args:
        kv_caches: Per-layer KV tensors, each 6-D
            ``[2, num_blocks, num_kv_heads, 1, block_size, head_dim]`` on device,
            or 3-D ``[num_blocks, block_size, head_size]`` under MLA.
        block_ids: Destination block indices, in chunk-token order.
        src: Source chunk shaped ``[2, L, H, len(block_ids) * block_size, D]``
            (``[1, L, 1, ...]`` for MLA).
        skip_prefix_n_blocks: Leading blocks already present in the KV cache;
            neither read from ``src`` nor written (whole-block prefix skip).

    Writes are batched into a single ``torch._foreach_copy_``.
    """
    dsts, srcs = full_block_scatter_pairs(
        kv_caches, block_ids, src, skip_prefix_n_blocks
    )
    if dsts:
        torch._foreach_copy_(dsts, srcs)


def full_block_scatter_pairs(
    kv_caches: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    src: torch.Tensor,
    skip_prefix_n_blocks: int = 0,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """The ``(cache view, src view)`` pairs that
    :func:`scatter_headmajor_to_full_blocks` copies, for a caller that batches
    several cache groups into one ``torch._foreach_copy_``."""
    mla = kv_caches[0].ndim == 3
    block_size = kv_caches[0].shape[1 if mla else 4]
    dsts: List[torch.Tensor] = []
    srcs: List[torch.Tensor] = []
    for i, b in enumerate(block_ids):
        if i < skip_prefix_n_blocks:
            continue
        s = src[:, :, :, i * block_size : (i + 1) * block_size, :]
        for layer_idx, kv in enumerate(kv_caches):
            if mla:
                # kv: [NB, BS, HS], src: [1, L, 1, T, HS]
                dsts.append(kv[b])
                srcs.append(s[0, layer_idx, 0])
            else:
                dsts.append(kv[0, b, :, 0, :])
                srcs.append(s[0, layer_idx])
                dsts.append(kv[1, b, :, 0, :])
                srcs.append(s[1, layer_idx])
    return dsts, srcs


def slot_runs(slots: list[int], kernel_size: int) -> list[tuple[int, int, int, int]]:
    """Split the slots into ``(block, offset, token, stop)`` runs.

    A run is the longest stretch of chunk tokens whose slots are consecutive
    inside one kernel block, and it also ends at multiples of ``kernel_size``
    in chunk tokens, so it stays inside one chunk kernel block (one slab of
    :func:`kernel_block_slabs`). A run of ``kernel_size`` tokens is therefore a
    whole kernel block on both sides: ``offset`` is 0 and ``token`` a multiple
    of ``kernel_size``. Negative slots belong to reclaimed sliding-window
    blocks and are left out.
    """
    runs: list[tuple[int, int, int, int]] = []
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


def kernel_block_slabs(plane: torch.Tensor, kernel_block: int) -> list[torch.Tensor]:
    """The ``[H, w, D]`` slabs a grouped chunk plane is stored as, in order.

    A grouped chunk keeps every layer plane (``[H, T, D]`` elements) as
    consecutive ``[H, w, D]`` slabs, one per ``kernel_block`` chunk tokens
    (``w`` is ``kernel_block`` except for a shorter last slab). Each full slab
    has exactly the layout of one cache kernel block, so a whole block moves as
    one contiguous copy instead of one per head. When ``kernel_block`` covers
    the whole chunk the one slab is the plain ``[H, T, D]`` plane. Chunk token
    ``t`` sits in slab ``t // kernel_block``, column ``t % kernel_block``.

    This is a stored format. The order depends on each layer's kernel block
    size, which vllm-rbln chooses (128 = the window for GPT-OSS SWA layers by
    default, the manager block under sub-block caching). Chunk keys do not
    record it, so every worker that shares a store must run the same SWA kernel
    block size. Chunks written before this layout (release-v0.12.0 wrote
    token-major SWA planes) read back scrambled: flush any persistent store
    shared across that upgrade.

    Args:
        plane: A contiguous ``[H, T, D]`` chunk plane.
        kernel_block: Kernel block size of the layer the plane belongs to.
    """
    if not plane.is_contiguous():
        # Slabs are runs of the plane's memory. Views over a strided plane
        # would follow its logical order instead, and ``reshape`` would hand
        # back a copy that the transfer fills instead of the chunk.
        raise ValueError("kernel_block_slabs needs a contiguous [H, T, D] plane")
    heads, tokens, dim = plane.shape
    full, tail = divmod(tokens, kernel_block)
    if not tail:
        # One view op per plane: this runs for every layer and K/V half of a
        # chunk, and the three of the tail path cost GPT-OSS ~0.1 ms a chunk.
        return list(plane.view(full, heads, kernel_block, dim).unbind(0))
    split = full * heads * kernel_block * dim
    flat = plane.view(-1)
    slabs = list(flat[:split].view(full, heads, kernel_block, dim).unbind(0))
    slabs.append(flat[split:].view(heads, tail, dim))
    return slabs

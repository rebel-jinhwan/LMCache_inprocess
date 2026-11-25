# SPDX-License-Identifier: Apache-2.0
"""RBLN NPU connector for LMCache.

vllm-rbln runs in device-tensor mode so KV caches are real ``torch`` tensors
allocated on ``device='rbln'``.  Under that mode, D2H uses a K/V-split
``stack → index_select → cat`` gather on device followed by a single trailing
DMA, and H2D uses a single bulk H2D then a per-block K/V-split scatter
batched into one ``torch._foreach_copy_`` per block — moving exactly the
requested token slots and avoiding whole-block DMA when RBLN block_size >>
LMCache chunk_size.

vLLM-RBLN allocates the attention cache blocks-first (``[num_blocks, 2, ...]``)
on its default kernel path and K/V-first on rbln_triton_ops; ``register_kv_caches``
stores a K/V-first view either way, so every gather/scatter site below reads one
layout.

MLA models keep one latent vector per token instead of a K/V pair, so their
cache is 3-D (``[num_blocks, block_size, head_size]``) and the chunk's leading
axis is 1.  The gather/scatter sites index it directly under
``if self.use_mla``; the surrounding block grouping, contiguity handling and
``torch._foreach_copy_`` batching are shared with the K/V path.

Host memory registration is left to the deployment:
:meth:`RBLNConnector.register_host_buffer` registers nothing here, so every
transfer takes the ordinary copy path. lmcache-rbln replaces it to register
its huge-host CPU pools on the consuming device.

The head-major gather/scatter kernels below handle the whole-block case (every
gathered block contributes all ``block_size`` slots), which is the fast path.
Partial-offset gathers stay in ``RBLNConnector`` because they arise only from
the token-precise offload slot_mapping; the pure layout arithmetic they share
is :func:`slot_runs` and :func:`kernel_block_slabs`.
"""

# Standard
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Sequence, Union
import math
import os

# Third Party
from typing_extensions import override
import torch

# First Party
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.inprocess.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Head-major KV gather/scatter kernels
#
# The native RBLN KV cache is 6-D ``[2, num_blocks, num_kv_heads, 1,
# block_size, head_dim]`` (heads before block tokens = HND); a chunk is
# head-major ``[2, L, H, T, D]``, with NO head<->token transpose. MLA caches
# are 3-D ``[num_blocks, block_size, head_size]`` and produce a
# ``[1, L, 1, T, D]`` chunk.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


def sized_head_groups(groups: Sequence[Sequence[int]]) -> bool:
    """Whether head groups carry their own head size, as a third element.

    All or none: a pair among triples would have no head size of its own, so
    a mix is refused.
    """
    lengths = {len(g) for g in groups}
    if len(lengths) > 1:
        raise ValueError(f"kv_head_groups mix sized and unsized entries: {groups}")
    return lengths == {3}


class RBLNConnector(GPUConnectorInterface):
    """RBLN NPU connector (device-tensor mode).

    Caller flow:
        1. Construct with model/cache geometry.
        2. After vLLM allocates KV caches, call ``register_kv_caches`` with
           the per-layer device tensor list (already sorted by layer index).
        3. ``from_gpu`` / ``to_gpu`` move data between those device tensors
           and CPU ``MemoryObj`` storage.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension of each attention head.
        block_size: Number of tokens per KV cache block.
        dtype: Data type of KV cache tensors.
        use_mla: Multi-head Latent Attention model. The cache holds one latent
            vector per token instead of a K/V pair, so layers arrive 3-D
            (``[num_blocks, block_size, head_size]``) and every chunk's leading
            axis is 1 instead of 2. ``num_kv_heads`` is 1 and ``head_dim`` is
            the latent size (``kv_lora_rank + qk_rope_head_dim``), both as vLLM
            reports them for MLA. Selects the MLA indexing at every
            gather/scatter site.
        kv_head_groups: ``(num_layers, num_kv_heads)`` per run of layers that
            share one KV head count, in layer order. Defaults to the single
            group ``[(num_layers, num_kv_heads)]``. A speculative draft can
            bring its own head count (eagle3 keeps MHA heads over a GQA
            target: MiniMax-M2.7 has 8, its draft 48); each group is then
            gathered into its own region of the chunk, the layout upstream's
            ``KVLayerGroupsManager`` describes and ``MemoryObj.get_tensor(i)``
            hands out. An entry may carry a third element, the group's head
            size, where groups differ in it (Gemma4: 256 sliding, 512 full);
            ``head_dim`` is the head size of a group without one.
        aux_groups: ``(num_layers, width, dtype)`` per per-token cache that
            travels with the attention KV in the same chunk -- the DSA indexer
            key and its scale. Registered as :meth:`register_kv_caches`'s
            ``aux_kv_caches``; each is a 3-D ``[NB, BS, width]`` cache (a 2-D
            ``[NB, BS]`` one is read as width 1) and fills one ``[1, L, 1, T,
            width]`` region after the head groups, moved by the MLA-layout path.
        block_groups: ``(num_caches, block_shape, dtype, checkpoint)`` per
            cache with no row per token (``CacheLayout`` block regions): a KDA
            state, an indexer key of several tokens per row, a block-aux cache.
            Registered as :meth:`register_kv_caches`'s ``block_kv_caches``;
            each fills one ``[1, N, *block_shape]`` region after the token
            regions with one whole block per cache, so a chunk is one block.
            A checkpoint cache keeps a block only where a request may resume,
            and a load skips it where its block-table entry is null.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
        use_mla: bool = False,
        kv_head_groups: Optional[Sequence[tuple[int, ...]]] = None,
        aux_groups: Optional[Sequence[tuple[int, int, torch.dtype]]] = None,
        slot_registration: bool = False,
        block_groups: Optional[
            Sequence[tuple[int, tuple[int, ...], torch.dtype, bool]]
        ] = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_dim_size = num_kv_heads * head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.use_mla = use_mla
        # Leading axis of every chunk: K and V, or the single MLA latent.
        self.kv_dim = 1 if use_mla else 2
        self.kv_head_groups: list[tuple[int, ...]] = (
            [tuple(int(x) for x in g) for g in kv_head_groups]
            if kv_head_groups
            else ([(num_layers, num_kv_heads)] if num_layers else [])
        )
        # (num_layers, num_kv_heads, head_size) of every head group.
        self._sized = sized_head_groups(self.kv_head_groups)
        self._head_geometry: list[tuple[int, int, int]] = [
            (g[0], g[1], g[2] if self._sized else head_dim) for g in self.kv_head_groups
        ]
        covered = sum(n for n, _, _ in self._head_geometry)
        if covered != num_layers:
            raise ValueError(
                f"kv_head_groups {self.kv_head_groups} cover {covered} layers, "
                f"expected {num_layers}"
            )
        # Layer indices per group, consecutive until ``register_kv_caches``
        # reads the real head count of every layer.
        self._group_layers: list[list[int]] = []
        offset = 0
        for n, _, _ in self._head_geometry:
            self._group_layers.append(list(range(offset, offset + n)))
            offset += n

        self.aux_groups: list[tuple[int, int, torch.dtype]] = [
            (int(n), int(w), d) for n, w, d in (aux_groups or [])
        ]
        # Only full-chunk allocation keeps the CPU pool on a fixed object grid.
        # The factory enables this when LMCache does not save unfull chunks.
        self.slot_registration = slot_registration
        self._aux_kv_caches: List[List[torch.Tensor]] = []
        self.block_groups: list[tuple[int, tuple[int, ...], torch.dtype, bool]] = [
            (int(n), tuple(shape), d, bool(c))
            for n, shape, d, c in (block_groups or [])
        ]
        # Per block region, each cache with the KV cache group whose block
        # table addresses it.
        self._block_caches: List[List[tuple[torch.Tensor, int]]] = []

        self._kv_caches: List[torch.Tensor] = []
        # The first KV cache exactly as vLLM allocated it. ``_kv_caches`` may
        # hold transposed views of the same storage (see ``as_kv_first``), and
        # ``set_device_layout_like`` reads the allocation, not the view.
        self._layout_ref: Optional[torch.Tensor] = None
        # Persistent device-side staging for the upload path, one flat buffer
        # per chunk region (:meth:`get_shapes` order: head groups, then aux
        # groups), grown on demand and viewed per call. A pre-laid-out device
        # buffer lets the upload run as one fast transfer plus an on-device
        # scatter, instead of a slow per-slot upload.
        self._staging: List[Optional[torch.Tensor]] = [None] * (
            len(self.kv_head_groups) + len(self.aux_groups)
        )
        # Persistent default-layout device buffers for bulk host downloads,
        # one per chunk region, grown on demand (:meth:`_download_view`).
        # Reusing them saves a bound-DRAM allocation per chunk, at the cost of
        # holding about one chunk of device memory for the connector's lifetime.
        self._download: List[Optional[torch.Tensor]] = [None] * len(self._staging)
        self._device: Optional[torch.device] = None
        self._layer_group_indices: Optional[List[int]] = None

    def _group_slots_by_block(
        self, slot_mapping: torch.Tensor
    ) -> dict[int, list[tuple[int, int]]]:
        """Group slot indices by their block index.

        Returns a dict mapping ``block_idx -> [(offset_in_block, token_idx)]``.
        Computed on CPU; ``slot_mapping`` is moved to CPU if needed.
        """
        if slot_mapping.device.type != "cpu":
            slot_mapping = slot_mapping.cpu()
        block_indices = slot_mapping // self.block_size
        offsets = slot_mapping % self.block_size

        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for token_idx, (blk, off) in enumerate(
            zip(block_indices.tolist(), offsets.tolist(), strict=True)
        ):
            groups[blk].append((off, token_idx))
        return groups

    def _full_blocks(self, slot_mapping: torch.Tensor) -> Optional[list[int]]:
        """Block ids when the chunk is distinct whole blocks, else ``None``.

        Decided with tensor ops: the per-token Python walk of
        :meth:`_group_slots_by_block` costs about as much as the device copy
        of a 2048-token chunk. A partial, repeated or reclaimed (negative)
        block returns ``None`` and takes the general path.
        """
        bs = self.block_size
        n = slot_mapping.numel()
        if n == 0 or n % bs:
            return None
        slots = slot_mapping.cpu().view(n // bs, bs)
        first = slots[:, :1]
        if (
            (first < 0).any()
            or (first % bs).any()
            or (slots != first + torch.arange(bs)).any()
        ):
            return None
        blocks = (first[:, 0] // bs).tolist()
        return blocks if len(set(blocks)) == len(blocks) else None

    def _cache_groups(self) -> list[List[torch.Tensor]]:
        """Per-layer caches of every chunk region: head groups, then aux."""
        return [
            [self._kv_caches[i] for i in layers] for layers in self._group_layers
        ] + self._aux_kv_caches

    def get_shapes(self, num_tokens: int) -> list[torch.Size]:
        """Per-group chunk layouts ``[2, L_g, H_g, T, D]`` (``[1, ...]`` for MLA),
        then ``[1, L_a, 1, T, W_a]`` per aux group, then ``[1, N_b, *block]``
        per block group.

        Under per-layer cache groups (HMA) each ``[H, T, D]`` layer plane holds
        its elements kernel-block-major (:func:`kernel_block_slabs`):
        axis 3 is the chunk token axis only for a layer whose kernel block
        covers the chunk."""
        return self._token_shapes(num_tokens) + [
            torch.Size([1, n, *shape]) for n, shape, _, _ in self.block_groups
        ]

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        slot_registration: bool = False,
    ) -> "RBLNConnector":
        """Create a connector from the RBLN engine metadata.

        ``metadata`` is the ``RBLNLMCacheMetadata`` that ``RBLNServiceFactory``
        builds: besides ``kv_shape`` it records the vLLM block size and the
        chunk's KV head, aux and block groups. ``use_gpu`` and ``device`` keep
        ``CreateGPUConnector``'s signature and are unused: the connector
        stages on the device of the KV caches vLLM registers.
        ``slot_registration`` registers CPU pools on the allocator's object
        grid, which only holds without unfull chunks.
        """
        block_size = getattr(metadata, "block_size", None)
        if block_size is None:
            raise ValueError(
                "RBLNConnector needs the vLLM block size in the engine metadata; "
                "configure vLLM with RBLNLMCacheConnectorV1, whose service "
                "factory records it"
            )
        kv_head_groups = list(getattr(metadata, "kv_head_groups", None) or [])
        kv_head_size = getattr(metadata, "kv_head_size", None)
        num_layers = metadata.kv_shape[0]
        num_kv_heads = metadata.kv_shape[3]
        head_dim = metadata.kv_shape[4]
        sized = sized_head_groups(kv_head_groups)
        if kv_head_size is not None or sized:
            # kv_shape only sizes the CPU budget here; the groups have the rest.
            num_layers = sum(g[0] for g in kv_head_groups)
            num_kv_heads = kv_head_groups[0][1] if kv_head_groups else 1
            head_dim = kv_head_groups[0][2] if sized else kv_head_size
        return cls(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=getattr(metadata, "kv_buffer_dtype", None) or metadata.kv_dtype,
            use_mla=metadata.use_mla,
            kv_head_groups=kv_head_groups,
            aux_groups=getattr(metadata, "aux_groups", None),
            slot_registration=slot_registration,
            block_groups=getattr(metadata, "block_groups", None),
        )

    def _token_shapes(self, num_tokens: int) -> list[torch.Size]:
        return self._head_shapes(num_tokens) + [
            torch.Size([1, n, 1, num_tokens, w]) for n, w, _ in self.aux_groups
        ]

    def _head_shapes(self, num_tokens: int) -> list[torch.Size]:
        return [
            torch.Size([self.kv_dim, n, h, num_tokens, d])
            for n, h, d in self._head_geometry
        ]

    @override
    def get_shape(self, num_tokens: int) -> torch.Size:
        """LMCache-side MemoryObj layout: ``[2, L, H, T, D]`` (``[1, ...]`` for MLA)."""
        shapes = self.get_shapes(num_tokens)
        if len(shapes) > 1:
            raise ValueError(
                f"chunk spans {len(shapes)} KV head groups; use get_shapes()"
            )
        return shapes[0]

    def register_kv_caches(
        self,
        ordered_kv_caches: List[torch.Tensor],
        layer_group_indices: Optional[List[int]] = None,
        aux_kv_caches: Optional[List[List[torch.Tensor]]] = None,
        block_kv_caches: Optional[List[List[tuple[torch.Tensor, int]]]] = None,
    ) -> None:
        """Register per-layer KV cache device tensors in layer order.

        Each tensor must be a 6-D RBLN attention cache -- ``[2, num_blocks,
        num_kv_heads, 1, block_size, head_dim]`` or the blocks-first
        ``[num_blocks, 2, ...]`` vLLM-RBLN allocates by default -- or
        ``[num_blocks, block_size, head_size]`` under MLA. Blocks-first
        tensors are stored as :func:`as_kv_first` views so the
        transfer paths index one layout; MLA caches are stored as-is.

        ``aux_kv_caches`` holds one layer-ordered list per ``aux_groups`` entry.
        A connector built with aux groups refuses to register without them:
        a chunk sized for the indexer state but never filled with it would
        restore zeros over it.

        ``block_kv_caches`` holds, per ``block_groups`` entry, each cache with
        the KV cache group whose block table addresses it. It needs
        ``layer_group_indices``: a block is read from the chunk's per-group
        slot row.
        """
        self._register_aux(aux_kv_caches or [])
        self._register_blocks(block_kv_caches or [])
        if self.block_groups and layer_group_indices is None:
            raise ValueError(
                "block groups need per-group slots: register with layer_group_indices"
            )
        if not ordered_kv_caches and self.block_groups and not self.num_layers:
            # Only block regions (a KDA-only model): nothing moves by token.
            self._device = self._block_caches[0][0][0].device
            self._layer_group_indices = layer_group_indices
            return
        if not ordered_kv_caches:
            raise ValueError("ordered_kv_caches must be non-empty")
        if len(ordered_kv_caches) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} KV caches, got {len(ordered_kv_caches)}"
            )

        first = ordered_kv_caches[0]
        if first.dtype != self.dtype:
            raise ValueError(
                f"KV cache dtype ({first.dtype}) does not match connector dtype "
                f"({self.dtype})"
            )
        expected_ndim = 3 if self.use_mla else 6
        if first.ndim != expected_ndim:
            raise ValueError(
                f"Expected {expected_ndim}-D KV caches "
                f"({'[NB, BS, HS]' if self.use_mla else '[2, NB, NH, 1, BS, HS]'}, "
                f"use_mla={self.use_mla}); got shape {tuple(first.shape)}"
            )

        kv_first = [as_kv_first(kv) for kv in ordered_kv_caches]
        # Group layers by their KV head count and head size in first-seen
        # order (what upstream's ``KVLayerGroupsManager`` does) and hold the
        # geometry to the one the chunk was sized for; a draft head that
        # disagrees with the target must fail here, not inside a stack of
        # unequal planes.
        head_dims = {d for _, _, d in self._head_geometry}
        by_heads: dict[tuple[int, int], list[int]] = {}
        for i, kv in enumerate(kv_first):
            if kv.shape[-1] not in head_dims:
                raise ValueError(
                    f"KV cache {i} has head_dim {kv.shape[-1]}, connector expects "
                    f"{' or '.join(map(str, sorted(head_dims)))}"
                )
            heads = 1 if self.use_mla else kv.shape[2]
            by_heads.setdefault((heads, kv.shape[-1]), []).append(i)
        actual = [(len(idx), h, d) for (h, d), idx in by_heads.items()]
        if actual != self._head_geometry:
            # In the connector's own terms: head sizes only where it has them.
            shown = [g if self._sized else g[:2] for g in actual]
            raise ValueError(
                f"KV caches form head groups {shown} (layers, kv_heads"
                f"{', head_dim' if self._sized else ''}); "
                f"connector was built for {self.kv_head_groups}"
            )
        self._group_layers = list(by_heads.values())

        self._kv_caches = kv_first
        self._download = [None] * len(self._download)
        self._layout_ref = first
        self._device = first.device
        if (
            layer_group_indices is not None
            and len(layer_group_indices) != self.num_layers
        ):
            raise ValueError("Every KV layer must have a cache group index")
        if layer_group_indices is not None and self.aux_groups:
            # The grouped (HMA) transfer walks the attention layers only; it
            # would leave the aux caches unrestored -- the DSA bug again.
            raise ValueError(
                "aux groups (indexer caches) are not supported with "
                "per-layer cache group block tables"
            )
        self._layer_group_indices = layer_group_indices
        logger.info(
            "Registered %d KV caches on %s, allocated shape=%s, indexed as=%s, "
            "dtype=%s",
            len(self._kv_caches),
            self._device,
            tuple(first.shape),
            tuple(self._kv_caches[0].shape),
            first.dtype,
        )

    def _register_aux(self, aux_kv_caches: List[List[torch.Tensor]]) -> None:
        if len(aux_kv_caches) != len(self.aux_groups):
            raise ValueError(
                f"connector was built for {len(self.aux_groups)} aux groups "
                f"{self.aux_groups}, got {len(aux_kv_caches)} aux cache lists"
            )
        registered: List[List[torch.Tensor]] = []
        for (n, w, dtype), caches in zip(self.aux_groups, aux_kv_caches, strict=True):
            if len(caches) != n:
                raise ValueError(f"aux group {(n, w, dtype)}: got {len(caches)} caches")
            views = []
            for kv in caches:
                # A [NB, BS] scale cache is read as [NB, BS, 1]; a view, no copy.
                kv3 = kv.unsqueeze(-1) if kv.ndim == 2 else kv
                if kv3.ndim != 3 or kv3.shape[-1] != w or kv.dtype != dtype:
                    raise ValueError(
                        f"aux group {(n, w, dtype)}: cache {tuple(kv.shape)} "
                        f"{kv.dtype} is not [NB, BS, {w}] {dtype}"
                    )
                views.append(kv3)
            registered.append(views)
        self._aux_kv_caches = registered

    def _register_blocks(
        self, block_kv_caches: List[List[tuple[torch.Tensor, int]]]
    ) -> None:
        if len(block_kv_caches) != len(self.block_groups):
            raise ValueError(
                f"connector was built for {len(self.block_groups)} block groups, "
                f"got {len(block_kv_caches)} block cache lists"
            )
        for (n, shape, dtype, _), caches in zip(
            self.block_groups, block_kv_caches, strict=True
        ):
            if len(caches) != n:
                raise ValueError(f"block group {shape}: got {len(caches)} caches")
            numel = math.prod(shape)
            for kv, _ in caches:
                if kv[0].numel() != numel or kv.dtype != dtype:
                    raise ValueError(
                        f"block group: cache {tuple(kv.shape)} {kv.dtype} is not "
                        f"[NB, *{shape}] {dtype}"
                    )
        self._block_caches = [list(caches) for caches in block_kv_caches]

    def _block_pairs(
        self,
        chunks: list[torch.Tensor],
        slots: torch.Tensor,
        loading: bool,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """``(chunk planes, cache blocks)`` of every block group.

        ``slots`` is the chunk's ``[T, groups]`` slot mapping; its first row
        gives each group's block, ``-1`` where the block-table entry is null.
        Caches may alias one allocation (a GLM-5.3 KDA state and the DSA
        latent share storage), so each copy addresses its own cache's block.
        """
        bs = self.block_size
        if slots.shape[0] != bs:
            raise ValueError(
                f"a chunk with block groups is one {bs}-token block, "
                f"got {slots.shape[0]} tokens"
            )
        row = slots[0].cpu().view(-1).tolist()
        planes: List[torch.Tensor] = []
        blocks: List[torch.Tensor] = []
        for (_, _, _, checkpoint), caches, chunk in zip(
            self.block_groups, self._block_caches, chunks, strict=True
        ):
            for plane, (cache, group) in enumerate(caches):
                slot = row[group]
                if slot < 0:
                    if loading and checkpoint:
                        continue
                    raise ValueError(f"block group cache {plane} has no block")
                if slot % bs:
                    raise ValueError(f"chunk starts at slot {slot}, not a block start")
                block = cache[slot // bs]
                # vllm-rbln registers a 1-D block as [NB, X, 1]; view the chunk
                # side, never the device cache.
                planes.append(chunk[0, plane].view(block.shape))
                blocks.append(block)
        return planes, blocks

    @staticmethod
    def _copy_per_dtype(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:
        """One ``torch._foreach_copy_`` per dtype.

        A single copy that mixes 1-byte and 2-byte tensors into one storage
        (an RDS chunk) took a slow runtime path: ~140 ms more per GLM-5.3
        chunk on store, gone when split by dtype.
        """
        by_dtype: dict[torch.dtype, tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        for dst, src in zip(dsts, srcs, strict=True):
            d, s = by_dtype.setdefault(dst.dtype, ([], []))
            d.append(dst)
            s.append(src)
        for d, s in by_dtype.values():
            torch._foreach_copy_(d, s)

    def _alloc_staging(
        self, numel: int, dtype: torch.dtype, like: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """A flat device staging buffer, laid out like ``like`` when one is given.

        ``torch.empty`` on the RBLN device is lazy: left alone the buffer stays
        host-side, and a copy from it into a device cache goes out as one
        host-to-device transfer per contiguous run of the destination.
        ``set_device_layout_like(buf, like)`` (no data copy) gives it ``like``'s
        layout and a device allocation instead, so the upload into it is one
        transfer and the copy from it into ``like`` a device-side one -- the
        fast path the API promises between the two. 0.6 ms vs 4.5 ms per K/V
        scatter, measured on Qwen3-1.7B.
        """
        buf = torch.empty(numel, dtype=dtype, device=self._device)
        if like is not None:
            # Third Party
            from torch_rbln.memory import set_device_layout_like

            set_device_layout_like(buf, like)
        return buf

    def _region_dtypes(self) -> list[torch.dtype]:
        """Element dtype of every token region, in :meth:`_token_shapes` order."""
        return [self.dtype] * len(self.kv_head_groups) + [
            dtype for _, _, dtype in self.aux_groups
        ]

    def _staging_views(self, num_tokens: int) -> list[torch.Tensor]:
        """One device staging view per chunk region, in :meth:`get_shapes` order.

        Each region keeps its own flat buffer, grown on demand, laid out by
        :meth:`_alloc_staging` like the cache it is scattered into: a head
        group like the K/V cache, an aux group like its own aux cache. The
        download does not gather into them: a run-by-run copy from the cache
        into a buffer with the cache's layout goes through the host, so
        downloads gather into default-layout buffers instead
        (:meth:`_download_view`).

        The K/V reference is ``_layout_ref`` rather than ``_kv_caches[0]``: the
        call requires a reference that covers its whole storage contiguously,
        and a blocks-first cache is registered as a transposed view that does
        not. Both name the same allocation, which is what it actually reads.

        Under MLA no region takes a layout. Matching the 3-D MLA latent cache
        turned the staging -> KV ``copy_`` into a host-shadow op on the cache:
        the runtime pulled the whole cache down from every chiplet and marked
        the host latest, so the next forward pushed it all back (the vmem
        sync-state trap ``rds_memory_allocator.allocate`` documents for
        ``fill_``/``zero_``). UMD showed four 1.3 GB ``rblnCmdCopyMemory``
        descriptors, one per chiplet, for a 288 KB write; the cost tracked
        cache size, not payload, and at 29.5 GB passed vLLM's 60 s step window.
        See rebellions-sw/lmcache-rbln#189. A 3-D indexer cache does not do
        that: matched to its own layout, the restore of a MiniMax-M3 indexer
        chunk into a 1 GiB indexer cache was one upload plus device-side
        copies in UMD, about 1 ms, byte-exact (UMD 3.5.1, RBLN-CR13; #261).
        """
        if self.use_mla:
            likes: list[Optional[torch.Tensor]] = [None] * len(self._staging)
        else:
            likes = [self._layout_ref] * len(self.kv_head_groups) + [
                caches[0] for caches in self._aux_kv_caches
            ]
        views: list[torch.Tensor] = []
        for i, (shape, dtype, like) in enumerate(
            zip(
                self._token_shapes(num_tokens),
                self._region_dtypes(),
                likes,
                strict=True,
            )
        ):
            buf = self._staging[i]
            if buf is None or buf.numel() < shape.numel():
                buf = self._alloc_staging(shape.numel(), dtype, like)
                self._staging[i] = buf
            views.append(buf[: shape.numel()].view(shape))
        return views

    def _chunk_views(
        self, memory_obj: MemoryObj, num_tokens: int
    ) -> list[torch.Tensor]:
        """Per-group ``[2, L_g, H_g, T, D]`` views over ``memory_obj``.

        ``MemoryObj.tensor`` views the whole object as group 0's shape, which
        only exists for a single group; a multi-group object is read region
        by region through ``get_tensor`` (as upstream's V3 connector does).
        """
        shapes = self.get_shapes(num_tokens)
        tensors = (
            [memory_obj.tensor]
            if len(shapes) == 1
            else [memory_obj.get_tensor(i) for i in range(len(shapes))]
        )
        views: list[torch.Tensor] = []
        for t, s in zip(tensors, shapes, strict=True):
            assert t is not None
            views.append(t.reshape(s))
        # Register the allocation, not each chunk view, on the device that
        # actually owns this connector's KV. CPU-only and resident paths are
        # unchanged; all gather/scatter variants below consume these views.
        allocation = self._pool_allocation(memory_obj)
        for view in views:
            self.register_host_buffer(view, allocation)
        return views

    def register_host_buffer(
        self, buffer: torch.Tensor, allocation: Optional[tuple[int, int]] = None
    ) -> bool:
        """Whether one registered host range covers ``buffer`` on this device.

        ``buffer`` is a host chunk about to be transferred to or from the KV
        caches on ``self._device``; ``allocation`` is its object's
        ``(address, aligned size)`` when the allocator keeps a fixed object
        grid (``slot_registration``). A covered buffer is gathered into with
        one 6-D block transfer; otherwise each block is copied directly.

        This connector registers nothing and returns False. lmcache-rbln
        replaces this method to register its huge-host CPU pools on the
        consuming device.
        """
        return False

    def _pool_allocation(self, memory_obj: MemoryObj) -> Optional[tuple[int, int]]:
        """``(address, aligned size)`` of the object's whole allocator slot."""
        raw = getattr(memory_obj, "raw_data", None)
        if not self.slot_registration or not isinstance(raw, torch.Tensor):
            return None
        return raw.data_ptr(), memory_obj.metadata.phy_size

    @override
    def initialize_kvcaches_ptr(self, **kwargs) -> None:
        """Hook used by upstream LMCache; the dict is registered separately."""
        ordered = kwargs.get("ordered_kv_caches")
        if ordered is not None:
            self.register_kv_caches(ordered)

    def _ensure_ready(self, op: str) -> None:
        if not self._kv_caches and not self._block_caches:
            raise RuntimeError(
                f"{op}: KV caches not registered — call register_kv_caches first"
            )

    # ------------------------------------------------------------------
    # D2H: NPU → CPU
    # ------------------------------------------------------------------

    def _grouped_pieces(
        self, chunks: list[torch.Tensor], slot_lists: list[list[int]]
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Pair cache pieces with chunk pieces for every grouped layer.

        Returns ``(cache_pieces, chunk_pieces)``: each pair is the same elements
        on both sides.

        A chunk plane is kernel-block-major (:func:`kernel_block_slabs`).
        A cache kernel block is ``[H, W, D]`` contiguous, so a whole block maps
        onto one chunk slab and moves as one copy: with GPT-OSS's 128-token SWA
        blocks a token-major chunk took 2592 head-sized copies, whose per-copy
        planning cost far more than the bytes. Runs never cross a chunk kernel
        block, so each lands in one slab; partial runs fall back to one copy per
        head.

        The copies are not sized against runtime limits: since
        rebel_compiler#15445 a D2D batch has no per-destination descriptor cap
        and a D2D copy may span up to 1024 physical mappings (a whole GPT-OSS
        full-layer block over the SWA-laid pool spans 64). Every piece here is
        D2D: the other side is a default-layout download buffer, a device
        upload staging buffer, or an RDS chunk. On an older runtime a
        destination past those caps falls back to a host round trip.
        """
        assert self._layer_group_indices is not None
        layer_chunks = {
            layer: (chunk, plane)
            for chunk, layers in zip(chunks, self._group_layers, strict=True)
            for plane, layer in enumerate(layers)
        }
        # Runs depend only on the group's slots and the kernel block size, so
        # every layer that shares the pair reuses one walk of the chunk.
        run_cache: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        cache_pieces: List[torch.Tensor] = []
        chunk_pieces: List[torch.Tensor] = []
        for layer, (cache, group) in enumerate(
            zip(self._kv_caches, self._layer_group_indices, strict=True)
        ):
            chunk, plane = layer_chunks[layer]
            # A 3-D MLA latent [NB, BS, W] keeps a token's row contiguous on
            # both sides, so a run is one copy.
            mla = cache.ndim == 3
            kernel = cache.shape[1] if mla else cache.shape[-2]
            key = (group, kernel)
            if key not in run_cache:
                run_cache[key] = slot_runs(slot_lists[group], kernel)
            if mla:
                for block, offset, token, stop in run_cache[key]:
                    cache_pieces.append(cache[block, offset : offset + stop - token])
                    chunk_pieces.append(chunk[0, plane, 0, token:stop])
                continue
            for half in range(self.kv_dim):
                # Views built once per plane, not per run: carving each slab
                # per run cost GPT-OSS ~3.5 ms per chunk for ~5 ms of copies.
                slabs = kernel_block_slabs(chunk[half, plane], kernel)
                blocks = cache[half, :, :, 0]
                for block, offset, token, stop in run_cache[key]:
                    slab = slabs[token // kernel]
                    if stop - token == kernel:
                        # A whole kernel block (offset 0, see slot_runs).
                        cache_pieces.append(blocks[block])
                        chunk_pieces.append(slab)
                        continue
                    # A partial run strides by head on both sides; one head is
                    # the unit contiguous on both.
                    start = token % kernel
                    cache_pieces.extend(
                        blocks[block, :, offset : offset + stop - token].unbind(0)
                    )
                    chunk_pieces.extend(slab[:, start : start + stop - token].unbind(0))
        return cache_pieces, chunk_pieces

    def _scatter_grouped(self, chunks: list[torch.Tensor], slots: torch.Tensor) -> None:
        """Restore HMA token runs using each layer's group and kernel block size.

        Slots are manager token addresses [tokens, groups]. Dividing by the
        actual cache's kernel block size also accounts for virtual splitting.
        Negative slots belong to reclaimed SWA blocks and are skipped on load.
        """
        caches, pieces = self._grouped_pieces(chunks, slots.cpu().T.tolist())
        # One batch across every layer: each call has a fixed submit cost, so
        # a call per layer dominated the restore. Full and SWA layers alias one
        # pool, which used to keep a batch per layer. One batch is safe:
        #   * the pieces are disjoint (each group owns its blocks of the pool),
        #     and torch-rbln checks a batch for overlap by byte range, not by
        #     storage, so it does not fall back to one copy at a time;
        #   * since rebel_compiler#15445 a D2D batch has no per-destination
        #     cap, so a batch into the shared pool stays on the device instead
        #     of completing through a host sync of the whole pool.
        # On that runtime the GPT-OSS e2e lanes check the forward after a
        # restore: warm output matches cold, logprob delta 0.
        if caches:
            torch._foreach_copy_(caches, pieces)

    def _download_view(self, index: int, shape: torch.Size) -> torch.Tensor:
        """Default-layout device memory for chunk region ``index``'s download.

        One flat buffer per region (a head group, then each aux group), grown
        on demand and reused by every later host download of that region. The
        view is therefore valid only until the next download of the region:
        copy it out first, as :meth:`from_gpu` does. It lives where the caches
        do.
        """
        buf = self._download[index]
        if buf is None or buf.numel() < shape.numel():
            device = self._kv_caches[0].device
            dtype = self._region_dtypes()[index]
            buf = torch.empty(shape.numel(), dtype=dtype, device=device)
            if device.type == "rbln":
                # Third Party
                from torch_rbln.memory import bind_device_memory

                # torch.empty alone is lazy host storage on RBLN.
                bind_device_memory(buf)
            self._download[index] = buf
        return buf[: shape.numel()].view(shape)

    def _gather_grouped(
        self, slots: torch.Tensor, chunks: Optional[list[torch.Tensor]] = None
    ) -> list[torch.Tensor]:
        """Gather every group's ``[2, L_g, H_g, T, D]`` chunk on device.

        Gather runs directly into each contiguous K/V layer plane, avoiding
        an intermediate layer stack. Resident destinations supply ``chunks``.
        Host destinations get views of the reused download buffers
        (:meth:`_download_view`), which the next download overwrites: copy
        them out before gathering again. Cache-layout staging can pull whole
        compiled caches through the host, so these buffers deliberately use
        the default device layout.
        """
        assert self._layer_group_indices is not None
        host_slots = slots.cpu()
        used = sorted(set(self._layer_group_indices))
        if (host_slots[:, used] < 0).any().item():
            raise ValueError("Cannot store KV from reclaimed sliding-window blocks")
        if chunks is None:
            chunks = [
                self._download_view(i, shape)
                for i, shape in enumerate(self._token_shapes(len(host_slots)))
            ]
        slot_lists = host_slots.T.tolist()
        # Every piece goes into one batched copy: a call per layer paid the
        # fixed submit cost 72 times (~11 ms for a 256-token GPT-OSS chunk) for
        # well under 1 ms of data. The destination is a default-layout buffer
        # (download staging, or an RDS chunk).
        caches, pieces = self._grouped_pieces(chunks, slot_lists)
        if caches:
            torch._foreach_copy_(pieces, caches)
        return chunks

    def _copy_out(self, dst: torch.Tensor, gathered: torch.Tensor) -> None:
        """Move one gathered group into ``dst`` (D2H, or D2D when resident)."""
        if split_kv_copy_enabled() and dst.shape[0] == 2:
            # A/B valve for the RBLN v2v half-corruption: write K and V as
            # two separate copies instead of one [2, ...] copy. Only a group
            # with K and V halves has two; the MLA latent and an aux group
            # (an indexer key next to GQA K/V included) have one (#256).
            dst[0].copy_(gathered[0])
            dst[1].copy_(gathered[1])
        else:
            dst.copy_(gathered)

    @override
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        """Copy KV cache from RBLN NPU into ``memory_obj`` (D2H or D2D).

        ``memory_obj.tensor`` may live on CPU (default LMCache memory pool)
        or on the RBLN device (when the P/D buffer is device-resident).

        MLA token runs are copied directly into the destination, including aux
        groups, using batched D2H or D2D copies without device staging. Grouped
        resident destinations receive the gather directly; grouped host
        destinations use one trailing transfer per group.

        Args:
            memory_obj: Destination memory object.
            start: Start token index within the sequence.
            end: End token index within the sequence.
            **kwargs: Must include ``slot_mapping`` (torch.Tensor).
        """
        slot_mapping = kwargs.get("slot_mapping")
        assert slot_mapping is not None, (
            "slot_mapping is required for RBLN D2H transfer"
        )
        self._ensure_ready("D2H")

        num_tokens = end - start
        dsts = self._chunk_views(memory_obj, num_tokens)
        n_token = len(dsts) - len(self.block_groups)
        dsts, block_dsts = dsts[:n_token], dsts[n_token:]

        chunk_slots = slot_mapping[start:end]
        if self.block_groups:
            planes, blocks = self._block_pairs(block_dsts, chunk_slots, loading=False)
            self._copy_per_dtype(planes, blocks)
        if not dsts:
            return
        dst_device = dsts[0].device
        if self._layer_group_indices is not None:
            resident = dst_device == self._device
            gathered = self._gather_grouped(chunk_slots, dsts if resident else None)
            if not resident:
                for dst, chunk in zip(dsts, gathered, strict=True):
                    self._copy_out(dst, chunk)
            return
        if self.use_mla:
            full_blocks = self._full_blocks(chunk_slots)
            if full_blocks is not None:
                bs = self.block_size
                runs = [(b, 0, i * bs, (i + 1) * bs) for i, b in enumerate(full_blocks)]
            else:
                host_slots = chunk_slots.cpu().tolist()
                if any(slot < 0 for slot in host_slots):
                    raise ValueError(
                        "Cannot store KV from reclaimed sliding-window blocks"
                    )
                runs = slot_runs(host_slots, self.block_size)
            # One submit for every group: each costs ~0.2 ms whatever its size.
            run_dsts: List[torch.Tensor] = []
            run_srcs: List[torch.Tensor] = []
            for caches, dst in zip(self._cache_groups(), dsts, strict=True):
                d, s = self._mla_run_pairs(caches, runs, dst)
                run_dsts += d
                run_srcs += s
            if run_dsts:
                torch._foreach_copy_(run_dsts, run_srcs)
            return
        block_groups = self._group_slots_by_block(chunk_slots)
        logger.debug(
            "[D2H] tokens=%d→%d (%d) blocks=%d dst=%s",
            start,
            end,
            num_tokens,
            len(block_groups),
            dst_device,
        )

        all_full = all(
            [o for (o, _) in slots] == list(range(self.block_size))
            for slots in block_groups.values()
        )
        groups = [[self._kv_caches[i] for i in layers] for layers in self._group_layers]
        for region, (caches, dst) in enumerate(
            zip(groups + self._aux_kv_caches, dsts, strict=True)
        ):
            self._gather_group(caches, block_groups, all_full, dst, region)

    @staticmethod
    def _mla_run_pairs(
        kv_caches: List[torch.Tensor],
        runs: list[tuple[int, int, int, int]],
        dst: torch.Tensor,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """``(chunk view, cache view)`` pairs that gather MLA or aux rows into
        a host or resident chunk.

        Runs retain chunk token positions, including interleaved blocks and
        scattered offsets. No intermediate device allocation or cache layout
        matching is needed.
        """
        dsts: List[torch.Tensor] = []
        srcs: List[torch.Tensor] = []
        for block, offset, token, stop in runs:
            for layer, cache in enumerate(kv_caches):
                dsts.append(dst[0, layer, 0, token:stop])
                srcs.append(cache[block, offset : offset + stop - token])
        return dsts, srcs

    def _gather_group(
        self,
        kv_caches: List[torch.Tensor],
        block_groups: dict[int, list[tuple[int, int]]],
        all_full: bool,
        dst: torch.Tensor,
        region: int,
    ) -> None:
        """Gather ``block_groups`` from one group's layers into chunk ``region``."""
        mla = kv_caches[0].ndim == 3  # MLA latent or an aux cache
        # Gather requested slots per block:
        #   * contiguous (full block = [0:block_size]): slice [s0:e0] before
        #     stacking — moves only those slots, no index_select.
        #   * scattered: stack the whole block, then index_select.
        # Gather a single 6-D block for a registered host pool to avoid small,
        # synchronously batched D2Hs. Unregistered buffers keep direct copies:
        # a chunk-sized bounce makes the gathered host transfer more expensive.
        gather_for_host = (
            all_full
            and not mla
            and len(block_groups) == 1
            and self.register_host_buffer(dst)
        )
        if all_full and not gather_for_host:
            gather_full_blocks_headmajor(kv_caches, list(block_groups), dst)
        elif mla:
            # Rows go run by run straight into a resident chunk. A host chunk
            # gets them in the reused download buffer first and then one bulk
            # download, so only that one intermediate is left of the per-block
            # stack and the cat that used to precede the download.
            target = dst
            if dst.device != kv_caches[0].device:
                target = self._download_view(region, dst.shape)
            cache_rows, chunk_rows = self._row_pairs(kv_caches, block_groups, target)
            torch._foreach_copy_(chunk_rows, cache_rows)
            if target is not dst:
                self._copy_out(dst, target)
        else:
            pieces: List[torch.Tensor] = []
            for block_idx, slot_list in block_groups.items():
                n_slots = len(slot_list)
                offs = [s[0] for s in slot_list]
                # Contiguous offsets slice; scattered ones index_select.
                sel: Union[slice, torch.Tensor]
                if offs == list(range(offs[0], offs[0] + n_slots)):
                    sel = slice(offs[0], offs[0] + n_slots)
                else:
                    sel = torch.tensor(offs, dtype=torch.long, device=self._device)
                # kv: [2, NB, NH, 1, BS, HD]. Keeping the K/V axis (dim0=2)
                # in the per-layer view means one stack over layers yields
                # [2, L, H, slots, D] directly — no separate k/v stacks and
                # no trailing stack([k, v]) recombine, so the gather is a
                # single v2v_multi submit (the D2H below is the only other).
                planes = [
                    kv[:, block_idx, :, 0, sel, :]
                    if isinstance(sel, slice)
                    else kv[:, block_idx, :, 0, :, :].index_select(2, sel)
                    for kv in kv_caches
                ]
                out = None
                if gather_for_host:
                    out = torch.empty(dst.shape, dtype=dst.dtype, device=self._device)
                    if out.device.type == "rbln":
                        # Third Party
                        from torch_rbln.memory import bind_device_memory

                        # Lazy V2V allocation inherits the source layout
                        # when storage sizes match. Bind flat storage first
                        # so the gathered block stays one host transfer.
                        bind_device_memory(out)
                pieces.append(torch.stack(planes, dim=1, out=out))
            # Cat blocks along the token axis (dim=3 of [2, L, H, T, D]).
            gathered = torch.cat(pieces, dim=3) if len(pieces) > 1 else pieces[0]
            self._copy_out(dst, gathered)

    def _row_pairs(
        self,
        kv_caches: List[torch.Tensor],
        block_groups: dict[int, list[tuple[int, int]]],
        chunk: torch.Tensor,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Cache rows paired with ``[1, L, 1, T, W]`` chunk rows of a 3-D group.

        An MLA latent or aux cache ``[NB, BS, W]`` keeps each token's row
        contiguous, partial blocks included, so every run of consecutive
        offsets in a block is one copy per layer, in either direction.
        """
        cache_rows: List[torch.Tensor] = []
        chunk_rows: List[torch.Tensor] = []
        block_size = kv_caches[0].shape[1]
        token = 0
        for block_idx, slot_list in block_groups.items():
            offsets = [off for off, _ in slot_list]
            for _, offset, begin, stop in slot_runs(offsets, block_size):
                for layer_idx, kv in enumerate(kv_caches):
                    cache_rows.append(kv[block_idx, offset : offset + stop - begin])
                    chunk_rows.append(
                        chunk[0, layer_idx, 0, token + begin : token + stop]
                    )
            token += len(slot_list)
        return cache_rows, chunk_rows

    # ------------------------------------------------------------------
    # H2D: CPU → NPU
    # ------------------------------------------------------------------

    @override
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        """Copy KV cache from ``memory_obj`` into the RBLN NPU (H2D or D2D).

        ``memory_obj.tensor`` may live on CPU (default LMCache memory pool)
        or on the RBLN device (when the P/D buffer is device-resident).
        Moves only chunk-sized data (not whole blocks). CPU sources for MLA
        caches are copied directly into the requested KV slots, including aux
        caches. Other cross-device sources are uploaded once to device staging
        buffers before scattering. Grouped (HMA) chunks and whole-block chunks
        go out in one ``torch._foreach_copy_``; 3-D groups of a partial chunk
        take one per group, and other partial chunks one per block and group.

        Args:
            memory_obj: Source memory object.
            start: Start token index within the sequence.
            end: End token index within the sequence.
            **kwargs: Must include ``slot_mapping`` (torch.Tensor).
        """
        slot_mapping = kwargs.get("slot_mapping")
        assert slot_mapping is not None, (
            "slot_mapping is required for RBLN H2D transfer"
        )
        self._ensure_ready("H2D")

        num_tokens = end - start
        srcs = self._chunk_views(memory_obj, num_tokens)
        n_token = len(srcs) - len(self.block_groups)
        srcs, block_planes = srcs[:n_token], srcs[n_token:]

        chunk_slots = slot_mapping[start:end]
        if self.block_groups:
            planes, blocks = self._block_pairs(block_planes, chunk_slots, loading=True)
            if blocks:
                self._copy_per_dtype(blocks, planes)
        if not srcs:
            return

        # MLA CPU sources use batched row-range H2D copies directly into caches.
        # Other cross-device sources use one staging upload per cache group.
        if self.use_mla and srcs[0].device.type == "cpu":
            src_devs = srcs
        elif srcs[0].device != self._device:
            src_devs = self._staging_views(num_tokens)
            for src, src_dev in zip(srcs, src_devs, strict=True):
                src_dev.copy_(src)  # upload the group in one transfer
        else:
            src_devs = srcs

        if self._layer_group_indices is not None:
            self._scatter_grouped(src_devs, chunk_slots)
            return
        full_blocks = self._full_blocks(chunk_slots)
        if full_blocks is not None:
            # One submit for every head and aux group.
            block_dsts: List[torch.Tensor] = []
            block_srcs: List[torch.Tensor] = []
            for caches, src_dev in zip(self._cache_groups(), src_devs, strict=True):
                d, s = full_block_scatter_pairs(caches, full_blocks, src_dev)
                block_dsts += d
                block_srcs += s
            if block_dsts:
                torch._foreach_copy_(block_dsts, block_srcs)
            return
        block_groups = self._group_slots_by_block(chunk_slots)
        logger.debug(
            "[H2D] tokens=%d→%d (%d) blocks=%d src=%s",
            start,
            end,
            num_tokens,
            len(block_groups),
            srcs[0].device,
        )

        all_full = all(
            [o for (o, _) in slots] == list(range(self.block_size))
            for slots in block_groups.values()
        )
        n_head = len(self._group_layers)
        for layers, src_dev in zip(self._group_layers, src_devs[:n_head], strict=True):
            self._scatter_group(
                [self._kv_caches[i] for i in layers], block_groups, all_full, src_dev
            )
        for caches, src_dev in zip(self._aux_kv_caches, src_devs[n_head:], strict=True):
            self._scatter_group(caches, block_groups, all_full, src_dev)

    def _scatter_group(
        self,
        kv_caches: List[torch.Tensor],
        block_groups: dict[int, list[tuple[int, int]]],
        all_full: bool,
        src: torch.Tensor,
    ) -> None:
        """Scatter a host or resident chunk into one group's layers."""
        # Use src rather than src_dev: CPU MLA restore bypasses device staging,
        # so the source is no longer guaranteed to be device resident.
        mla = kv_caches[0].ndim == 3  # MLA latent or an aux cache
        # Scatter into KV slots, batched into one _foreach_copy_ per block:
        #   * contiguous (full block = [0:block_size]): one [s0:e0] slot-slice
        #     view per layer.
        #   * scattered: one per-slot integer-index view (basic index →
        #     writable view; fancy index would return a copy).
        # Fast path: every group is a whole block — use the head-major kernel
        # above. Partial/scattered slots stay inline below; 3-D groups
        # move run by run, like their gather.
        if all_full:
            scatter_headmajor_to_full_blocks(kv_caches, list(block_groups), src)
        elif mla:
            cache_rows, chunk_rows = self._row_pairs(kv_caches, block_groups, src)
            torch._foreach_copy_(cache_rows, chunk_rows)
        else:
            tok_offset = 0
            for block_idx, slots in block_groups.items():
                n_slots = len(slots)
                offs = [s[0] for s in slots]
                src_chunk = src[:, :, :, tok_offset : tok_offset + n_slots, :]
                dsts: List[torch.Tensor] = []
                srcs: List[torch.Tensor] = []
                # kv: [2, NB, NH, 1, BS, HD]
                if offs == list(range(offs[0], offs[0] + n_slots)):
                    s0, e0 = offs[0], offs[0] + n_slots
                    for layer_idx, kv_cache in enumerate(kv_caches):
                        for half in (0, 1):
                            dsts.append(kv_cache[half, block_idx, :, 0, s0:e0])
                            srcs.append(src_chunk[half, layer_idx])
                else:
                    for layer_idx, kv_cache in enumerate(kv_caches):
                        for k, off in enumerate(offs):
                            for half in (0, 1):
                                dsts.append(kv_cache[half, block_idx, :, 0, off])
                                srcs.append(src_chunk[half, layer_idx, :, k])
                torch._foreach_copy_(dsts, srcs)
                tok_offset += n_slots

    # ------------------------------------------------------------------
    # Batched operations
    # ------------------------------------------------------------------

    @override
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ) -> None:
        """Batched D2H transfer."""
        for memory_obj, s, e in zip(memory_objs, starts, ends, strict=True):
            self.from_gpu(memory_obj, s, e, **kwargs)

    @override
    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Batched H2D transfer."""
        if memory_objs is None or starts is None or ends is None:
            return
        for memory_obj, s, e in zip(memory_objs, starts, ends, strict=True):
            self.to_gpu(memory_obj, s, e, **kwargs)

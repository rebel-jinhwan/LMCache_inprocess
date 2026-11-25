# SPDX-License-Identifier: Apache-2.0
"""Chunk layout of a worker's KV caches, read from vLLM's per-cache specs.

A chunk holds token regions, then block regions. Each cache's spec picks its
kind; no layer index, config key or cache name fragment takes part.

A token region keeps one row per token, so a chunk of any length holds a
token run of it:

- A KV region is the main attention cache: K/V caches
  (``FullAttentionSpec``, ``SlidingWindowSpec``) of one head count and head
  size, or, in a model without K/V caches, the MLA family with the largest
  page.
- An aux region is every other one-row-per-token ``MLAAttentionSpec`` of one
  head size and dtype (an indexer key, its scale). Aux regions are ranked by
  page, largest first.

A block region has no row per token, so a chunk is one block of it:

- An ``RBLNMambaSpec`` group keeps a block only at a block end a request may
  resume from in ``"align"`` mode, and one running state a request in
  ``"none"`` mode (prefix caching off). Either is a checkpoint region.
- An ``MLAAttentionSpec`` whose state covers several tokens.
- A vllm-rbln block-aux cache, with its owner's block ids. One whose
  ``prefix_cacheable`` is false is registered but not stored.

Caches of one kind, block shape and dtype form one block region. Paged regions
come first, then checkpoint ones, each ranked by page, largest first.

Within a region caches follow layer order where every token cache name has a
single layer index, as trunk built it, and vLLM's group order otherwise.

Only a worker can build this: the scheduler's ``KVCacheConfig`` collapses a
group of mixed caches to one layer's spec.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TypeGuard
import math

# Third Party
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
import torch

try:
    # Third Party
    from vllm_rbln.v1.kv_cache import RBLNMambaSpec

    _CHECKPOINT_SPECS: tuple[type[MambaSpec], ...] = (RBLNMambaSpec,)
except ImportError:  # vllm-rbln before #1475 builds no checkpoint spec placed here
    _CHECKPOINT_SPECS = ()

# Exact types: a subclass may change what a block holds. vllm-rbln's
# RBLNSlidingWindowSpec subclasses SlidingWindowSpec but gives a request one
# block whose contents slide in place, while still reporting
# prefix_cacheable=True. A checkpoint group is placed only as RBLNMambaSpec,
# what vllm-rbln routes to RBLNMambaManager; today only GLM-5.3's KDA state, the
# one verified layout. An upstream MambaSpec is refused.
_KV_SPECS: tuple[type[AttentionSpec], ...] = (FullAttentionSpec, SlidingWindowSpec)
_SUPPORTED_SPECS: tuple[type[KVCacheSpec], ...] = (
    *_KV_SPECS,
    MLAAttentionSpec,
    *_CHECKPOINT_SPECS,
)


def is_checkpoint_spec(spec: KVCacheSpec) -> TypeGuard[MambaSpec]:
    """Whether ``spec`` is a checkpoint group's, as this module places it."""
    return type(spec) in _CHECKPOINT_SPECS


@dataclass(frozen=True)
class KVRegion:
    """Every main KV cache with ``num_kv_heads`` heads of ``head_size``, in
    layer order."""

    num_kv_heads: int
    dtype: torch.dtype
    layer_names: tuple[str, ...]
    # As vLLM allocated it; the config's head size can differ (GLM-5.3's
    # 512-byte latent against 640).
    head_size: int | None = None


@dataclass(frozen=True)
class AuxRegion:
    """One aux family across layers, stored as a ``[1, L, T, width]`` region."""

    width: int
    dtype: torch.dtype
    layer_names: tuple[str, ...]


class BlockAuxSpec(Protocol):
    """vllm-rbln's ``RBLNBlockAuxSpec``: a per-block cache outside every group.

    It uses its owner's block ids. ``prefix_cacheable=False`` means a prefix
    resumed at a block boundary never reads it, so it is not stored.
    """

    @property
    def owner(self) -> str: ...

    @property
    def block_shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def prefix_cacheable(self) -> bool: ...


@dataclass(frozen=True)
class BlockCache:
    name: str
    # KV cache group whose block table addresses this cache (its owner's, for
    # a block-aux cache).
    group: int


@dataclass(frozen=True)
class BlockRegion:
    """One block of each cache, stored as a ``[1, len(caches), *block_shape]``
    region in the caches' dtype."""

    caches: tuple[BlockCache, ...]
    block_shape: tuple[int, ...]
    dtype: torch.dtype
    # A checkpoint group keeps a block only where a request may resume; its
    # other block-table entries are null.
    checkpoint: bool

    @property
    def chunk_shape(self) -> torch.Size:
        return torch.Size([1, len(self.caches), *self.block_shape])


@dataclass(frozen=True)
class CacheLayout:
    kv_regions: tuple[KVRegion, ...]
    aux_regions: tuple[AuxRegion, ...]
    block_regions: tuple[BlockRegion, ...] = ()
    unstored: tuple[str, ...] = ()

    @property
    def kv_head_groups(self) -> list[tuple[int, ...]]:
        """``(num_layers, num_kv_heads)``, as ``RBLNLMCacheMetadata`` takes it.

        With KV regions of several head sizes (Gemma4: 256 sliding, 512 full)
        each entry also carries its region's ``head_size``.
        """
        if len({r.head_size for r in self.kv_regions}) > 1:
            return [
                (len(r.layer_names), r.num_kv_heads, r.head_size or 0)
                for r in self.kv_regions
            ]
        return [(len(r.layer_names), r.num_kv_heads) for r in self.kv_regions]

    @property
    def aux_groups(self) -> list[tuple[int, int, torch.dtype]]:
        """``(num_layers, width, dtype)``, as ``RBLNLMCacheMetadata`` takes it."""
        return [(len(r.layer_names), r.width, r.dtype) for r in self.aux_regions]

    @property
    def block_groups(self) -> list[tuple[int, tuple[int, ...], torch.dtype, bool]]:
        """``(num_caches, block_shape, dtype, checkpoint)`` per block region."""
        return [
            (len(r.caches), r.block_shape, r.dtype, r.checkpoint)
            for r in self.block_regions
        ]

    @property
    def cache_names(self) -> frozenset[str]:
        """Every cache vLLM registers, stored or not."""
        names = [n for r in self.kv_regions for n in r.layer_names]
        names += [n for r in self.aux_regions for n in r.layer_names]
        names += [c.name for r in self.block_regions for c in r.caches]
        return frozenset(names).union(self.unstored)

    @classmethod
    def from_kv_cache_config(
        cls,
        kv_cache_config: KVCacheConfig,
        aux_specs: Mapping[str, BlockAuxSpec] | None = None,
    ) -> "CacheLayout":
        """Raises ``ValueError`` naming the cache it cannot place."""
        token: list[tuple[str, AttentionSpec]] = []
        group_of: dict[str, int] = {}
        position: dict[str, int] = {}
        block_key: dict[str, tuple[Any, ...]] = {}
        blocks: dict[tuple[Any, ...], list[BlockCache]] = {}
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            for name, spec in _group_specs(group).items():
                if type(spec) not in _SUPPORTED_SPECS:
                    raise ValueError(
                        f"{name} in KV cache group {group_id} is a "
                        f"{type(spec).__name__}; CacheLayout places only "
                        f"{', '.join(t.__name__ for t in _SUPPORTED_SPECS)}"
                    )
                group_of[name] = group_id
                position[name] = len(position)
                block = _block_of(name, group_id, spec)
                if block is None:
                    assert isinstance(spec, AttentionSpec)
                    token.append((name, spec))
                    continue
                block_key[name] = block
                blocks.setdefault(block, []).append(BlockCache(name, group_id))

        order = _layer_order(token, position)
        token.sort(key=lambda c: order[c[0]])
        kv_regions, aux_regions = _token_regions(token)
        block_regions, unstored = _block_regions(
            blocks, block_key, group_of, position, aux_specs or {}
        )
        if block_regions and not kv_regions:
            # A KDA-only model (GLM-5.3 cut to its first layers) has no token
            # region; LMCache then reads the chunk's tokens from the first
            # block region's rows, so they must be one per token.
            specs = [g.kv_cache_spec for g in kv_cache_config.kv_cache_groups]
            if any(getattr(s, "mamba_cache_mode", None) == "none" for s in specs):
                # A running state's block spans max_model_len, not a chunk.
                raise ValueError(
                    "no cache keeps one row per token, and with prefix caching "
                    "off no block gives a chunk's token count"
                )
            block_sizes = {s.block_size for s in specs}
            first = block_regions[0]
            if block_sizes != {first.block_shape[0]}:
                raise ValueError(
                    f"no cache keeps one row per token, and the first block "
                    f"region {first.block_shape} has no row per token of a "
                    f"{sorted(block_sizes)}-token block"
                )
        return cls(kv_regions, aux_regions, block_regions, unstored)


def _group_specs(group: Any) -> dict[str, KVCacheSpec]:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return dict(spec.kv_cache_specs)
    return dict.fromkeys(group.layer_names, spec)


def _block_of(
    name: str, group_id: int, spec: KVCacheSpec
) -> tuple[bool, tuple[int, ...], torch.dtype] | None:
    """``(checkpoint, block shape, dtype)`` of a block-region cache, or
    ``None`` for a cache with one row per token."""
    where = f"{name} in KV cache group {group_id}"
    if is_checkpoint_spec(spec):
        if spec.mamba_cache_mode not in ("align", "none"):
            raise ValueError(
                f"{where} is a {type(spec).__name__} in mamba_cache_mode "
                f"{spec.mamba_cache_mode!r}; a block region needs 'align', where "
                "a block holds the state at its last token, or 'none', where "
                "a request's block holds its running state"
            )
        if len(spec.shapes) != 1 or spec.page_size_bytes != (
            math.prod(spec.shapes[0]) * spec.dtypes[0].itemsize
        ):
            raise ValueError(
                f"{where} holds {len(spec.shapes)} state tensors in a "
                f"{spec.page_size_bytes}-byte page; a block region moves one"
            )
        return True, tuple(spec.shapes[0]), spec.dtypes[0]
    if type(spec) is MLAAttentionSpec and spec.tokens_per_state > 1:
        rows, rem = divmod(spec.block_size, spec.tokens_per_state)
        block_shape = (rows, spec.head_size)
        if (
            rem
            or spec.num_kv_heads != 1
            or spec.page_size_bytes != (math.prod(block_shape) * spec.dtype.itemsize)
        ):
            raise ValueError(
                f"{where}: a {spec.page_size_bytes}-byte page is not "
                f"{block_shape} {spec.dtype}"
            )
        return False, block_shape, spec.dtype
    return None


def _layer_order(
    token: list[tuple[str, AttentionSpec]], position: Mapping[str, int]
) -> dict[str, tuple[int, int]]:
    """Layer index order when every token cache has one, else group order.

    The layer index only orders caches, keeping the chunk trunk shipped; it
    places none of them.
    """
    try:
        return {name: (extract_layer_index(name), position[name]) for name, _ in token}
    except AssertionError:
        return {name: (0, position[name]) for name, _ in token}


def _token_regions(
    token: list[tuple[str, AttentionSpec]],
) -> tuple[tuple[KVRegion, ...], tuple[AuxRegion, ...]]:
    kv: dict[tuple[int, int], list[tuple[str, AttentionSpec]]] = {}
    mla: dict[tuple[int, torch.dtype], list[tuple[str, AttentionSpec]]] = {}
    for name, spec in token:
        if type(spec) in _KV_SPECS:
            key = (spec.num_kv_heads, spec.head_size)
            kv.setdefault(key, []).append((name, spec))
        else:
            mla.setdefault((spec.head_size, spec.dtype), []).append((name, spec))

    by_page = sorted(
        mla.items(),
        key=lambda f: (-f[1][0][1].page_size_bytes, f[0][0], str(f[0][1])),
    )
    if not kv and by_page:
        # A model without K/V caches keeps its main KV as the MLA latent, the
        # widest per-token row.
        if len(by_page) > 1:
            (_, a), (_, b) = by_page[0], by_page[1]
            if a[0][1].page_size_bytes == b[0][1].page_size_bytes:
                raise ValueError(
                    f"{a[0][0]} and {b[0][0]} have equal pages "
                    f"({a[0][1].page_size_bytes} bytes); cannot tell the latent"
                )
        (_, latent), by_page = by_page[0], by_page[1:]
        kv[(1, latent[0][1].head_size)] = latent

    kv_regions = tuple(_kv_region(h, caches) for (h, _), caches in kv.items())
    aux_regions = tuple(
        _aux_region(rank, caches) for rank, (_, caches) in enumerate(by_page)
    )
    return kv_regions, aux_regions


def _kv_region(num_kv_heads: int, caches: list[tuple[str, AttentionSpec]]) -> KVRegion:
    first_name, first = caches[0]
    for name, spec in caches:
        if spec.dtype != first.dtype:
            raise ValueError(
                f"KV region of {num_kv_heads} heads mixes {first.dtype} at "
                f"{first_name} with {spec.dtype} at {name}"
            )
    return KVRegion(
        num_kv_heads=num_kv_heads,
        dtype=first.dtype,
        layer_names=tuple(name for name, _ in caches),
        head_size=first.head_size,
    )


def _aux_region(rank: int, caches: list[tuple[str, AttentionSpec]]) -> AuxRegion:
    first_name, first = caches[0]
    for name, spec in caches:
        row_bytes = spec.head_size * spec.dtype.itemsize
        if spec.page_size_bytes != spec.block_size * row_bytes:
            raise ValueError(
                f"{name} (aux rank {rank}) holds {spec.page_size_bytes} bytes per "
                f"block, not one {spec.head_size}-wide {spec.dtype} row per token"
            )
    return AuxRegion(
        width=first.head_size,
        dtype=first.dtype,
        layer_names=tuple(name for name, _ in caches),
    )


def _block_regions(
    blocks: dict[tuple[Any, ...], list[BlockCache]],
    block_key: Mapping[str, tuple[Any, ...]],
    group_of: Mapping[str, int],
    position: Mapping[str, int],
    aux_specs: Mapping[str, BlockAuxSpec],
) -> tuple[tuple[BlockRegion, ...], tuple[str, ...]]:
    """Grouped block caches, then each stored block-aux cache after its
    owner's region, in its owners' order."""
    aux: dict[tuple[Any, ...], dict[tuple[Any, ...], list[BlockCache]]] = {}
    unstored = []
    for name, spec in sorted(
        aux_specs.items(), key=lambda a: position.get(a[1].owner, -1)
    ):
        if name in group_of:
            raise ValueError(f"{name} is both in a KV cache group and block-aux")
        if spec.owner not in group_of:
            raise ValueError(
                f"block-aux {name}: owner {spec.owner} is in no KV cache group"
            )
        if not spec.prefix_cacheable:
            unstored.append(name)
            continue
        # A block-aux cache of a token cache is still one block per chunk.
        owner_key = block_key.get(spec.owner, (False, (), None))
        aux_key = (tuple(spec.block_shape), spec.dtype)
        aux.setdefault(owner_key, {}).setdefault(aux_key, []).append(
            BlockCache(name, group_of[spec.owner])
        )

    def page(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        return math.prod(shape) * dtype.itemsize

    regions = []
    owners = set(blocks) | set(aux)
    for key in sorted(
        owners, key=lambda k: (k[0], -page(k[1], k[2]) if k[2] is not None else 0)
    ):
        checkpoint, block_shape, dtype = key
        if key in blocks:
            regions.append(
                BlockRegion(tuple(blocks[key]), block_shape, dtype, checkpoint)
            )
        by_page = sorted(aux.get(key, {}).items(), key=lambda a: -page(*a[0]))
        for (aux_shape, aux_dtype), caches in by_page:
            regions.append(BlockRegion(tuple(caches), aux_shape, aux_dtype, checkpoint))
    return tuple(regions), tuple(sorted(unstored))

# SPDX-License-Identifier: Apache-2.0
"""Runtime patches the RBLN in-process connector applies to LMCache.

``_allocate_cpu_memory`` / ``_free_cpu_memory`` own huge-host CPU pools. The
connector registers each pool on its consuming device before use; free
unregisters it before releasing the host allocation.

``LMCacheEngine._broadcast_or_receive_memory_objs`` is patched so that a
single-rank MLA engine restores from the host chunks instead of from device
copies made for a TP broadcast nobody receives
(see :func:`_patched_broadcast_or_receive_memory_objs`).

LMCache's shared-memory pinning is routed through torch-rbln's managed host
registration (:mod:`lmcache.inprocess.rbln._shm_registration`).

The RDS storage plugin, shipped separately by ``lmcache-rbln``, applies its
own patches (allocator selection and staging-bounded store/retrieve) when
LMCache loads it.
"""

# Standard
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

# First Party
from lmcache.inprocess.rbln._host_memory import (
    allocate_cpu_pool,
    host_register_enabled,
    release_cpu_pool,
)
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.memory_management import NUMAMapping

logger = init_logger(__name__)

_applied = False


def _patch_cpu_slab_huge_host() -> None:
    """Route LMCache's CPU slab through huge-page host memory for fast D2H.

    ``torch.rbln.huge_host_empty`` returns prefaulted pages, so the transfer
    does not take a page fault per 4 KiB. Patches the ``_allocate_cpu_memory``
    / ``_free_cpu_memory`` globals that ``MixedMemoryAllocator`` reads;
    ``shm``/empty and allocation failures fall back to upstream.
    """
    # First Party
    import lmcache.v1.memory_management as mm

    _orig_alloc = mm._allocate_cpu_memory
    _orig_free = mm._free_cpu_memory

    def _alloc(
        size: int,
        numa_mapping: "Optional[NUMAMapping]" = None,
        shm_name: Optional[str] = None,
        use_hugepages: bool = False,
    ) -> torch.Tensor:
        if size == 0 or shm_name is not None:  # shm / empty -> upstream
            return _orig_alloc(size, numa_mapping, shm_name, use_hugepages)
        try:
            buffer = allocate_cpu_pool(size)
        except (RuntimeError, MemoryError) as e:
            logger.warning("RBLN: huge-host alloc(%d) failed (%s); upstream", size, e)
            return _orig_alloc(size, numa_mapping, shm_name, use_hugepages)
        return buffer

    def _free(
        buffer: torch.Tensor,
        size: Optional[int] = None,
        numa_mapping: "Optional[NUMAMapping]" = None,
        shm_name: Optional[str] = None,
        use_hugepages: bool = False,
    ) -> None:
        if release_cpu_pool(buffer):
            return
        _orig_free(buffer, size, numa_mapping, shm_name, use_hugepages)

    mm._allocate_cpu_memory = _alloc
    mm._free_cpu_memory = _free
    logger.info("RBLN: LMCache CPU slab -> huge host memory")


_original_broadcast_or_receive_memory_objs = None


def _patched_broadcast_or_receive_memory_objs(  # type: ignore[no-untyped-def]
    self, reordered_chunks, ret_mask
):
    """Skip the MLA dedup broadcast's device copy when no rank receives it.

    LMCache turns ``save_only_first_rank`` on for every MLA model. On retrieve
    the leader then copies each chunk into a fresh device tensor
    (``raw_tensor.to(device)``), broadcasts it over the TP group and keeps it
    in ``_leader_gpu_substitute_objs``; ``retrieve()`` restores the KV cache
    from those copies instead of the host chunks. With a single rank nobody
    receives the broadcast, and on RBLN the restore from the device copy is no
    faster than a direct restore from host memory, so the copy is pure cost:
    15.4 of 31.9 ms per two-chunk retrieve on A.X-K3 (32 layers), 17.0 ms per
    chunk on A.X-K2 (61-layer KV). See rebellions-sw/lmcache-rbln#310.

    With one rank, hand ``retrieve()`` the retrieved chunks themselves as the
    substitutes: ``batched_to_gpu`` then restores from where the backend
    holds them (host memory for the CPU pool), and ``retrieve()``'s cleanup
    still releases each chunk once (the list takes no LMCache ref count or
    pin of its own). ``save_only_first_rank``
    stays on, so everything else it selects (the leader-only store, the CPU
    pool sizing, the cache key, the scheduler-side lookup) is untouched.

    TEMPORARY: this belongs in ``LMCacheEngine`` itself (skip the broadcast
    when the TP group has one member).
    """
    assert _original_broadcast_or_receive_memory_objs is not None
    if self.metadata.world_size != 1:
        return _original_broadcast_or_receive_memory_objs(
            self, reordered_chunks, ret_mask
        )
    self._leader_gpu_substitute_objs = [mo for _, mo, _, _ in reordered_chunks]
    return None


def apply() -> None:
    """Apply all RBLN runtime patches to LMCache.  Safe to call multiple times."""
    global _applied, _original_broadcast_or_receive_memory_objs
    # Fail on an invalid setting before any patch, in every transport mode.
    host_register_enabled()
    if _applied:
        return

    # First Party
    from lmcache.inprocess.v1.cache_engine import LMCacheEngine

    # With one rank, restore MLA chunks straight from host memory instead of
    # from a device copy made for a broadcast nobody receives (#310).
    _original_broadcast_or_receive_memory_objs = (
        LMCacheEngine._broadcast_or_receive_memory_objs
    )
    LMCacheEngine._broadcast_or_receive_memory_objs = (
        _patched_broadcast_or_receive_memory_objs
    )

    # Before the LMCache pool is allocated (apply() runs at import time).
    _patch_cpu_slab_huge_host()

    # First Party
    from lmcache.inprocess.rbln._shm_registration import patch_shm_registration

    patch_shm_registration()

    _applied = True
    logger.info("LMCache RBLN runtime patches applied")

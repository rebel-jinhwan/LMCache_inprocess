.. _rbln_npu:

RBLN NPU (vLLM-RBLN)
====================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.

``lmcache.inprocess.rbln`` connects the in-process engine to vLLM-RBLN on
Rebellions RBLN NPUs. KV caches stay ``torch`` tensors on ``device='rbln'``
and every transfer between device and host goes through torch device ops.

Requirements
------------

- An RBLN NPU host with the Rebellions driver installed.
- ``torch-rbln`` and ``vllm-rbln`` from the RBLN SDK. They are not
  dependencies of ``lmcache_inprocess``; install them in the same
  environment.
- vLLM-RBLN in device-tensor mode (``VLLM_RBLN_USE_VLLM_MODEL=1``, with
  ``VLLM_RBLN_USE_DEVICE_TENSOR`` left at its default), set at process
  start-up.
- ``lmcache-rbln`` for the RBLN runtime patches to LMCache: a huge-page,
  device-registered CPU pool, registered staging for MP transfers, and no
  device copy for an MLA broadcast with a single rank. They apply on
  ``import lmcache_rbln``, before the engine allocates its CPU pool;
  vLLM-RBLN's registered ``RBLNLMCacheConnectorV1`` does that import.
  Without them, transfers from unregistered host memory take the ordinary
  copy path.

The RDS (NVMe) storage tier is a separate LMCache storage plugin,
``RDSBackend``, shipped by ``lmcache-rbln``. The connector does not import it:
it finds a loaded ``RDSBackend`` by name, publishes its evictions as
``BlockRemoved`` KV events and, with ``rds_failure_policy=fail``, raises its
recorded failures on the worker.

Start the server
----------------

.. code-block:: bash

   LMCACHE_CHUNK_SIZE=64 \
   LMCACHE_LOCAL_CPU=True \
   LMCACHE_MAX_LOCAL_CPU_SIZE=5.0 \
   vllm serve Qwen/Qwen3-1.7B \
     --max-model-len 1024 \
     --block-size 512 \
     --enable-prefix-caching \
     --max-num-batched-tokens 128 \
     --max-num-seqs 4 \
     --kv-transfer-config \
     '{"kv_connector":"RBLNLMCacheConnectorV1",
       "kv_connector_module_path":"lmcache.inprocess.rbln.integration.vllm.connector",
       "kv_role":"kv_both"}'

Send two requests whose shared prefix fills at least one chunk
(``LMCACHE_CHUNK_SIZE`` tokens); the second one is served from the cache.

Limitations
-----------

- ``LMCACHE_USE_LAYERWISE`` is rejected: RBLN compiles the full forward graph,
  so vLLM never calls the per-layer hooks.
- Hybrid full/sliding attention (HMA) keeps a separate block table per KV
  group. It requires vLLM-RBLN to register every layer's cache view and
  ``VLLM_RBLN_SUB_BLOCK_CACHE=false``.
- HMA requires ``--tensor-parallel-size 1 --pipeline-parallel-size 1``: load
  failures are not coordinated across model-parallel workers. Data and expert
  parallelism are not affected.
- HMA saves and reuses complete ``--block-size`` blocks only, and
  ``LMCACHE_CHUNK_SIZE`` must divide the block size. Partial blocks are not
  saved, even with ``save_unfull_chunk``.
- HMA chunks store sliding-window layers in kernel-block order, and chunk keys
  do not record the kernel block size: every worker that shares a store must
  run the same sliding-window kernel block size.
- With ``RBLN_COMP_DTYPE=dlfloat``, the physical DLFloat16 KV cache is stored
  through its logical BF16 dtype, which loses precision and can change tokens
  generated after a restore. Use the compiler default when exact cold/warm
  parity is required.

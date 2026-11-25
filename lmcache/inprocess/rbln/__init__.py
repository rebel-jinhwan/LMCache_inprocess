# SPDX-License-Identifier: Apache-2.0
"""In-process LMCache connector for the RBLN NPU (Rebellions).

Integration with vLLM-RBLN:
    Set ``kv_connector`` to ``RBLNLMCacheConnectorV1`` and
    ``kv_connector_module_path`` to
    ``lmcache.inprocess.rbln.integration.vllm.connector`` in the vLLM KV
    transfer config. Importing this package does not patch LMCache.

    The runtime patches RBLN deployments apply to LMCache (huge-host CPU
    slab, managed shared-memory registration, the single-rank MLA broadcast
    skip) ship in ``lmcache-rbln`` and take effect on ``import lmcache_rbln``,
    which must happen before the engine allocates its CPU pool.

    The RDS (NVMe) storage tier is a separate LMCache storage plugin shipped
    by ``lmcache-rbln``; see :mod:`lmcache.inprocess.rbln._rds_plugin` for the
    interface the connector expects from it.

Note:
    This subpackage needs ``torch-rbln`` (and ``vllm-rbln`` for the vLLM
    integration) at runtime. They are not dependencies of
    ``lmcache_inprocess``: install them from the RBLN SDK.

    vllm-rbln must run in device-tensor mode (platform ``device_type ==
    "rbln"``) so KV caches are torch tensors on ``device='rbln'``;
    :class:`~lmcache.inprocess.rbln.v1.rbln_connector.RBLNConnector`
    transfers them using torch device ops only. vllm-rbln enables that mode
    from ``VLLM_RBLN_USE_VLLM_MODEL=1`` plus ``VLLM_RBLN_USE_DEVICE_TENSOR``
    (on by default), frozen when its platform class is first imported, so
    both belong at process start-up. This package intentionally does NOT
    touch either env var: ``RBLNLMCacheConnectorV1`` asks the resolved
    platform instead.
"""

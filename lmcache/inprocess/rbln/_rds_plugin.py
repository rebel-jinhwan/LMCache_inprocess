# SPDX-License-Identifier: Apache-2.0
"""What the RBLN connector expects from the RDS storage plugin.

``RDSBackend`` ships separately in ``lmcache-rbln`` and LMCache loads it as
a storage plugin (``LMCACHE_STORAGE_PLUGINS=RDSBackend``). This module keeps
the connector free of an import dependency on it: the connector finds the
backend by name and calls only the methods listed here.

- ``raise_if_failed()``: raise the first terminal failure the backend
  recorded under ``rds_failure_policy=fail``.
- ``take_evicted_hashes()``: return and clear the chunk hashes evicted since
  the last call, published as ``BlockRemoved`` with medium
  :data:`RDS_KV_EVENT_MEDIUM`.
"""

# Standard
from typing import Any, Optional

RDS_BACKEND_NAME = "RDSBackend"

# The ``medium`` RDS stores and removals carry in KV cache events.
RDS_KV_EVENT_MEDIUM = "rds"


def get_rds_backend(engine: object) -> Optional[Any]:
    """Return the active RDS backend for ``engine`` (or None)."""
    sm = getattr(engine, "storage_manager", None)
    if sm is None:
        return None
    for b in getattr(sm, "storage_backends", {}).values():
        if type(b).__name__ == RDS_BACKEND_NAME:
            return b
    return None


def fail_fast_enabled(config: Any) -> bool:
    policy = (config.extra_config or {}).get("rds_failure_policy", "ignore")
    if policy not in ("fail", "ignore"):
        raise ValueError("rds_failure_policy must be 'fail' or 'ignore'")
    return policy == "fail"


def validate_rds_backend(config: Any, engine: Any) -> None:
    """Reject a swallowed plugin-init error after KV registration in fail mode."""
    manager = getattr(engine, "storage_manager", None)
    # Passive ranks can deliberately omit storage after a successful post_init.
    if manager is None and getattr(engine, "post_inited", False):
        return
    backends = getattr(manager, "storage_backends", {})
    if backends.get(RDS_BACKEND_NAME) is None and RDS_BACKEND_NAME in (
        config.storage_plugins or []
    ):
        raise RuntimeError("RDSBackend was requested but did not initialize")


def raise_if_engine_failed(engine: Any) -> None:
    """Surface a recorded failure from the current backend, without reading config."""
    manager = getattr(engine, "storage_manager", None)
    backends = getattr(manager, "storage_backends", {})
    # LMCache can close and recreate backends through its management API.
    backend = backends.get(RDS_BACKEND_NAME)
    if backend is not None:
        backend.raise_if_failed()

# SPDX-License-Identifier: Apache-2.0
"""``RBLNLMCacheConnectorV1`` starts only when vllm-rbln is in device-tensor mode.

The guard used to require ``VLLM_RBLN_USE_DEVICE_TENSOR`` to be exported even
though vllm-rbln defaults it on, so a launch that relied on that default
refused to start. It now asks the resolved platform instead, so these cases
leave that variable at vllm-rbln's default and vary only
``VLLM_RBLN_USE_VLLM_MODEL``, the setting launches still have to provide.

Each case runs in its own interpreter against the real ``current_platform``:
vllm resolves and freezes the platform on first import, so in-process cases
would all see whichever env the first one had. Only the engine construction
behind the guard is mocked, so no NPU or compile is needed.
"""

# Standard
from typing import Optional
import os
import subprocess
import sys

_MODE_ENV = ("VLLM_RBLN_USE_VLLM_MODEL", "VLLM_RBLN_USE_DEVICE_TENSOR")

_PROBE = """
from unittest.mock import MagicMock, patch

from lmcache.inprocess.rbln.integration.vllm import connector as m

with (
    patch.object(m.KVConnectorBase_V1, "__init__"),
    patch.object(m, "RBLNLMCacheConnectorV1Impl"),
):
    try:
        m.RBLNLMCacheConnectorV1(MagicMock(), MagicMock())
    except RuntimeError as e:
        print("GUARD=rejected", e)
    else:
        print("GUARD=started")
"""


def _run_probe(use_vllm_model: Optional[str]) -> str:
    """Construct the connector in a fresh interpreter; return its GUARD line."""
    # Drop inherited mode vars so the caller's shell cannot steer the result.
    env = {k: v for k, v in os.environ.items() if k not in _MODE_ENV}
    # Satisfy the PYTHONHASHSEED guard that runs first.
    env["PYTHONHASHSEED"] = "0"
    if use_vllm_model is not None:
        env["VLLM_RBLN_USE_VLLM_MODEL"] = use_vllm_model
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("GUARD=")]
    assert len(lines) == 1, proc.stdout[-4000:]
    return lines[0]


def test_starts_on_vllm_rbln_default() -> None:
    """The bug case: USE_VLLM_MODEL=1 and nothing else must be enough."""
    assert _run_probe("1") == "GUARD=started"


def test_rejects_host_tensor_platform() -> None:
    """Without USE_VLLM_MODEL vllm-rbln resolves a ``cpu`` platform."""
    line = _run_probe(None)
    assert line.startswith("GUARD=rejected"), line
    assert "device-tensor mode" in line, line

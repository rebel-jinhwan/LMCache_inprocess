# SPDX-License-Identifier: Apache-2.0
"""Check the RBLN runtime and that LMCache InProcess dispatches to RBLN.

Run after LMCache and LMCache InProcess are installed. Exits non-zero when
torch.rbln sees no NPU, a tensor op on rbln:0 gives a wrong result, LMCache
auto-detected a different device backend, or LMCache InProcess was not
installed from the checkout under test.

Usage: python preflight.py
"""

# Standard
from pathlib import Path
import importlib.metadata

# Third Party
import torch

# First Party
from lmcache.inprocess.v1 import gpu_connector
import lmcache
import lmcache.inprocess

#: The checkout this script belongs to: .buildkite/k3_tests/rbln/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """Print the RBLN stack versions and fail on any broken check.

    Raises:
        SystemExit: If no NPU is visible, the probe op is wrong, LMCache did
            not select the ``rbln`` backend, or ``lmcache.inprocess`` resolves
            to the source tree or to another installation.
    """
    for dist in (
        "torch",
        "torch-rbln",
        "rebel-compiler",
        "vllm",
        "vllm-rbln",
        "lmcache",
        "lmcache_inprocess",
    ):
        try:
            print(dist, importlib.metadata.version(dist))
        except importlib.metadata.PackageNotFoundError:
            print(dist, "not installed")
    print("lmcache", lmcache.__file__)
    print("lmcache.inprocess", lmcache.inprocess.__file__)

    if torch.rbln.device_count() == 0:
        raise SystemExit("torch.rbln sees no NPU")
    probe = torch.arange(6, dtype=torch.float32).to("rbln:0")
    result = (probe + probe).cpu().tolist()
    if result != [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        raise SystemExit(f"wrong result on rbln:0: {result}")

    if lmcache.torch_device_type != "rbln":
        raise SystemExit(
            f"LMCache selected {lmcache.torch_device_type!r}, expected 'rbln'"
        )
    # CreateGPUConnector dispatches on its own import of the device type.
    if gpu_connector.torch_device_type != "rbln":
        raise SystemExit(
            "lmcache.inprocess.v1.gpu_connector sees "
            f"{gpu_connector.torch_device_type!r}, expected 'rbln'"
        )

    installed = Path(lmcache.inprocess.__file__).resolve()
    if installed.is_relative_to(REPO_ROOT):
        raise SystemExit(f"lmcache.inprocess imports from the checkout: {installed}")
    if not (installed.parent / "v1" / "gpu_connector" / "rbln_connector.py").is_file():
        raise SystemExit(f"lmcache.inprocess at {installed} has no RBLNConnector")
    print("rbln preflight ok")


if __name__ == "__main__":
    main()

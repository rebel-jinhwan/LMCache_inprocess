#!/usr/bin/env bash
# LMCache InProcess RBLN smoke: install LMCache at UPSTREAM_REVISION with the
# RBLN build profile plus this checkout into the rbln-serve image, check the
# RBLN runtime, then run the RBLNConnector tests on a real NPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"

echo "--- :package: Install LMCache and LMCache InProcess"
LMCACHE_CI_BUILD_PROFILE=rbln PYTHON="${PYTHON}" \
    bash "${REPO_ROOT}/scripts/install_ci_dependencies.sh"
"${PYTHON}" -m pip install --no-cache-dir pytest

# Run from outside the checkout so the tests import the installed package,
# not the source tree.
cd "$(mktemp -d)"

echo "--- :mag: RBLN preflight"
"${PYTHON}" "${SCRIPT_DIR}/preflight.py"

echo "--- :test_tube: RBLNConnector tests on the NPU"
# -rs lists skips: every rbln-parametrized case must run here, and preflight
# has already failed the build if torch-rbln saw no NPU.
"${PYTHON}" -m pytest -q -rs \
    "${REPO_ROOT}/tests/inprocess/v1/test_rbln_connector.py"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
revision_file="${repo_root}/UPSTREAM_REVISION"

if [[ ! -s "${revision_file}" ]]; then
  echo "missing upstream revision file: ${revision_file}" >&2
  exit 1
fi

upstream_revision="$(sed -n 's/^revision=//p' "${revision_file}" | head -n 1)"
if [[ ! "${upstream_revision}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "UPSTREAM_REVISION must start with a 40-character commit SHA" >&2
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$(command -v "${PYTHON}")"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif command -v python3.12 >/dev/null 2>&1; then
  python_bin="$(command -v python3.12)"
else
  python_bin="$(command -v python3)"
fi

build_profile="${LMCACHE_CI_BUILD_PROFILE:-}"

if [[ "${build_profile}" == "rbln" ]]; then
  # Install into the rbln-serve image's own environment, next to its
  # torch-rbln / rebel-compiler pair; the image has neither git nor curl.
  installer=("${python_bin}" -m pip install --no-cache-dir)
elif [[ -n "${build_profile}" ]]; then
  echo "unknown LMCACHE_CI_BUILD_PROFILE: ${build_profile}" >&2
  exit 1
elif command -v uv >/dev/null 2>&1; then
  installer=(uv pip install)
else
  installer=("${python_bin}" -m pip install)
fi

if [[ "${build_profile}" == "rbln" ]]; then
  # PyPI's lmcache wheel is the CUDA build. LMCache publishes its RBLN wheel,
  # built against torch-rbln's torch, to a <tag>-rbln GitHub Release instead.
  # Install the one a user gets for the `lmcache` dependency: the latest
  # LMCache release, unless LMCACHE_RBLN_RELEASE names a tag (e.g. v0.5.8).
  rbln_release="${LMCACHE_RBLN_RELEASE:-$(
    "${python_bin}" - <<'PY'
import urllib.request

# /releases/latest redirects to /releases/tag/<tag>; no API token needed.
url = "https://github.com/LMCache/LMCache/releases/latest"
with urllib.request.urlopen(url, timeout=30) as resp:
    print(resp.geturl().rstrip("/").rsplit("/", 1)[1])
PY
  )}"
  echo "Installing the LMCache ${rbln_release} RBLN wheel"
  "${installer[@]}" \
    --find-links "https://github.com/LMCache/LMCache/releases/expanded_assets/${rbln_release}-rbln" \
    "lmcache==${rbln_release#v}+rbln"
  # The checkout may be owned by another user than the job, which hides its
  # Git metadata from setuptools_scm.
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_INPROCESS="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_INPROCESS:-0.0.0+ci}"
else
  echo "Installing LMCache dependency at ${upstream_revision}"
  "${installer[@]}" \
    "lmcache @ git+https://github.com/LMCache/LMCache.git@${upstream_revision}"
fi

echo "Installing the current lmcache_inprocess checkout"
# A regular wheel overwrites the legacy inprocess tree bundled by older LMCache
# versions; a PEP 660 editable finder loses to that existing package on sys.path.
"${installer[@]}" --no-deps --force-reinstall "${repo_root}"

"${python_bin}" - <<'PY'
from importlib.metadata import requires, version

from packaging.requirements import Requirement

import lmcache
import lmcache.inprocess

print(f"LMCache dependency: {lmcache.__file__}")
print(f"LMCache InProcess checkout: {lmcache.inprocess.__file__}")

# The checkout was installed with --no-deps, so check its lmcache requirement.
installed = version("lmcache")
for line in requires("lmcache_inprocess") or []:
    requirement = Requirement(line)
    if requirement.name == "lmcache" and not requirement.specifier.contains(
        installed, prereleases=True
    ):
        raise SystemExit(f"lmcache {installed} does not satisfy {requirement}")
print(f"LMCache {installed} satisfies lmcache_inprocess's requirement")
PY

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
  # The rbln-serve image pins a matched torch-rbln / rebel-compiler pair that a
  # resolver must not replace, and it ships neither git nor curl.
  installer=("${python_bin}" -m pip install --no-cache-dir)
elif [[ -n "${build_profile}" ]]; then
  echo "unknown LMCACHE_CI_BUILD_PROFILE: ${build_profile}" >&2
  exit 1
elif command -v uv >/dev/null 2>&1; then
  installer=(uv pip install)
else
  installer=("${python_bin}" -m pip install)
fi

echo "Installing LMCache dependency at ${upstream_revision}"
if [[ "${build_profile}" == "rbln" ]]; then
  lmcache_src="$(mktemp -d)"
  trap 'rm -rf "${lmcache_src}"' EXIT
  "${python_bin}" - "${upstream_revision}" "${lmcache_src}" <<'PY'
import io
import sys
import tarfile
import urllib.request

revision, dest = sys.argv[1:]
url = f"https://codeload.github.com/LMCache/LMCache/tar.gz/{revision}"
with urllib.request.urlopen(url, timeout=300) as resp:
    payload = resp.read()
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    members = []
    for member in archive.getmembers():
        # Drop the archive's single top-level LMCache-<revision>/ directory.
        _, _, member.name = member.name.partition("/")
        if member.name:
            members.append(member)
    archive.extractall(dest, members=members, filter="data")
PY
  "${installer[@]}" \
    -r "${lmcache_src}/requirements/build.txt" \
    -r "${lmcache_src}/requirements/common.txt"
  # The RBLN profile builds only LMCache's common C++ extensions, against the
  # image's torch; --no-deps keeps the image's torch-rbln pair untouched. The
  # tarball has no Git metadata, so setuptools_scm needs the version given.
  BUILD_WITH_RBLN=1 \
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${LMCACHE_CI_VERSION:-0.5.8.dev0+ci}" \
    "${installer[@]}" --no-deps --no-build-isolation --force-reinstall \
    "${lmcache_src}"
  # The checkout may be owned by another user than the job, which hides its
  # Git metadata from setuptools_scm as well.
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_INPROCESS="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_INPROCESS:-0.0.0+ci}"
else
  "${installer[@]}" \
    "lmcache @ git+https://github.com/LMCache/LMCache.git@${upstream_revision}"
fi

echo "Installing the current lmcache_inprocess checkout"
# A regular wheel overwrites the legacy inprocess tree bundled by older LMCache
# versions; a PEP 660 editable finder loses to that existing package on sys.path.
"${installer[@]}" --no-deps --force-reinstall "${repo_root}"

"${python_bin}" - <<'PY'
import lmcache
import lmcache.inprocess

print(f"LMCache dependency: {lmcache.__file__}")
print(f"LMCache InProcess checkout: {lmcache.inprocess.__file__}")
PY

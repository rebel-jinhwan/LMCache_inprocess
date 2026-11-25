# Buildkite Web UI Setup: RBLN InProcess Hardware Smoke

**Steps editor**: paste the contents of `buildkite-pipeline.yml`.

**GitHub trigger settings**:

- Filter: `build.pull_request.labels includes "rbln" || build.pull_request.labels includes "full"`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

**Cluster prerequisites**: the same as LMCache's `rbln-mp-test` pipeline.
`rbln-queue` is served by agent-stack-k8s on the `npu2` node group. The step
claims one NPU from the `buildkite/npu1` DRA ResourceClaimTemplate and runs
the `rbln-serve` image, whose `/opt/venv` already holds a matched
`torch-rbln` / `rebel-compiler` pair, so no RBLN Portal credentials are
needed. The image runs as uid 2000, which cannot write `/opt/venv`, so the
step sets `runAsUser: 0`. Verified images live on `harbor.mgmt.rbln.in`,
which rejects anonymous pulls, so the `rbln-queue` job pods need an image
pull secret for it.

**Image selection**: `upload-pipeline.sh` sets the step's image to the last
`rbln-serve` image LMCache's `rbln-mp-nightly` lane verified, read from
`tested_runtimes.jsonl` on LMCache's `buildkite_latest_tested_vllm` branch by
`k3_harness/resolve-pinned-vllm.sh`. There is no fallback image: when no
verified record can be read, the upload step fails. Set `PINNED_RBLN_IMAGE`
on a build to override the lookup. This repository has no nightly lane of
its own.

**What runs**: `run.sh` installs the `lmcache` that `lmcache_inprocess`
depends on, as RBLN users get it: PyPI's `lmcache` wheel is the CUDA build, so
it takes the `lmcache==<version>+rbln` wheel from the latest LMCache release's
`<tag>-rbln` GitHub Release (set `LMCACHE_RBLN_RELEASE`, e.g. `v0.5.8`, to pick
another), then this checkout as a regular wheel, and fails if that `lmcache`
does not satisfy the checkout's `lmcache` requirement. Unlike the CUDA
pipelines it does not use `UPSTREAM_REVISION`: LMCache publishes no RBLN wheel
per commit. `preflight.py` checks the NPU, that LMCache
selected the `rbln` backend, and that `lmcache.inprocess` is the installed
checkout. `tests/inprocess/v1/test_rbln_connector.py` then runs its
`rbln`-parametrized round trips on real `device='rbln'` KV caches.

There is no vLLM end-to-end step yet.

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the repository path filter. Add the `force-ci` label to a PR to bypass.

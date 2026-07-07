# Docker images

Two images serve two different execution environments:

## `local/` — the dev image

The interactive development environment. Built by `./build_docker_image.py` (or the setup
wizard) and launched by `./run_docker.py`, which bind-mounts the live repo at
`/workspace/repo` and the persistent mount dir at `/workspace/mount`. It carries the full
toolchain (compilers, CUDA/TensorRT, PyTorch, Node) and never contains repo code or data —
everything comes in through the bind mounts. Stays on the local machine; never pushed to a
registry.

## `worker/` — the cloud worker image

A slim, headless runtime for rented cloud machines (Runpod CPU pods). Deliberately contains
**dependencies only** — no repo code, no compiled binaries, no lexica — so that the image
pushed to the registry is stable and code iteration never requires an image rebuild or
re-pull. At pod start its baked-in `bootstrap.py` downloads a code+binary bundle (uploaded
from the dev container by `py/scripts/cloud_push_binaries.py`) from the R2 bucket, unpacks
it at `/workspace/repo`, and hands off to the bundle's worker entrypoint, which fetches the
remaining runtime data (lexica, Macondo strategy files) from their public upstreams.

Built and pushed by `./build_worker_image.py`. Rebuild only when the worker's runtime
dependencies change. See docs/cloud_compute.md for the full architecture.

Shared shell scripts (`entrypoint.sh`, `devuser-setup.sh`) referenced by the local
Dockerfile are overlaid into the build context from `subtrees/devenv_utils/docker/` at
build time; they do not live in this directory.

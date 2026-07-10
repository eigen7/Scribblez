# Distributed data generation on rented cloud CPUs

How Scribblez runs `generate_kill_test_data.py`-style workloads across a fleet of rented
cloud machines (Runpod CPU pods), with results flowing back to the local machine where
analysis (`kill_test.py`, dashboards) runs unchanged. Designed with an eye toward two
futures: GPU workloads on the same scaffolding, and volunteer-donated compute
(katagotraining.org-style).

## Why this workload distributes trivially

- **Embarrassingly parallel.** Each cycle (one self-play batch + its sim sidecar) is
  independent; no shared state between cycles or machines.
- **Merge-by-copy.** Output files are named by nanosecond timestamp, so batches produced by
  any number of machines coexist in one directory with no coordination.
- **Interruption-safe.** `.slog` batches and `.sobs` sidecars land atomically; a killed
  worker loses at most its in-flight cycle (~2-3 minutes of work).
- **Tiny data.** A cycle produces ~4 MB; a week of 28-core laptop time is ~15 GB — data
  transfer cost is a non-issue.
- **Light runtime deps.** HastyBot self-play + `sim_obs_tool` need the engine binaries,
  `.kwg` lexica (fetched from the public liwords URL), and Macondo's `data/strategy` files
  (sparse-cloned from the public Macondo repo). No GPU, no Macondo binary, no models.

## Architecture

Three stores decouple everything: a **container registry** (stable worker image), the **R2
bucket** (code bundles outbound, results inbound), and the **local mount dir** (analysis
home).

```
 dev container                                    cloud
 ─────────────                                    ─────
 py/build.py -b            (per-arch binaries)
 cloud_push_binaries.py ──────────────────────►  R2: bundles/<id>/bundle-<arch>.tar.gz
 build_and_push_worker_image.py  ──(deps only, rare)───►  docker.io: worker image
 cloud_fleet.py up      ──────────────────────►  N Runpod CPU pods
                                                   └─ bootstrap.py: pull bundle for its
                                                      CPU arch, exec worker entrypoint
                                                   └─ loop: generate cycle, upload pair
                                                          │
 cloud_sync.py ◄──────────────────────────────  R2: <workload>/<tag>/{slogs|staging,stats,params}/
      │
 <mount>/tags/<workload>/<tag>/data/   ◄── analysis (kill_test.py) / the generation
                                           scheduler run here, locally, as always
```

Principles:

1. **The registry image is dependency-only and stable.** It changes only when worker
   *dependencies* change — never for code iteration — so Runpod's per-host image cache
   stays warm and pods never pull a fat uncached image because the engine was recompiled.
2. **Code travels as bundles through R2, not through Docker or git.** What runs in the
   cloud is bit-for-bit what was last built and tested locally (uncommitted changes
   included, flagged as `-dirty` in the bundle id). Workers need no repo credentials and
   never compile.
3. **Workers are stateless and disposable.** A worker starts, fetches its bundle and data
   deps, and loops generate-upload until terminated; SIGTERM flushes completed pairs. The
   bucket is the durable archive; the laptop holds a synced copy for analysis.

## The pieces

### Worker image — `docker-setup/worker/`, `./build_and_push_worker_image.py`

`ubuntu:24.04` + engine runtime libraries + rclone + `g++` (only for CPU-arch detection) +
the baked-in `bootstrap.py` entrypoint. The NVIDIA/TensorRT runtime `.so`s (which the
engine binaries link but CPU workloads never execute) are copied out of the local dev image
rather than pulled via a multi-gigabyte NVIDIA base. Contains no repo code, no binaries, no
lexica. Built and pushed to the private registry repo (`registry.worker_image` in the
credentials file) by `./build_and_push_worker_image.py` on the host; rebuild only when
`docker-setup/worker/` changes.

### Bundles — `py/cloud/bundles.py`, `./py/scripts/cloud_push_binaries.py`

The engine builds once per CPU microarchitecture in `SUPPORTED_ARCHS` (py/build.py
`--build-for-all-archs`), including the generic `x86-64` fallback. A push uploads one
tarball per arch (that arch's `play_game` / `sim_obs_tool` / `libscribblez_ffi.so` plus the
arch-independent `py/` tree) under `bundles/<bundle_id>/` and points `bundles/LATEST` at
it. At pod start, `bootstrap.py` detects the pod's arch (same `g++ -march=native` probe as
py/build.py), downloads the matching tarball — falling back to `x86-64` with a warning that
names the arch to add to `SUPPORTED_ARCHS` — unpacks it at `/workspace/repo`, and execs the
bundle's `py/cloud/worker_entrypoint.py`. So even the worker-loop logic is iterable without
touching the image.

### Worker entrypoint — `py/cloud/worker_entrypoint.py`

Configured entirely by environment variables (see its docstring). Dispatches on
`(SCZ_WORKLOAD, SCZ_ROLE)` to the role's runner from the workload registry
(scribblez/workloads/): the runner's declared deps are fetched from their public
upstreams (idempotent; a dev container's populated mount short-circuits it), a
provenance manifest lands at `<workload>/<tag>/params/<worker_id>.json` (params, role,
bundle id, arch), and the runner loops its cycle, delivering whole output files through
the results sink (`py/cloud/sinks.py`; kill-test pairs upload `.sobs` first so the
bucket only ever presents complete pairs plus inert orphans). SIGTERM flushes completed
output and exits.

### Fleet control — `./py/scripts/cloud_fleet.py`

`up` / `status` / `down` over the Runpod REST API. `up -n 4 --vcpus 16 -t hello` launches
pods named `scz-hello-<suffix>` from the worker image with the R2 credentials and workload
knobs in their environment; `--bundle latest` resolves to a concrete bundle id at launch so
one fleet is homogeneous even if newer bundles land meanwhile. `status -t hello` also
counts the tag's complete pairs in the bucket. `down` only ever touches `scz-` pods.

### Results sync — `./py/scripts/cloud_sync.py`

`-t hello [--watch]` pulls the workload's inbound bucket prefixes — its declared data
dirs (kill_test: `slogs/`; the training workloads: `staging/`) plus `stats/` and
`params/` — into `<mount>/tags/<workload>/<tag>/`, merging with locally generated data
for the same tag. Prefixes the controller host itself maintains in the bucket (the
generation dirs the scheduler's ingest mirroring populates) are deliberately not pulled.
Analysis stays local and unchanged.

### Credentials — `<mount>/cloud/credentials.json`

Single operator-filled file (template written by setup_wizard.py or
`cloud_check_credentials.py`; validated end-to-end by the latter): Runpod API key +
registry-auth id, worker image repo, R2 account/keys/bucket. Workers receive only the R2
subset, via pod env vars.

## The daily loop

```
# once per code change (seconds + a ~40 MB/arch upload):
./py/build.py -b            # build all SUPPORTED_ARCHS
./py/scripts/cloud_push_binaries.py

# run an experiment:
./py/scripts/cloud_fleet.py up -n 8 --vcpus 16 -t hello
./py/scripts/cloud_fleet.py status -t hello
./py/scripts/cloud_sync.py -t hello --watch    # meanwhile, kill_test.py -t hello locally
./py/scripts/cloud_fleet.py down -t hello
```

## Economics

Target scale: a week of 28-core laptop time ≈ **4,700 vCPU-hours** per experiment.

| Option | ~$/vCPU-hr | Cost per laptop-week | Notes |
|---|---|---|---|
| Runpod CPU pods | ~$0.02–0.05 (console has per-flavor rates) | ~$100–200 | per-second billing, free egress |
| GCP c3d/c2d spot | ~$0.008 (mid-2026) | ~$40 | preemption is a non-event for this workload |

Storage/transfer is noise (~15 GB per laptop-week; R2 has zero egress fees). GCP spot is
~3-4× cheaper for pure CPU; Runpod is the implemented provider (preference, simpler API,
and the GPU future lands there). The provider surface is one small adapter
(`py/cloud/runpod_api.py` + the `create_pod` call in cloud_fleet.py); a GCP spot adapter —
`gcloud compute instances create-with-container` on spot Container-Optimized-OS VMs running
the same worker image — is worth adding if CPU spend starts to matter.

## Instance discovery (`fetch_cloud_offers`)

`py/cloud/runpod_api.py` exposes the live Runpod instance catalog through the public
GraphQL endpoint (`api.runpod.io/graphql`), which answers the read-only catalog queries
with no API key (a User-Agent header is the only requirement). `fetch_cloud_offers()`
returns `{"cpu": [...], "gpu": [...]}`: the six fixed CPU flavors with on-demand
`$/vCPU/hr` price, RAM multiplier, disk-per-vCPU, and stock; and the GPU types with
per-cloud on-demand and spot pricing, VRAM, max GPU count, and stock. GPU types with no
stock or no price in either the secure or community cloud are dropped, and a per-cloud
price is reported only for a cloud the type is actually offered in. The REST API used for
pod CRUD has no discovery or pricing endpoints, so this is the only source. The dashboard
serves it (cached) at `GET /api/cloud/offers` to drive the add-cloud instance selector;
the CPU spot rate is not exposed by the API and is not shown.

## GPU pod creation

`pod_create_spec` (scripts/cloud_fleet.py) builds either a CPU spec (`computeType: "CPU"`,
`cpuFlavorIds`, `vcpuCount`) from `CpuResources`, or a GPU spec (`computeType: "GPU"`,
`gpuTypeIds`, `gpuCount`) from `GpuResources`. A `RoleSpec` declares `gpu=True` for a role
that runs on GPU hardware (the trainer roles), and the dashboard offers GPU instances for
such roles and CPU flavors otherwise. This pod-spec path is forward-looking: no cloud role
declares `gpu` today, and the worker image is CPU-only, so launching a GPU pod that runs
the current worker loop is not yet wired end-to-end.

## Future: GPU workloads

The remaining pieces for GPU work in the cloud: a cloud-capable GPU role in the workload
registry, and (since the workload needs CUDA at runtime) a worker-image variant on a CUDA
runtime base with the same bootstrap contract. Bundles and the bucket layout are
workload-agnostic, and the fleet CLI and dashboard already build GPU pod specs.

## Future: volunteer compute

The day-1 choices keep this cheap to add later:

- A volunteer is, to first order, someone running the worker image with a participation
  token — the image/bundle/env-var contract is already exactly that shape, and the image
  contains no redistributable-restricted data (lexica are fetched from public upstream), so
  it can be made publicly pullable.
- The one necessary change is credentials: raw bucket write keys can't go to strangers.
  The worker's uploads are already funneled through two small functions
  (`upload_completed_pairs`, `upload_manifest`), the seam where rclone-with-keys becomes
  HTTPS-ingest-with-token (presigned URLs from a small ingest service, plus server-side
  validation/quarantine of uploaded pairs).

## Status / remaining work

Everything above is implemented and smoke-tested from the dev container (worker entrypoint
end-to-end against the real bucket, fleet CLI against the live Runpod API). Not yet
exercised: a real pod launch (first `build_and_push_worker_image.py` push + a 1-pod `up` with small
knobs), and per-flavor throughput calibration (cycles/hour on `cpu3c` vs `cpu5c` at real
parameters) to pin down actual $/laptop-week.

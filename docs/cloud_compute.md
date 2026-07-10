# Distributed data generation on rented cloud CPUs

How Scribblez runs workload roles (from the workload registry,
scribblez/workloads/) across a fleet of rented cloud machines (Runpod pods),
with results flowing back to the local mount where analysis runs unchanged.
Designed with an eye toward GPU workloads on the same scaffolding and
volunteer-donated compute.

Generation-style roles distribute trivially: cycles are embarrassingly
parallel, output files are uniquely named and land atomically (so batches from
any number of machines merge by copy, and a killed worker loses at most its
in-flight cycle), data volumes are small, and runtime deps are light and
fetched from public upstreams.

## Architecture

Three stores decouple everything: a **container registry** (stable worker
image), the **R2 bucket** (code bundles outbound, results inbound), and the
**local mount dir** (analysis home).

```
 dev container                                    cloud
 ─────────────                                    ─────
 py/build.py -b            (per-arch binaries)
 cloud_push_binaries.py ──────────────────────►  R2: bundles/<id>/bundle-<arch>.tar.gz
 build_and_push_worker_image.py  ──(deps only, rare)───►  docker.io: worker image
 cloud_fleet.py up      ──────────────────────►  N Runpod pods
                                                   └─ bootstrap.py: pull bundle for its
                                                      CPU arch, exec worker entrypoint
                                                   └─ loop: generate cycle, upload
                                                          │
 cloud_sync.py ◄──────────────────────────────  R2: <workload>/<tag>/...
      │
 <mount>/tags/<workload>/<tag>/data/   ◄── analysis runs here, locally, as always
```

Principles:

1. **The registry image is dependency-only and stable.** It changes only when
   worker *dependencies* change — never for code iteration — so Runpod's
   per-host image cache stays warm.
2. **Code travels as bundles through R2, not through Docker or git.** What
   runs in the cloud is bit-for-bit what was last built locally (uncommitted
   changes included, flagged `-dirty` in the bundle id). Workers need no repo
   credentials and never compile.
3. **Workers are stateless and disposable.** A worker fetches its bundle and
   data deps and loops generate-upload until terminated. The bucket is the
   durable archive; the local mount holds a synced copy for analysis.

## The pieces

- **Worker image** (`docker-setup/worker/`,
  `./build_and_push_worker_image.py`): dependency-only runtime image with the
  baked-in `bootstrap.py` entrypoint — no repo code, binaries, or lexica.
  Rebuild only when `docker-setup/worker/` changes.
- **Bundles** (`py/cloud/bundles.py`, `./py/scripts/cloud_push_binaries.py`):
  the engine builds once per supported CPU microarchitecture
  (`py/build.py --build-for-all-archs`); a push uploads one tarball per arch
  (binaries + the arch-independent `py/` tree). At pod start `bootstrap.py`
  detects the pod's arch, downloads the matching tarball (generic `x86-64`
  fallback), unpacks it, and execs the bundle's worker entrypoint — so even
  the worker-loop logic is iterable without touching the image.
- **Worker entrypoint** (`py/cloud/worker_entrypoint.py`): configured by
  environment variables (see its docstring). Dispatches to the (workload,
  role) runner from the workload registry, fetches the runner's declared data
  deps from public upstreams, writes a provenance manifest to the bucket, and
  loops the runner's cycle, delivering whole output files through the results
  sink (`py/cloud/sinks.py`, which orders uploads so the bucket only ever
  presents complete outputs). SIGTERM flushes completed output and exits.
- **Fleet control** (`./py/scripts/cloud_fleet.py`): `up` / `status` / `down`
  over the Runpod REST API. `--bundle latest` resolves to a concrete bundle id
  at launch, so one fleet is homogeneous even if newer bundles land meanwhile.
  Fleet pods are recognized by their `scz-` name prefix; `down` never touches
  anything else. The master dashboard launches workers through the same
  pod-spec path, so pods are identical however launched.
- **Results sync** (`./py/scripts/cloud_sync.py`): pulls the workload's
  inbound bucket prefixes into `<mount>/tags/<workload>/<tag>/`, merging with
  locally generated data for the same tag. Prefixes the controller host itself
  maintains in the bucket are deliberately not pulled.
- **Credentials** (`<mount>/cloud/credentials.json`): one operator-filled file
  (template from setup_wizard.py; validated end-to-end by
  `cloud_check_credentials.py`). Workers receive only the R2 subset, via pod
  env vars.
- **Instance discovery** (`fetch_cloud_offers` in `py/cloud/runpod_api.py`):
  the live Runpod catalog — CPU flavors and GPU types with pricing and stock —
  via the public GraphQL endpoint (the REST API used for pod CRUD has no
  discovery or pricing). The dashboard serves it, cached, at
  `GET /api/cloud/offers` to drive its instance selector.
- **GPU pod specs** (`pod_create_spec` in cloud_fleet.py): builds either a CPU
  or a GPU pod spec; a `RoleSpec` declares `gpu=True` for roles that need GPU
  hardware, and the dashboard offers instance types accordingly.
  Forward-looking: no cloud role declares `gpu` today, and the worker image is
  CPU-only.

## The daily loop

```
# once per code change (seconds + a small per-arch upload):
./py/build.py -b            # build all SUPPORTED_ARCHS
./py/scripts/cloud_push_binaries.py

# run an experiment:
./py/scripts/cloud_fleet.py up -n 8 --vcpus 16 -t hello
./py/scripts/cloud_fleet.py status -t hello
./py/scripts/cloud_sync.py -t hello --watch    # analysis runs locally meanwhile
./py/scripts/cloud_fleet.py down -t hello
```

## Economics

Target scale is ~4,700 vCPU-hours per laptop-week-equivalent experiment:
~$100–200 on Runpod CPU pods, roughly 3–4× cheaper on GCP spot. Runpod is the
implemented provider (simpler API, and the GPU future lands there); the
provider surface is one small adapter (`py/cloud/runpod_api.py` plus the
`create_pod` call in cloud_fleet.py), so a GCP spot adapter is worth adding if
CPU spend starts to matter. Storage/transfer cost is noise (R2 has zero egress
fees).

## Futures

- **GPU workloads.** Remaining pieces: a cloud-capable GPU role in the
  workload registry, and a worker-image variant on a CUDA runtime base with
  the same bootstrap contract. Bundles, bucket layout, fleet CLI, and
  dashboard pod specs are already GPU-ready.
- **Volunteer compute.** A volunteer is, to first order, someone running the
  worker image (publicly pullable; it contains no redistribution-restricted
  data) with a participation token. The one necessary change is credentials —
  raw bucket keys can't go to strangers. Uploads already funnel through the
  results sink, which is the seam where rclone-with-keys becomes
  token-authenticated HTTPS ingest with server-side validation.

## Status

Implemented and smoke-tested from the dev container (worker entrypoint
end-to-end against the real bucket, fleet CLI against the live Runpod API).
Not yet exercised: a real pod launch, and per-flavor throughput calibration to
pin down actual $/laptop-week.

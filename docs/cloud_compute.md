# Distributed data generation on rented cloud CPUs

A proposal for running `generate_kill_test_data.py`-style workloads across a fleet of rented
cloud machines, with results flowing back to the local machine where analysis
(`kill_test.py`, dashboards) continues to run unchanged. Written with an eye toward two
futures: GPU workloads on the same scaffolding, and eventually volunteer-donated compute
(katagotraining.org-style).

## Why this workload is easy to distribute

The kill-test generator has properties that make the distributed design almost trivial:

- **Embarrassingly parallel.** Each cycle (one self-play batch + its sim sidecar) is
  independent. There is no shared state between cycles or machines.
- **Merge-by-copy.** Output files are named by nanosecond timestamp, so batches produced by
  any number of machines coexist in one directory with no coordination and no realistic
  collision risk. "Merging" fleet output into the local mount dir is a file copy.
- **Interruption-safe.** `.slog` batches and `.sobs` sidecars land atomically; a killed
  worker loses at most its in-flight cycle (~2-3 minutes of work). This makes preemptible /
  spot instances usable with zero extra engineering.
- **Tiny data.** A cycle produces ~3.8 MB. A full week of 28-core laptop time is roughly
  15 GB — small enough that data-transfer cost is a non-issue on every provider.
- **Light runtime dependencies.** HastyBot self-play + `sim_obs_tool` need: the engine
  binaries, `.kwg` lexica (fetchable from the public liwords URL), and Macondo's
  `data/strategy` files (fetchable from the public Macondo repo). No GPU, no Macondo binary,
  no model files.

So the design centers on three artifacts — a **worker image**, a **results bucket**, and a
**fleet CLI** — and deliberately avoids anything fancier (shared filesystems, job queues,
coordinators).

## Architecture

```
 laptop (dev container)                     cloud
 ─────────────────────                      ─────
 build engine ──► build worker image ──► container registry (docker.io private / GHCR)
                                              │ pulled by
 fleet CLI (up/status/down) ────────────► N CPU workers (Runpod pods or GCP spot VMs)
                                              │ each runs generate loop,
                                              │ uploads finished pairs after every cycle
                                          object storage bucket  (R2 recommended)
                                              │
 sync tool (pull) ◄───────────────────────────┘
      │
 <mount>/kill_test/<tag>/slogs/   ◄── analysis (kill_test.py, dashboards) runs here,
                                      locally, exactly as today
```

Three principles:

1. **Workers are stateless and disposable.** A worker is one container: it starts, fetches
   its data dependencies, loops generate-and-upload until killed. No worker ever talks to
   another worker. Killing one loses ≤ one cycle.
2. **The bucket is the transport and durable archive; the laptop is the analysis home.**
   Analysis stays local and interactive — no change to `kill_test.py`, the dashboard, or
   any downstream ergonomics. The bucket also means workers never need inbound network
   access to the laptop (which is NAT'd), and it is exactly the ingest shape a future
   volunteer client needs.
3. **Provider-agnostic worker, thin provider adapters.** The worker container is identical
   everywhere. Only the fleet CLI knows about Runpod vs GCP.

## The worker image

Runpod pods *are* containers — there is no VM to run `run_docker.py` inside. So the cloud
unit of deployment is a self-sufficient image whose entrypoint does the work, not the
interactive dev image. Proposed: a second, slim image alongside the dev image.

`docker-setup/worker/Dockerfile`, multi-stage:

- **Stage 1 (builder):** `FROM scribblez` (the local dev image). Copy the repo in, run
  `py/build.py` for the engine targets needed (`play_game`, `sim_obs_tool`,
  `libscribblez_ffi.so`). This stage runs on the laptop at image-build time; cloud nodes
  never compile anything.
- **Stage 2 (runtime):** `FROM ubuntu:24.04` (same glibc/ABI family as the dev image's
  base). Install only runtime deps: Boost runtime libs, libprotobuf, python3 + the small
  set of py/ imports the generator uses, `rclone`. `COPY --from=builder` the binaries and
  the `py/` tree. Result: a few hundred MB instead of the multi-GB dev image — fast to pull
  onto N nodes.

**Entrypoint** (`worker_entrypoint.py`), parameterized entirely by env vars:

1. Fetch data deps into the container-local `/workspace/mount`:
   - `.kwg` lexica from the liwords URL (same fetch `setup_wizard.py` automates — this also
     keeps copyrighted wordlists *out of the image*, which matters if the image is ever
     pulled by volunteers).
   - Macondo `data/strategy` via shallow clone / sparse checkout of the public repo.
2. Write a `params.json` manifest (tag, rollouts, top-k, positions-per-game, open-leaves,
   image version) into the tag's bucket prefix. Every worker's generation parameters come
   from the same launch config; the manifest makes the tag self-describing and lets
   analysis verify that all contributors used identical settings.
3. Loop: run one generation cycle (reusing the existing batch logic), then
   `rclone copy` the finished pair — `.slog` first only after its `.sobs` exists, uploaded
   together — to `bucket:kill_test/<tag>/slogs/`. Uploading only complete pairs preserves
   the invariant that anything visible downstream is analyzable.
4. On SIGTERM (spot preemption, `fleet down`): finish the upload of any completed pair and
   exit. In-flight cycle is discarded — by design, that's cheap.

Env interface (also the future volunteer-client interface): `SCRIBBLEZ_WORKLOAD=kill_test`,
`TAG`, `THREADS`, workload-specific knobs (`ROLLOUTS`, `TOP_K`, ...), `RCLONE_*` /
storage credentials, `WORKER_ID` (informational, for logs).

Making the entrypoint dispatch on `SCRIBBLEZ_WORKLOAD` from day 1 is the cheap generality
that pays off later: a future GPU workload is a new workload name plus whatever extra the
image needs, not a new scaffolding.

**Registry:** a private repo on docker.io (or GHCR — both fine; docker.io was named as
acceptable and Runpod pulls from any registry given credentials). Push via a small
`py/cloud/build_worker_image.py` that builds the multi-stage image and pushes
`:latest` + a content tag.

## Data plane

**Recommended store: Cloudflare R2** (S3-compatible).

- Zero egress fees — the "do we pay to bring data home?" question disappears structurally,
  not just at today's volume.
- Provider-neutral: the same bucket serves Runpod workers today, GCP workers tomorrow,
  volunteers later. Not tied to the compute vendor.
- Cost at this volume is pocket change: ~15 GB per laptop-week of data at $0.015/GB-month
  (first 10 GB free).

Alternatives: GCS if the fleet ends up on GCP anyway (egress back home at ~$0.12/GB is ~$2
per laptop-week — negligible at this data size, but it grows with future workloads);
Backblaze B2 (free egress up to 3× storage). Runpod's own network-volume storage is
rejected: it ties data to one provider and one datacenter, and doesn't serve the volunteer
future.

Layout:

```
kill_test/<tag>/params.json
kill_test/<tag>/slogs/<ns-timestamp>.slog
kill_test/<tag>/slogs/<ns-timestamp>.sobs
```

**Sync back:** `py/scripts/cloud_sync.py -t <tag>` — an `rclone copy` of the tag prefix
into `<mount>/kill_test/<tag>/slogs/`, downloading only pairs where both files exist, with
an optional `--watch` loop. Local and cloud-generated data for the same tag land in the
same directory; `kill_test.py` runs on the union with no changes. This answers the
"where does data live?" question: the bucket is the durable archive, the laptop holds a
synced copy, and analysis ergonomics are untouched.

## Fleet CLI

`py/cloud/` package, host-side or in-container (it only talks to HTTP APIs):

- `fleet.py up -n 8 --vcpus 32 --provider runpod --workload kill_test -t hello [knobs]`
  — launch N workers with the given env config.
- `fleet.py status` — list running workers, uptime, spend rate, cycles uploaded (cheap to
  derive from bucket listing).
- `fleet.py down [--tag hello]` — terminate; workers flush on SIGTERM.
- `fleet.py logs <worker>` — tail one worker's logs.

Provider adapters behind a small interface (`launch(n, spec, env)`, `list()`,
`terminate(ids)`):

- **Runpod adapter:** the `runpod` Python SDK / REST API creates CPU pods from the worker
  image with env vars set. Per-second billing; free ingress/egress. `runpodctl` remains
  available for ad-hoc ssh.
- **GCP adapter:** `gcloud compute instances create-with-container` on Container-Optimized
  OS spot VMs — the VM boots straight into the same worker container; preemption is just
  SIGTERM, which the worker already handles. No GCP-specific image work.

Secrets (Runpod API key, registry pull creds, R2 keys, GCP project) live in an untracked
`<mount>/cloud/credentials.json`; `fleet.py up` injects the worker-facing subset (R2
write credentials only) as pod env vars.

Prefer fewer, bigger nodes (16–32 vCPU): the engine already scales by threads within one
process, fewer image pulls, fewer moving parts in `status`.

## Economics

Target: one laptop-week ≈ 24 × 7 × 28 = **~4,700 vCPU-hours** per experiment.

| Option | ~$/vCPU-hr | Cost per laptop-week | Notes |
|---|---|---|---|
| Runpod CPU pods | ~$0.02–0.05 (console has exact per-class rates) | ~$100–200 | per-second billing, free egress, no spot tier for CPU |
| GCP c3d/c2d spot | ~$0.008 (c3d-standard us-central1, mid-2026) | **~$40** | highcpu shapes slightly cheaper; preemption is a non-event here |
| GCP on-demand | ~$0.045 | ~$210 | no reason to pay this given spot tolerance |

Storage/transfer is noise at this volume: ~15 GB/laptop-week; R2 storage ≈ $0.25/month,
egress $0.

Read on the numbers: **GCP spot is ~3–4× cheaper for pure CPU**, because spot discounts
(~80%+) exist there and Runpod's pricing edge is GPUs, not CPUs. But the absolute delta is
~$100–150 per experiment-week. Given the Runpod preference and that the workload is the
ideal spot candidate anyway, the pragmatic plan is: **build the Runpod adapter first**
(preferred provider, simplest API, and the GPU future lands there), keep the adapter
interface honest, and add the GCP spot adapter when either (a) CPU spend starts to matter
or (b) Runpod CPU capacity/availability disappoints. The GCP adapter is small — one
`gcloud` invocation template — so deferring it costs little.

Throughput example: 10 × 32-vCPU nodes ≈ 320 vCPU ≈ 11× the laptop → a laptop-week of data
in ~15 wall-clock hours, for roughly $40 (GCP spot) / $150 (Runpod).

## Future: GPU workloads

Everything above is GPU-ready by construction:

- The fleet CLI grows a `--gpu <type>` spec; the Runpod adapter passes it through (this is
  where Runpod's pricing is actually competitive).
- Worker images stay per-purpose: the kill-test worker stays slim; a training/GPU worker
  derives from a CUDA runtime base and reuses the same entrypoint contract
  (`SCRIBBLEZ_WORKLOAD` + env knobs + bucket upload).
- The bucket layout generalizes to `<workload>/<tag>/...`.

## Future: volunteer compute

Not built now, but the day-1 choices keep the door open:

- A volunteer is, to first order, someone running `docker run scribblez/worker` with a
  participation token — the worker image, env-var contract, and "upload complete pairs to
  an ingest point" flow are already exactly that shape.
- The image never contains lexica (fetched from public upstream at start), so it can be
  made publicly pullable without redistributing wordlists.
- The one piece that must change for volunteers is credentials: raw bucket write keys can't
  be handed to strangers. The worker's upload step should therefore be a single seam
  (one `push_results()` call) so that swapping `rclone`-with-keys for
  HTTPS-ingest-with-token (presigned URLs from a small ingest service, plus server-side
  validation of uploaded pairs) touches one function, not the workload logic. Untrusted
  data will need a validation/quarantine pass at ingest — that's deliberately a
  later-problem, but the bucket-centric design is what makes it addable.

## Day-1 checklist

1. `docker-setup/worker/Dockerfile` (multi-stage: build in dev image, slim runtime) +
   `py/cloud/build_worker_image.py` (build & push).
2. `worker_entrypoint.py`: dep fetch, params manifest, generate-upload loop, SIGTERM flush.
3. R2 bucket + credentials file convention.
4. `py/cloud/fleet.py` with the Runpod adapter.
5. `py/scripts/cloud_sync.py` (bucket → local mount, pairs-only, `--watch`).
6. Smoke test: 1 small pod, 1 cycle, verify the pair appears locally and `kill_test.py`
   consumes the merged directory.

Decisions to confirm before building: registry (docker.io private vs GHCR), R2 vs
GCS/B2, and whether a GCP spot adapter is wanted in the first pass or deferred.

# Work on rented machines

How Scribblez runs workload roles (from the workload registry,
scribblez/workloads/) on machines it does not own -- rented from a cloud
provider by the dashboard, or an operator's own reached over ssh -- with
results flowing back to the local mount where analysis runs unchanged. The
machine model itself (renting, idling, terminating, the provider seam) is
[cloud_machines.md](plans/cloud_machines.md); this document is the
plumbing every remote worker shares: the image, the bundles, the bucket.

Generation-style roles distribute trivially: cycles are embarrassingly
parallel, output files are uniquely named and land atomically (so batches from
any number of machines merge by copy, and a killed worker loses at most its
in-flight cycle), data volumes are small, and runtime deps are light and
fetched from public upstreams. The trainers distribute too, through the
bucket: generations (position_eval) or the pair store (move_set_eval) in,
exports and checkpoints out ([cloud_training.md](plans/cloud_training.md)).
A move-set-eval trainer prunes its exports and, when its run completes,
retires its training pairs bucket-side as well; the controller's `models/`
copy mirrors the bucket (`cloud_sync`), its `slogs/` copy is the archive
and stays.

## Architecture

Three stores decouple everything: a **container registry** (stable worker
image), the **R2 bucket** (code bundles outbound, results inbound), and the
**local mount dir** (analysis home).

```
 dev container                                    rented machine
 ─────────────                                    ──────────────
 py/build.py -b            (per-arch binaries)
 deploy (dashboard) ──────────────────────────►  R2: bundles/<id>/bundle-<arch>.tar.gz
 build_and_push_worker_image.py  ──(deps only, rare)───►  docker.io: worker images
 dashboard: rent, then start a slot ──(ssh)────►  docker run <worker image>
                                                   └─ bootstrap.py: pull bundle for its
                                                      CPU arch, exec worker entrypoint
                                                   └─ loop: generate cycle, upload
                                                          │
 cloud_sync.py (dashboard) ◄──────────────────  R2: <workload>/<tag>/...
      │
 <mount>/tags/<workload>/<tag>/data/   ◄── analysis runs here, locally, as always
```

Principles:

1. **The registry image is dependency-only and stable.** It changes only when
   worker *dependencies* change — never for code iteration — so a machine
   pulls it once.
2. **Code travels as bundles through R2, not through Docker or git.** What
   runs remotely is bit-for-bit what was last built locally (uncommitted
   changes included, flagged `-dirty` in the bundle id). Workers need no repo
   credentials and never compile.
3. **Workers are stateless and disposable.** A worker fetches its bundle and
   data deps and loops until terminated. The bucket is the durable archive;
   the local mount holds a synced copy for analysis. A machine can be stopped
   or terminated at any time; a trainer resumes from its checkpoint.

## The pieces

- **Worker images** (`docker-setup/worker/`,
  `./build_and_push_worker_image.py`): dependency-only runtime images with
  the baked-in `bootstrap.py` entrypoint — no repo code, binaries, or lexica.
  Two come out of the one Dockerfile, one per *runtime* a role declares
  (`RoleSpec.runtime`, `py/cloud/runtime_abi.py`): the **engine** image
  (numpy, the C++ and NVIDIA runtime libraries) every generator and match-eval
  slot runs, and the **torch** image, a further stage adding PyTorch and the
  training stack in a venv, for the train roles; it is pushed under the
  engine image's tag with `-torch` appended, so one credential names both.
  They supply the runtime every bundle links against, so they are a matched
  set with the dev image: a compiler upgrade there (gcc-16's newer libstdc++,
  say) leaves every bundle unable to load until they are rebuilt, and torch
  is pinned to the dev image's own version. The registry's owner rebuilds
  them after such a dev-image change; every push records what its image
  provides, so a deploy from a dev container an image has fallen behind
  refuses instead of shipping binaries no worker can start, and names the
  script to run. A rented machine pulls both images at first boot; an
  operator's own machine pulls the one a slot's role needs when the dashboard
  next creates a container. GPU roles included: the engine image carries
  TensorRT's 2 GB builder resource so a worker can turn an ONNX export into
  an engine plan, which a machine with its own GPU must do for itself -- a
  plan is valid only for the compute capability it was built on, so the
  controller cannot build one for it. Such a container is run with `--gpus
  all`, which needs the NVIDIA container toolkit (preinstalled on the image
  rented machines boot from).
- **Bundles** (`py/cloud/bundles.py`, `./py/scripts/cloud_push_binaries.py`):
  the engine builds once per supported CPU microarchitecture
  (`py/build.py --build-for-all-archs`); a push uploads one tarball per arch
  (binaries + the arch-independent `py/` tree). Deploying is automatic --
  `deploy_current_tree` builds every arch and pushes unless the bucket's
  LATEST already carries this tree, and a task pins the result when its first
  remote slot starts, so no fleet runs code you did not deploy because you
  forgot to. Its staleness test is the manifest's `source_hash` (a digest of
  the files a bundle ships), since the bundle_id is deliberately fresh on
  every push and a `-dirty` git sha says a tree changed without saying into
  what. The explicit push CLI remains for pushing a bundle without launching
  anything. At container start `bootstrap.py` detects the machine's arch,
  downloads the matching tarball (generic `x86-64` fallback), unpacks it, and
  execs the bundle's worker entrypoint — so even the worker-loop logic is
  iterable without touching the image.
- **Worker entrypoint** (`py/cloud/worker_entrypoint.py`): configured by
  environment variables (see its docstring). Refuses to start when the
  environment names a workload parameter its bundle's schema does not know --
  a bundle behind the controller would otherwise ignore the parameter and
  deliver data silently unlike its fleetmates'. Dispatches to the (workload,
  role) runner from the workload registry, fetches the runner's declared data
  deps (`py/cloud/worker_deps.py`: lexica and Macondo tables from their public
  upstreams; for a train role also the eval datasets, from the bucket's
  `deps/` prefix at the content version its bundle's manifest names, which
  the deploy uploads once per version rather than 40 MB into every per-arch
  tarball), writes a provenance manifest to the bucket, and loops the
  runner's cycle, delivering whole output files through the results sink
  (`py/cloud/sinks.py`, which orders uploads so the bucket only ever presents
  complete outputs). SIGTERM flushes completed output and exits non-zero, so
  a stop is never mistaken for the role's terminal condition.
- **Out-of-tag inputs** (`RoleSpec.inputs`): a role whose slots read a file
  outside their own tag -- the move-set-eval generator's teacher, a
  position_eval export -- names it under a tag-relative key. A local worker
  reads the source in place; for a remote slot the controller stages a copy
  where the slot will look before it needs it (the tag's bucket prefix for a
  bucket-delivering slot, pushed into the container over the control link
  otherwise), and the runner resolves it through `workloads.base.resolve_input`.
- **Results sync** (`./py/scripts/cloud_sync.py`): pulls the workload's
  inbound bucket prefixes into `<mount>/tags/<workload>/<tag>/`, merging with
  locally generated data for the same tag; for a tag whose trainer delivers
  through the bucket, its outputs too (`--trainer-outputs`, which the
  dashboard passes for such a tag). Bucket-delivering slots only -- a slot
  on an operator's own machine has its results read straight out of its
  container over the control link (`py/cloud/ssh_transfer.py`), which is
  both faster and one fewer place for them to be. Prefixes the controller
  host itself maintains in the bucket are deliberately not pulled.
- **Credentials** (`<mount>/cloud/credentials.json`): one operator-filled file
  (template from setup_wizard.py; validated end-to-end by
  `cloud_check_credentials.py`): the R2 bucket, the image registry with a
  read-only pull token, and the provider's access key. Workers receive only
  the R2 subset, through their container's environment; a rented machine
  receives the pull token at first boot.
- **Providers** (`py/cloud/providers/`): what the dashboard asks of a cloud
  to rent a machine (catalog, launch, describe, stop, start, terminate,
  refusal), answered for AWS in `aws.py`; the account-side one-time setup is
  `./py/scripts/aws_setup.py`.

## Economics

R2 storage and transfer cost is noise (zero egress fees). Compute is the
machine's hourly rate for as long as it is up; the dashboard stops a machine
nothing has run on for ten minutes and shows what a task's machines bill.
The first measured run (2026-09-16) is recorded in
[cloud_machines.md](plans/cloud_machines.md).

## Futures

- **Volunteer compute.** A volunteer is, to first order, someone running the
  worker image (publicly pullable; it contains no redistribution-restricted
  data) with a participation token. The one necessary change is credentials —
  raw bucket keys can't go to strangers. Uploads already funnel through the
  results sink, which is the seam where rclone-with-keys becomes
  token-authenticated HTTPS ingest with server-side validation.

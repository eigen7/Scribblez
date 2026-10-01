# Work on remote machines

How Scribblez runs workload roles on machines other than the controller's:
machines the dashboard rents from a cloud provider, and the operator's own
machines reached over ssh. Results flow back to the local mount, where
analysis runs unchanged.

This document covers the plumbing every remote worker shares: the worker
image, the code bundles, and how results come back. The machine model (renting,
idling, terminating, the provider seam) is in
[plans/cloud_machines.md](plans/cloud_machines.md), and operating machines
from the dashboard is in [master_dashboard.md](master_dashboard.md).

Generation-style roles distribute trivially. Their cycles are embarrassingly
parallel; output files are uniquely named and land atomically, so batches
from any number of machines merge by copying and a killed worker loses at
most its in-flight cycle; data volumes are small; and runtime deps are light
and fetched from public upstreams. A position_eval trainer distributes too:
its data home (the tag's data plane, beside it) assembles its own
generations, the controller relaying in the chunks of generators elsewhere,
and its exports, records and checkpoint come out over ssh, collected like any
slot's records and installed under the cursor rule
([plans/cloud_training.md](plans/cloud_training.md)). A
move_set_eval trainer runs only locally.

## Architecture

Three stores decouple everything: a **container registry** (the stable worker
images), the **R2 bucket** (code bundles and data deps outbound), and the
**local mount dir** (where results are collected and analysis runs).

```
 dev container                                    remote machine
 ─────────────                                    ──────────────
 dashboard: deploy (build + push) ─────────────►  R2: bundles/<id>/bundle-<arch>.tar.gz
 build_and_push_worker_image.py ──(deps only, rare)─►  registry: worker images
 dashboard: start a slot ──────────(ssh)───────►  docker run <worker image>
                                                   └─ bootstrap.py: fetch the bundle for its
                                                      CPU arch, exec the worker entrypoint
                                                   └─ loop: run a cycle, deliver output
                                                      into the container's own tree
                                                          │
 dashboard: collect ◄──────────────(ssh)──────────────────┘
      │
 <mount>/tags/<workload>/<tag>/data/   ◄── analysis runs here, locally, as always
```

Principles:

1. **The registry images hold dependencies only, and rarely change.** They
   are rebuilt only when worker *dependencies* change, never for code
   iteration, so a machine pulls them once.
2. **Code travels as bundles through R2, not through Docker or git.** What
   runs remotely is bit-for-bit what was last built locally, uncommitted
   changes included (flagged `-dirty` in the bundle id). Workers need no repo
   credentials and never compile.
3. **Workers are stateless and disposable.** A worker fetches its bundle and
   data deps and loops until terminated. Its output is collected into the
   local mount, the durable copy. A machine can be stopped or terminated at
   any time; a trainer resumes from the controller's copy of its checkpoint.

Every slot's output (a generator's data, its stats and params, and a
trainer's exports, records and checkpoint) is collected over ssh straight
out of its container, wherever it runs (see "Results collection" below).
No result travels through the bucket.

## The pieces

### Worker images

`docker-setup/worker/`, built and pushed by `./build_and_push_worker_image.py`
on the host. These are dependency-only runtime images with the baked-in
`bootstrap.py` entrypoint: no repo code, binaries or lexica. One Dockerfile
produces two images, one per *runtime* a role declares (`RoleSpec.runtime`,
`py/cloud/runtime_abi.py`):

- the **engine** image (numpy, the C++ and NVIDIA runtime libraries,
  TensorRT's builder), which every generator and match-eval slot runs;
- the **torch** image, a further stage adding PyTorch and the training stack,
  for the train roles. It is pushed under the engine image's tag with `-torch`
  appended, so one credential names both.

The images supply the runtime every bundle links against, so they must match
the dev image. A compiler upgrade there (a newer libstdc++, say) leaves every
new bundle unable to load until the images are rebuilt, and torch is pinned
to the dev image's version. The registry's owner rebuilds them after such a
dev-image change. Every push records what its image provides, so a deploy
from a dev container that an image has fallen behind is refused, naming the
script to run.

A rented machine pulls both images at first boot; one of your own pulls the
image a slot's role needs when the dashboard next creates a container.

GPU roles run on the engine image too. It carries TensorRT's 2 GB builder
resource so the worker can turn an ONNX export into an engine plan itself: a
plan is valid only for the compute capability it was built on, so the
controller cannot build one for another machine's GPU. Such a container runs
with `--gpus all`, which needs the NVIDIA container toolkit (preinstalled on
the image rented machines boot from).

### Bundles

`py/cloud/bundles.py`. A bundle is one tarball per CPU microarchitecture,
each holding that arch's engine binaries plus the arch-independent `py/`
tree. The archs are the ones the task's machines report: a rented machine's
from the provider catalog, one of your own asked once, through the worker
image's compiler, at its first slot start.

Deploying is automatic. When a task's first remote slot starts,
`deploy_current_tree` builds those archs and pushes a bundle unless the
bucket's `LATEST` already carries this tree for them, and the task pins the
result. No fleet runs code you forgot to deploy. A later slot whose arch the
pinned bundle lacks triggers a rebuild with its arch added.

The "already deployed?" test is the manifest's `source_hash`, a digest of the
files a bundle ships. The bundle id cannot serve: it is deliberately fresh on
every push, and a `-dirty` git sha says the tree changed without saying into
what.

At container start, `bootstrap.py` detects the machine's arch, downloads the
matching tarball (falling back to generic `x86-64`), unpacks it, and execs the
bundle's worker entrypoint, so even the worker-loop logic can change without
touching the image.

To push a bundle without launching anything, build the archs
(`py/build.py --archs <a,b>`) and run `./py/scripts/cloud_push_binaries.py
--archs <a,b>`.

### Worker entrypoint

`py/cloud/worker_entrypoint.py`, configured entirely by environment variables
(listed in its docstring). It:

1. refuses to start when the environment names a workload parameter its
   bundle's schema does not know (a bundle behind the controller would
   otherwise ignore the parameter and silently produce data unlike its
   fleetmates');
2. dispatches to the (workload, role) runner from the workload registry;
3. fetches the runner's declared data deps (`py/cloud/worker_deps.py`):
   lexica and Macondo tables from their public upstreams, and for a train
   role the eval datasets from the bucket's `deps/` prefix, at the content
   version the bundle's manifest names (uploaded once per version rather than
   copied into every per-arch tarball);
4. writes a provenance manifest to the bucket;
5. loops the runner's cycle, delivering whole output files through the
   results sink (`py/cloud/sinks.py`), which orders uploads so the bucket
   only ever presents complete outputs.

SIGTERM flushes completed output and exits non-zero, so a stop is never
mistaken for the role's terminal condition.

### Out-of-tag inputs

`RoleSpec.inputs`. A role that reads a file outside its own tag (the
move_set_eval generator's teacher, a position_eval export) names it under a
tag-relative key. A local worker reads the source in place. For a remote
slot, the controller pushes a copy into the container over the control link
right after creating it, and the runner waits for it there
(`workloads.base.resolve_input`).

### Results collection

Every ssh worker delivers into its own container, and each reconcile pass
collects from it over the control link (`py/cloud/ssh_transfer.py`): its data
(`WorkloadSpec.collected_dirs`) into `<mount>/tags/<workload>/<tag>/data/`,
merging with locally generated data for the same tag, its records (stats,
params) and a trainer's outputs (records, exports, its checkpoint and cursor
as a state pair). A generator cannot count a store it never sees whole, so a
target on the store's size (blind_spots' positions, move_set_eval's pairs)
is enforced by the workload's scheduler tick on the controller. For a tag
whose data home is an ssh machine, each pull from its trainer is followed by
two more legs: the chunks collected from its other generators are relayed on
into the home's staging, and the home's complete generations are copied
here and then acknowledged there, which is what lets its trainer evict them.
The controller's tree is thereby the durable copy of every tag. A trainer's
checkpoint and cursor are installed under the cursor rule
(`generational/state_pair.py`); a new trainer container is seeded with the
controller's copy, and a new remote home's volume with its window of
generations and ingest ledger, before the container starts. When a trainer
leaves a remote home, its generations, staged chunks and ledger are swept
here first. So a tag can move to any machine and pick up where it stopped.

### Credentials

`<mount>/cloud/credentials.json` is one operator-filled file (template from
`setup_wizard.py`, validated end-to-end by
`./py/scripts/cloud_check_credentials.py`): the R2 bucket, the image registry
with a read-only pull token, and the provider's access key. Workers receive
only the R2 subset, through their container's environment; a rented machine
receives the pull token at first boot.

### Providers

`py/cloud/providers/` is what the dashboard asks of a cloud to rent a machine:
catalog, launch, describe, stop, start, terminate, and how a refusal is
reported. AWS is implemented in `aws.py`; its one-time account setup is
`./py/scripts/aws_setup.py`.

## Economics

R2 storage and transfer cost is negligible (no egress fees). Compute is the
machine's hourly rate for as long as it is up. The dashboard stops a machine
nothing has run on for ten minutes and shows what a task's machines bill. The
first measured run is recorded in [plans/cloud_machines.md](plans/cloud_machines.md).

## Future: volunteer compute

A volunteer is, to first order, someone running the worker image with a
participation token. The image contains no redistribution-restricted data, so
it could be made publicly pullable. The one necessary change is credentials: raw bucket keys cannot go to
strangers. Uploads already funnel through the results sink, which is where
direct bucket writes would become token-authenticated HTTPS ingest with
server-side validation.

# Plan: the trainer on rented GPUs

**Status: landed, then partly superseded.** Everything this plan built for
the trainer has landed: the trainer I/O contract (PR 2), the GPU runtime
(PR 1), the controller's bucket legs (3a: generation publish, trainer-output
pull, controls push), the trainer's artifacts through its sink (3b), and a
cloud train slot (3c). The slot and its machine model (the "machines, slots"
section below) are superseded by [cloud_machines.md](cloud_machines.md):
Runpod gave way to AWS instances driven as ssh machines, and the Runpod
client, the `cloud` worker kind and the pod forms described here are
deleted. The trainer contract and the bucket legs carried over unchanged.

**Goal.** Run several position_eval training runs at once on rented GPUs, so
A/B experiments (trunk, optimizer, loss weights, ...) stop queueing behind
the one laptop GPU. Each run stays first-class in the local dashboard: in the
tag list, with its Loss / Match / Positions / Stats tabs populated,
comparable against any other run in the Loss tab overlay, steerable from the
Controls tab, and archived durably. Running with the laptop off is *not* a
goal; cloud generators and the ssh match-eval machine already depend on the
dashboard's reconcile loop.

**Decision.** The trainer becomes a location-independent worker that talks
to the bucket at generation boundaries; the dashboard's reconcile loop stays
the one controller of a tag. The trainer stops writing `dashboard.db` and
instead delivers immutable, generation-keyed records that the controller
ingests, the same split match eval already uses.

## The survey, and what each finding became

The plan started from a survey of the code as it then stood. Each finding
below is followed by what was built for it.

- **Trainer outputs bypassed the results sink.** `dashboard.db`, the ONNX
  exports, the checkpoint, `train_state.json` and generation eviction were
  direct filesystem writes under the tag root; only the stats record went
  through `ctx.sink`, and the checkpoint save was not atomic. → The trainer
  I/O contract (below) and an atomic checkpoint save
  (`generational/checkpoint.py`).
- **The trainer needs the lexicon at process start.** Opening the FFI
  session loads `/workspace/mount/lexica/NWL23.kwg`, and a missing lexicon is
  a `std::terminate` across the C ABI, not a catchable error. (Macondo's
  strategy tables are needed only by the agent factories.) → The train role
  declares `deps` that fetch it.
- **The eval datasets were not in the bundle.** The quality and placement
  evals read `positions/NWL23/position-eval-test-dataset{,-large}` (about
  40 MB, git-tracked); without them the trainer logged a line and trained on
  without its `eval_*` metrics, which wastes a run meant for an A/B. → The
  datasets travel as a deps object, and the trainer refuses to start without
  them.
- **The worker image could not host torch**: apt `python3` and numpy on
  Ubuntu 24.04, no pip, no torch stack, one image and one ABI record. → A
  second image (below).
- **Nothing reaches into a pod**; a pod must push. **The scheduler is a
  plain function** whose `mirror` hook moved cloud-origin chunks within the
  bucket but uploaded nothing and never mirrored `manifest.json`, so the
  bucket held chunks by generation but not completion. → Generation publish
  on completion.
- **Downward data paths existed in two places only**: the bundle fetch and a
  worker's own stats record. `cloud_sync.py` pulled `sync_data_dirs`,
  `stats/` and `params/` only. → The trainer-output pull.

## Measured numbers

From the live `transformer-face-up` tag (RTX 5000 Ada laptop GPU, 28 cores),
per 20,000-game generation:

| Quantity | Value |
|---|---|
| Generation, local generator at 24 threads | ~1.5 s per 1000-game chunk, ~30 s per generation |
| Trainer, one epoch over the 4-generation window (79,872 rows) | ~99 s train + ~3 s eval |
| One generation on disk (20 `.slog` chunks) | 12 MB |
| One ONNX export | ~40 MB (34 GB over 865 generations) |
| Rolling checkpoint | 117 MB |
| `dashboard.db` | 3.5 MB |
| Match eval on the ssh machine, 400 games | ~52 s, every 5th generation |

Locally the trainer is the bottleneck by 3×. Every leg through the bucket is
a latency question, never a bandwidth one. Runpod GPU offers from its live
catalog on 2026-09-04: RTX 4090 $0.74/h (secure, 8 vCPU), A6000 $0.53/h (9),
L40S $1.09/h (16), A100 SXM 80GB $1.59/h (16). At planning time no cloud
generator had ever run against a tag on this mount, so cloud per-vCPU
generation throughput was uncalibrated.

## Design

```
 generators (local/ssh/cloud) ──chunks──► staging ──scheduler ingest──► generations/gen_N
                                                          │ on completion, for a tag whose
                                                          │ trainer delivers through the
                                                          │ bucket: publish gen_N
                                                          ▼
                                     R2: position_eval/<tag>/generations/gen_N/{*.slog, manifest.json}
                                                          │ poll + pull (remote trainer)
                                                          ▼
                                     remote trainer: window on scratch disk, train, checkpoint
                                                          │ deliver per generation, record last
                                                          ▼
                     R2: <tag>/models/model_epoch_N.onnx, <tag>/records/gen_N.{json,npz},
                         <tag>/checkpoints/model.pt, <tag>/train_state.json
                                                          │ sync watcher (as for staging)
                                                          ▼
                     controller: ingest tick writes metrics/preds into dashboard.db (sole writer),
                                 scheduler reads the cursor, match dispatch sees the export
 Controls tab ──► controller writes <tag>/controls.json ──► trainer reads it each generation
```

Generation for a remotely trained tag comes from wherever the operator
attaches generators, exactly as before. Match eval stays a local or ssh slot,
so it never competes with training for a GPU.

### The trainer's I/O contract (every kind)

The trainer's outputs are immutable, generation-keyed records delivered
through the sink, and the controller ingests them into the database. Its
operator inputs (learning rate, loader workers, torch threads) arrive as a
small `controls.json` that the controller writes and the trainer reads once
per generation. This makes the trainer location-independent, and because it
applies to the local kind too, there is one trainer code path and one
database writer for every kind. As built (`generational/records.py`,
`generational/train_ingest.py`):

- **Per generation the trainer delivers** the ONNX export, the rolling
  checkpoint, `train_state.json`, and under `records/` the generation's
  `gen_NNNNNN.npz` (the Positions tab predictions) and last of all
  `gen_NNNNNN.json` (the metrics row and the control events since the last
  record). Each object is atomic on its own (rclone exposes only whole
  objects; the local sink renames). The plan had a separate
  `commit_NNNNNN.json` marker; as built, the generation record is written
  last and is itself the commit marker. The checkpoint is copied, not moved,
  since the trainer resumes from it. Delivery runs on its own thread so the
  next generation trains while the previous one uploads.
- **The controller's ingest tick** (`RoleSpec.ingest`, run by the reconcile
  loop like match eval's dispatch) consumes committed generations in order
  into `dashboard.db`: the run's config on the first, metrics and
  predictions on each. It is idempotent, keyed by generation. The dashboard
  shows a generation one reconcile pass after it lands, for local and remote
  trainers alike.
- **Controls.** The controller writes `controls.json` under the tag whenever
  the Controls tab changes a value, and pushes it to the tag's bucket prefix
  for a trainer that delivers through the bucket. The trainers' CPU
  controller reads it instead of the database's control table.

### The remote trainer

- **Inputs.** For each generation it needs, the trainer polls the bucket for
  `generations/gen_N/manifest.json`, and once the manifest says complete,
  pulls the directory (12 MB) onto scratch disk; local eviction beyond the
  window is unchanged. A fresh machine first pulls the latest committed
  checkpoint and `train_state.json`, then the window's generations, so its
  first epoch is the same epoch a local resume would run. The machine holds
  no state worth keeping: restoring is the same pull as a first start.
- **Publishing generations.** For a tag whose trainer delivers through the
  bucket, the scheduler's completion step publishes each generation: an
  rclone copy of the local generation directory to its bucket prefix
  (chunks already moved there by the mirror are skipped by size), then the
  manifest. Local- and ssh-origin chunks are uploaded this way; ingest and
  the ledger are unchanged.
- **Pacing.** The trainer's cursor reaches the scheduler through
  `train_state.json` on the existing sync watcher (30 s). With `open_ahead=4`
  and about 60 s per generation on a desktop GPU, generation stays 3 to 4
  generations ahead and the bucket handoff is hidden. Acceptance: the
  trainer's per-generation wait (cycle time minus `train_s` + `eval_s` in its
  stats record) stays under 10% of the cycle.
- **Pulling outputs.** `cloud_sync.py --trainer-outputs` pulls the tag's
  `records/`, `models/`, `checkpoints/` and `train_state.json` alongside the
  staging, stats and params directories, with `--size-only` for the
  immutable prefixes (an S3-style listing carries no modtime, so a plain copy
  would HEAD every unchanged export on every pass).
- **Bucket layout** follows the existing convention: root-level tag files and
  `data/` subdirectories share the tag prefix (`<tag>/models/`,
  `<tag>/records/`, `<tag>/generations/gen_N/`).

### The torch worker image

A second worker image, the engine image's tag with `-torch` appended, built
as a further stage of `docker-setup/worker/Dockerfile`. It adds
`python3-venv` and `python3-dev` (`torch.compile` needs the headers; the
compiler is already there), torch from the cu129 index as the dev image
installs it, and onnx, onnxscript, natsort, schedulefree and tqdm. Ubuntu
24.04's interpreter is externally managed, so the stage installs into a venv
that sees the system packages. torch wheels vendor their own CUDA libraries,
so the image's `libcudart` stays the one TensorRT needs. The engine image
keeps its size.

A role declares `RoleSpec.runtime` (`engine` or `torch`), and the container
creation sites pick the image from it. `build_and_push_worker_image.py`
builds and pushes both, and the ABI record (`cloud/runtime_abi.py`) is keyed
by runtime, so the check before a launch compares the image that launch will
use. match_eval stays on the engine runtime, so the ssh machine's image does
not grow.

### Runtime data on a remote machine

- The train role declares `deps`: the lexicon fetch
  (`worker_deps.fetch_lexicon`) plus the eval datasets, fetched from the
  bucket object `deps/positions-<content digest>.tar.gz`
  (`worker_deps.fetch_eval_positions`), which the deploy path pushes when
  absent. This leaves the bundle format and the image-baked bootstrap alone:
  the bundle stays per-arch code, and a 40 MB dataset does not ride every
  deploy.
- The trainer refuses to start if the eval datasets or the lexicon are
  missing, on every kind. There is no opt-out.

## Sequence

0. **Calibrate and probe** (no repo code, under $2): one CPU pod for ten
   minutes against a throwaway tag, for games/s per vCPU (how many CPU pods a
   remotely trained tag needs); one GPU pod by hand on a stock torch image,
   to confirm that a stop delivers SIGTERM to PID 1 with a grace period that
   generation-boundary checkpointing can live with.
1. **PR 1, the runtime.** The torch image, the venv, `RoleSpec.runtime`,
   image selection at both creation sites, the ABI record keyed by runtime;
   the eval-dataset deps object and fetch; the train role's `deps`; loud
   startup checks; `SCZ_DEVICE` allow-listed in the worker entrypoint.
   Verified by running the unchanged trainer inside the torch image on the
   laptop (`docker run --gpus all`) against a local tag.
2. **PR 2, the trainer I/O contract**, local kind only: records out and the
   controller's ingest tick, `controls.json` in, the atomic checkpoint save,
   the commit ordering. Acceptance: a short run through the new path produces
   a `dashboard.db` identical to the old direct-write path. No cloud code.
3. **PR 3, the trainer remote.** Generation publish on completion; the
   trainer's input adapter (poll, pull, restore); output delivery through the
   sink; the sync watcher's new prefixes; the controls push; and a cloud
   train slot. Verified by a real remote run of about 20 generations against
   a local run of the same params: identical metric schema, the wait fraction
   under the acceptance line, restore after a deliberate stop.
4. **Later, separately:** a per-tag generator autoscaler (below); an
   always-on controller host if unattended runs become a goal (the controller
   is one Tornado process plus the mount, so moving it is deployment, not
   code); interruptible trainer machines once restore is proven cheap.
   (Spot machines landed under [cloud_machines.md](cloud_machines.md).)

## Machines, slots and utilization

Added after discussing where the money goes and how independently launched
tags should share hardware. Its conclusion, that a GPU box should be a
*machine* hosting slots rather than a slot itself, is what
[cloud_machines.md](cloud_machines.md) built, on AWS instances over ssh
rather than Runpod pods.

**Where the cost is in the HastyBot regime.** Per 20,000-game generation,
taking a cloud vCPU as half the laptop's per-thread speed and a desktop 4090
as twice its GPU:

| Resource | Per generation | Cost |
|---|---|---|
| Generation, HastyBot vs HastyBot | ~1400 vCPU-s | ~$0.023 at $0.06 per vCPU-hour |
| Training, 4090 fully busy | ~50 GPU-s | ~$0.010 at $0.74/h |
| Training, 4090 alone with its 8 vCPUs (GPU ~30% busy) | | ~$0.037 |

Generation is already the larger cost, and the idle-GPU problem is bounded:
perfect sharing saves about $25 over a 1000-generation run. Sharing one GPU
box between two HastyBot tags buys nothing. The box's binding resource is its
vCPUs, so two generation-bound tags on one 4090 produce the same total
generations per hour at the same cost each. The GPU becomes worth sharing
only once generation comes from elsewhere.

So the lever is per tag and local: feed each trainer enough generation. The
signals already exist (the scheduler's gate says generation is ahead; the
trainer's cycle time minus `train_s` + `eval_s` says it is behind), and
generators are stateless workers the reconcile loop already restarts. A
**per-tag generator autoscaler** that holds the trainer's wait fraction near
zero is the cheap form of utilization control, with no coupling between tags.

**The neural-generation regime inverts this.** Once self-play runs neural
agents, inference dominates GPU time by a wide margin and the trainer is the
small consumer. Colocating a tag's generator with its trainer is then natural:
inference fills the GPU between epochs, and the model handoff is a local
file. Cross-tag sharing is still unnecessary, since one tag's self-play
saturates a GPU on its own. The knob is how many GPU machines a tag gets;
utilization within a machine is the batching evaluation service's job.

**One abstraction serves both regimes: machines are resources, tags are
work, and slots bind the two.** The ssh kind already modeled this (one
machine, many containers, from any tags); the Runpod kind conflated them
(one pod was one slot of one tag). With a GPU box as a machine:

- a companion generator is a generate slot placed on the trainer's machine
  (the bundled vCPUs are the cheapest generation compute there is);
- two trainers on one GPU are two train slots on one machine, sharing it by
  process co-residency and CUDA time-slicing with no new abstraction (a
  waiting trainer sleeps and releases the GPU; at about 5 GiB each, a 24 GB
  card holds several);
- gates and pauses become per-slot process control, so a paused slot no
  longer idles a whole box;
- the neural-generation future is a placement choice, not an architecture
  change.

Placement stays manual: with two to four parallel runs an operator beats any
bin-packer. Pacing between tags is the existing per-tag generational gate;
the one automated dimension is the generator autoscaler. Deliberately not
built: a work-queue scheduler that time-slices one trainer process across
tags. Trainer state (optimizer, window, compiled model, checkpoint) makes
switching expensive, and the OS shares processes better than we would.

## Review record

Panel: hidden complexity (session tier), rival designer (codex,
cross-vendor), scope (sonnet), integration (sonnet), each blind to the
others. They reviewed a first draft that hosted the whole run (scheduler,
generator, trainer, match eval) on one "standalone" GPU pod, with a
one-to-one mirror of the tag root pushed to the bucket and a read-only copy
pulled by the dashboard.

**Revised on the rival designer's critique** (serious): a bucket-native
trainer under the existing controller follows the architecture instead of
adding a control plane on the pod. It removed at a stroke the machinery the
other panelists found under the standalone shape: a nested worker
entrypoint, an inline generator refactor, a supervisor with process-group
semantics, a second scheduler and match dispatcher (which, as the
hidden-complexity panelist showed, would have raced the dashboard's own ticks
on a local-kind tag), hosted-tag exclusivity, write guards in a data plane
with no task record, whole-file sqlite replacement, pod volumes with an
unverified stop/start story, and a restore sequence that keyed the window on
`train_state.json` while the checkpoint lagged it (a trainer waiting forever
on a generation the scheduler would never open). It also delivers
independently scalable generation and a free local GPU for match eval in the
first slice. The one property given up, running with the laptop off, was not
a stated goal.

**Adopted from the other critiques:** the atomic checkpoint save; the
checkpoint pushed every generation as one rolling object, with the commit
marker written last; eval datasets fetched as a dependency from a bucket
prefix rather than shipped in a bundle format the image-baked bootstrap
would have had to learn; the ABI record keyed by image; the venv decision
for the torch stage; `--size-only` pulls of the immutable prefixes; probing
the SIGTERM grace period in step 0; dropping a planned scheduler change
(with the cursor seeded, `_next_index` already floors at it).

**Made moot by the revision:** a contended vCPU/GPU budget on one pod;
volume sizing as a dashboard control; slicing match eval out; a local kind of
the standalone role; a second bucket layout.

**Open, human call:** whether unattended operation (laptop off) should become
a requirement. If so, the standalone shape or an always-on controller host is
back on the table. The trainer contract of PR 2 is common to every answer.

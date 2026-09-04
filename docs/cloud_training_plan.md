# Plan: the trainer on rented GPU pods

Status: plan-reviewed draft (critique profile, codex rival seat), revised
after review, then extended with the machines/slots discussion. PR 2 (the
trainer I/O contract) is implemented in the PR that carries this document.
The review record is at the end.

## Goal

Run several position_eval training runs at once on rented cloud GPUs, so
A/B experiments (trunk, optimizer, loss weights, ...) stop queueing behind
the one laptop GPU. Each run must stay first-class in the local master
dashboard: in the tag list, its Loss / Match / Positions / Stats tabs
populated, comparable against any other run in the Loss tab overlay,
steerable through the Controls tab, and archived durably. Unattended
operation with the laptop off is *not* a goal of this slice (it is not how
runs operate today either: cloud generators and the ssh match-eval machine
already depend on the dashboard's reconcile loop).

Today only the `generate` role can run on a cloud pod. The train role is
`kinds=("local",)`, and everything about the cloud path assumes generators.

## What the survey established (verified against the code)

- **Trainer outputs bypass the results sink.** `dashboard.db` (WAL sqlite),
  `models/model_epoch_NNNN.onnx`, `checkpoints/model.pt`, `train_state.json`
  and generation eviction are direct filesystem operations under the tag
  root (`position_eval/trainer.py`, `generational/checkpoint.py`,
  `generational/lifecycle.py`). Only the stats record uses `ctx.sink`.
  `checkpoint.save` is also not atomic (plain `torch.save` onto the live
  path), unlike every other writer in the chain.
- **The trainer needs the engine's lexicon at process start.**
  `get_input_shapes()` opens the FFI session, which loads
  `/workspace/mount/lexica/NWL23.kwg` (a compiled-in directory). A missing
  lexicon is a `std::terminate` across the C ABI, not a catchable error. The
  train role declares no `deps`. Macondo's strategy tables are not needed by
  the trainer (only by the agent factories).
- **The eval datasets are not in the bundle.** The quality and placement
  evals read `positions/NWL23/position-eval-test-dataset{,-large}` under the
  repo root (~40 MB, git-tracked); the bundle ships engine binaries plus
  `py/`. When missing, the trainer logs one line and trains on with the
  `eval_*` metrics absent -- logged, not silent, but a cloud run without its
  eval curves is a wasted run for A/B purposes.
- **The worker image cannot host torch as built.** `python3` and
  `python3-numpy` from apt on Ubuntu 24.04, no pip, no `python3-dev`, no
  torch/onnx/natsort/schedulefree. One image name
  (`registry.worker_image`), one ABI record (`cloud/worker_image.json`,
  a single object), no variant concept; the Dockerfile's comments already
  contemplate a `-gpu` tag split. `bootstrap.py` is baked into the image and
  fetches exactly one per-arch tarball, so anything that changes what a pod
  downloads at start is an image change, not a bundle change.
- **Nothing reaches into a pod.** The Runpod client does create/stop/start/
  delete only; pod specs set no ports, keys or volumes. A pod must push.
- **The scheduler is a plain function** (`tick_for_task(spec, task,
  hooks)`); the dashboard's reconcile loop is its only caller. Its `mirror`
  hook moves cloud-origin chunks within the bucket into the generation
  prefix but uploads nothing (local-origin chunks are skipped) and never
  mirrors `manifest.json`, so the bucket holds chunks by generation but not
  completion.
- **Downward data paths exist in two places:** the bundle fetch in
  `bootstrap.py` and a worker's own stats record (`R2Sink.read_json`).
  `cloud_sync.py` pulls `sync_data_dirs` + `stats/` + `params/` only.
- **GPU cloud plumbing is half there.** `pod_create_spec` builds GPU pod
  bodies; the add-worker form has a GPU instance selector fed by the live
  catalog; `add_cloud` records `gpu_type_id`. No role declares `gpu=True`
  together with `"cloud"`. `SCZ_DEVICE`, which the trainer reads, is not in
  the entrypoint's allow-list of worker variables.

## Measured numbers

From the live `transformer-face-up` tag (RTX 5000 Ada laptop GPU, 28
cores), per 20 000-game generation:

| Quantity | Value |
|---|---|
| Generation, local generator at 24 threads | ~1.5 s per 1000-game chunk, ~30 s per generation |
| Trainer, one epoch over the 4-generation window (79 872 rows) | ~99 s train + ~3 s eval |
| One generation on disk (20 `.slog` chunks) | 12 MB |
| One ONNX export | ~40 MB (34 GB over 865 generations) |
| Rolling checkpoint | 117 MB |
| `dashboard.db` | 3.5 MB |
| Match eval on the ssh machine, 400 games | ~52 s, every 5th generation |

The trainer is the bottleneck locally by 3x. Every leg through the bucket is
a latency question, never a bandwidth one. Runpod GPU offers (live catalog,
2026-09-04): RTX 4090 $0.74/h secure with 8 vCPU, A6000 $0.53/h with 9,
L40S $1.09/h with 16, A100 SXM 80GB $1.59/h with 16. No cloud generator has
ever run against a tag on this mount (no `stats/cloud-*.json` anywhere), so
cloud per-vCPU generation throughput is uncalibrated.

## Design: `train` becomes a cloud-capable role; the controller stays local

The dashboard's reconcile loop remains the one controller of a tag, as it is
for cloud generators and ssh match eval today. The trainer becomes a
self-directing worker that speaks to the bucket at generation boundaries:
it takes complete generations down, and sends generation-keyed outputs up.
Generation for a cloud-trained tag comes from wherever the operator attaches
generators -- local, ssh or cloud pods -- exactly as now. Match eval stays a
local or ssh slot, so it never competes with training for a GPU.

```
 generators (local/ssh/cloud) ──chunks──► staging ──scheduler ingest──► generations/gen_N
                                                          │ on completion, for a tag with
                                                          │ a cloud trainer: publish gen_N
                                                          ▼
                                     R2: position_eval/<tag>/generations/gen_N/{*.slog, manifest.json}
                                                          │ poll + pull (trainer pod)
                                                          ▼
                                            trainer pod: window on scratch disk, train, checkpoint
                                                          │ deliver per generation, commit marker last
                                                          ▼
                     R2: <tag>/models/model_epoch_N.onnx, <tag>/train/gen_N.{json,npz},
                         <tag>/checkpoints/model.pt, <tag>/train_state.json, <tag>/train/commit_N.json
                                                          │ sync watcher (as for staging today)
                                                          ▼
                     controller: ingest tick writes metrics/preds into dashboard.db (sole writer),
                                 scheduler reads the cursor, match dispatch sees the export
 Controls tab ──► controller writes <tag>/train/controls.json ──► trainer reads it each generation
```

### The trainer's I/O contract (all kinds)

The trainer stops touching `dashboard.db`. Its outputs become immutable,
generation-keyed records delivered through the sink, and the controller
ingests them into the database -- the match-eval pattern (`dispatch.tick`
+ `ingest`) applied to the trainer. Its inputs from the operator (the
control table: LR, loader workers, torch threads) become a small
`controls.json` the controller writes and the trainer reads at its natural
cadence, each generation. This is what makes the trainer location-
independent, and it applies to the local kind too, so there is one trainer
code path and one database writer for every kind.

- Per generation the trainer delivers: `train/gen_NNNNNN.json` (the
  metrics row, the adopted controls and control events), `train/
  gen_NNNNNN.npz` (the Positions tab predictions), the ONNX export, the
  rolling checkpoint (written atomically, then copied, not moved, since the
  trainer resumes from it), `train_state.json`, and last of all
  `train/commit_NNNNNN.json` naming the objects above. Everything before the
  marker is per-object atomic (rclone presents only whole objects; the
  local sink renames); the marker gives cross-object atomicity.
- The controller's `ingest` tick (a new `RoleSpec.ingest` hook, run by the
  reconcile loop like `dispatch`) consumes committed generations in order
  into `dashboard.db`: meta and loss weights on the first, metrics and
  preds on each. Idempotent, keyed by generation. The dashboard therefore
  shows a generation a reconcile pass after it lands, for local and cloud
  trainers alike.
- The controller writes `controls.json` under the tag whenever the Controls
  tab changes a value (local kind: the file; cloud kind: also pushed to the
  tag's bucket prefix). `CpuController.refresh` and `build_optim_arm` read
  it instead of the control table.

### The cloud kind

- **Inputs.** Each generation the trainer needs, it lists the bucket prefix
  for `generations/gen_N/manifest.json`, and when the manifest says
  complete, pulls the directory (12 MB) onto scratch disk; local eviction
  beyond the window is unchanged. On a fresh pod it first pulls the latest
  committed checkpoint and `train_state.json`, then the window's
  generations, so the first epoch after a replacement is the same epoch a
  local resume would run. The pod holds no state worth keeping: no volume,
  no pause/resume story beyond stop and start, and restore is the same
  pull as a first start.
- **Publishing generations.** For a tag with a cloud train slot, the
  scheduler's completion step publishes the generation: an rclone copy of
  the local generation directory to its bucket prefix (cloud-origin chunks
  are already there after the existing mirror move and are skipped by
  size), then the manifest. Local-origin and ssh-origin chunks are thereby
  uploaded; nothing about ingest or the ledger changes.
- **Pacing.** The cursor reaches the scheduler through `train_state.json`
  on the existing sync watcher (30 s); with `open_ahead=4` and ~60 s per
  generation on a desktop GPU, generation stays 3-4 generations ahead and
  the bucket handoff is hidden. Acceptance: the trainer's per-generation
  wait (cycle time minus `train_s`+`eval_s` in its stats record) stays
  under 10% of the cycle.
- **Pulling outputs.** `cloud_sync.py` pulls the tag's `train/`, `models/`,
  `checkpoints/` and `train_state.json` alongside `staging/`, `stats/` and
  `params/`, with `--size-only` for the immutable prefixes (an S3-style
  listing carries no modtime, so a plain copy would HEAD every unchanged
  export each pass).
- **Slot.** `train` gains `"cloud"` in its kinds; the add-worker form's GPU
  selector and `add_cloud` already handle the rest. Non-interruptible (it
  is the singleton the run waits on). `SCZ_DEVICE` joins the entrypoint's
  allow-list. The pod's scratch disk is the default 20 GB: exports are
  delivered (uploaded and unlinked), so only the window, the checkpoint and
  the current export live there.
- **Bucket layout** follows the existing convention: root-level tag files
  and `data/` subdirectories share the tag prefix, as `stats/` and
  `params/` already do (`<tag>/models/`, `<tag>/train/`, `<tag>/generations/
  gen_N/`).

### The GPU worker image

A second tag of the worker image, `<worker_image>-gpu`, a further stage on
the CPU image adding `python3-pip`, `python3-dev` (torch.compile needs
headers; the compiler is already there), torch from the cu129 index as the
dev image installs it, plus onnx, onnxscript, natsort, schedulefree, tqdm.
Ubuntu 24.04's interpreter is externally managed, so the stage installs into
a venv layered under the bundle's `PYTHONPATH` (decided up front rather than
discovered at build time). torch wheels vendor their CUDA libraries, so the
image's own `libcudart` stays what TensorRT needs. The CPU tag keeps its
size.

`RoleSpec` gains `runtime: "engine" | "torch"`; the two container-creation
sites (`pod_create_spec`, `ssh_machine.run_container`) pick the tag from
it. `build_and_push_worker_image.py` builds and pushes both tags, and the
ABI record becomes a dict keyed by tag so `check_worker_image_current`
checks the tag a launch is about to use. match_eval stays on the engine
runtime, so the ssh machine's image does not grow.

### Runtime data on a pod

- The train role declares `deps`: the lexicon fetch (the existing
  `worker_deps.fetch_lexicon`) plus a new eval-dataset fetch from a bucket
  prefix `deps/positions-<content hash>.tar.gz`, which `deploy_current_tree`
  pushes when absent. This keeps `bootstrap.py` and the bundle format
  untouched (the bundle stays per-arch code, and a 40 MB dataset does not
  ride every deploy).
- The trainer aborts at startup if the eval datasets or the lexicon are
  missing, for every kind. Today's log-and-continue goes away; no opt-out.

## Sequence

0. **Calibrate and probe** (no repo code, under $2):
   `cloud_fleet.py up -n 1 --vcpus 16` on a CPU flavor against a throwaway
   tag for ten minutes, for games/s per vCPU (how many CPU pods a cloud-
   trained tag needs). One GPU pod by hand on a stock torch image to
   confirm that a stop delivers SIGTERM to PID 1 with a grace period the
   trainer's generation-boundary checkpointing can live with.
1. **PR 1 -- runtime.** GPU image tag, venv, `RoleSpec.runtime`, image
   selection at both creation sites, ABI record keyed by tag; eval-dataset
   deps prefix + fetch; train role `deps`; loud startup checks;
   `SCZ_DEVICE` allow-listed. Verified by running the unchanged trainer
   inside the GPU image on the laptop (`docker run --gpus all`) against a
   local tag.
2. **PR 2 -- trainer I/O contract**, local kind only: records out +
   controller ingest tick, `controls.json` in, atomic checkpoint save,
   commit markers. Acceptance test: a short run through the new path
   produces a `dashboard.db` identical to the old direct-write path.
   Reviewable on its own; no cloud code.
3. **PR 3 -- the trainer on a pod.** Generation publish on completion,
   trainer input adapter (poll, pull, restore), output delivery through
   the sink, sync watcher prefixes, controls push. Two shapes for the slot
   model, decided before PR 3 starts (see "Machines, slots and
   utilization" below): the narrow one adds `"cloud"` to the train role's
   kinds, one pod per slot; the general one makes a GPU pod a *machine*
   that hosts slots, which subsumes the companion-generator case. Either
   way, verified end to end by one real pod run of ~20 generations on the
   flavor chosen in step 0 against a local run of the same params:
   identical metric schema, wait fraction under the acceptance line,
   restore after a deliberate pod stop.
4. **Later, separately:** a per-tag generator autoscaler (below); an
   always-on controller host if unattended runs become a goal (the
   controller is one Tornado process plus the mount; moving it is
   deployment, not code); interruptible trainer pods once restore is
   proven cheap.

## Machines, slots and utilization

Added after discussion of where the money goes and how independently
launched tags should share hardware.

**Where the cost is in the hasty regime.** Per 20 000-game generation, with
cloud vCPUs taken as half the laptop's per-thread speed and a desktop 4090
as twice its GPU:

| Resource | Per generation | Cost |
|---|---|---|
| Generation, hasty-vs-hasty | ~1400 vCPU-s | ~$0.023 at $0.06 per vCPU-hour |
| Training, 4090 fully busy | ~50 GPU-s | ~$0.010 at $0.74/h |
| Training, 4090 alone with its 8 vCPUs (GPU ~30% busy) | | ~$0.037 |

Generation is already the larger cost, and the GPU idle problem is bounded
(perfect sharing saves about $25 on a 1000-generation run). Sharing one GPU
pod between two hasty tags buys nothing: the pod's binding resource is its
vCPUs, so two generation-bound tags on one 4090 produce the same
generations per hour in total, at the same cost each. The GPU becomes
shareable only once generation comes from elsewhere. The lever is
therefore per tag and local: feed each trainer enough generation. The
signals exist already -- the scheduler's gate (generation ahead) and the
trainer's cycle time minus `train_s`+`eval_s` (generation behind) -- and
generators are stateless spot pods the reconcile loop already restarts, so
a **per-tag generator autoscaler** that holds the trainer's wait fraction
near zero is the cheap form of utilization control. No coupling between
tags is needed.

**The neural-generation regime inverts this.** Once self-play runs neural
agents, inference dominates GPU time by a wide margin and the trainer is
the small consumer. Colocating a tag's generator with its trainer is then
natural: inference fills the GPU between epochs, and the model handoff is
a local file. Cross-tag sharing is still unnecessary, since one tag's
self-play saturates a GPU on its own; the knob is how many GPU machines a
tag gets, and utilization within a machine is the batching evaluation
service's job.

**The abstraction that serves both: machines are resources, tags are work,
slots bind the two.** The ssh kind already models this exactly (one
machine, many containers, from any tags); the cloud kind conflates them
(one pod is one slot of one tag). If a GPU pod became a *machine* -- a
pod-side agent reconciling slot processes against an assignment record it
polls from the bucket, the one channel that reaches a pod -- then:

- a companion generator is a generate slot placed on the trainer's machine
  (the bundled vCPUs are the cheapest generation compute there is: a
  16-vCPU CPU pod costs more per hour than a 4090 pod);
- two trainers on one GPU are two train slots on one machine, shared by
  process co-residency and CUDA time-slicing with no new abstraction (a
  trainer's wait is a sleep-poll that releases the GPU; at ~5 GiB each a
  24 GB card holds several);
- gates and pauses become per-slot process control rather than pod
  stop/start, so a paused slot no longer idles a whole pod;
- the neural-generation future is a placement choice, not an architecture
  change.

Placement stays manual (with two to four parallel runs an operator beats
any bin-packer); pacing between tags is the existing per-tag generational
gate; the one automated dimension is the generator autoscaler. What is
deliberately not built: a work-queue scheduler time-slicing one trainer
process across tags -- trainer state (optimizer, window, compiled model,
checkpoint) makes switching expensive, and the OS shares processes better
than we would.

PR 1 and PR 2 are unchanged under either shape of PR 3. The machine shape
costs a pod-side agent plus the assignment channel, roughly what a
companion-generator option on a one-pod-per-slot design would cost, and
subsumes both that option and the original standalone-pod idea. It is the
recommended shape for PR 3.

## Review record

Panel: hidden-complexity (session tier), rival-designer (codex, cross-
vendor), scope (sonnet), integration (sonnet), each blind to the others,
against the first draft, which hosted the whole run -- scheduler,
generator, trainer, match eval -- on one "standalone" GPU pod with a
one-to-one mirror of the tag root pushed to the bucket and a read-only
copy pulled by the dashboard.

Revised on the rival-designer critique (serious): a bucket-native trainer
under the existing controller follows the architecture instead of adding a
pod-local control plane, and it removes at a stroke the machinery the other
panelists found under the standalone shape -- a nested worker entrypoint, an
inline generator refactor, a supervisor with process-group semantics, a
second scheduler and match dispatcher (which, as hidden-complexity showed,
would have raced the dashboard's own ticks on a local-kind tag), hosted-tag
exclusivity, write guards in a data plane that has no task record, whole-
file sqlite replacement, pod volumes with an unverified stop/start story,
and a restore sequence that keyed the window on `train_state.json` while
the checkpoint lagged it (a trainer waiting forever on a generation the
scheduler would never open). It also delivers independently scalable
generation and a free local GPU for match eval in the first slice. The one
property given up, running with the laptop off, was not a stated goal and
is not how runs operate today.

Adopted from the other critiques: atomic checkpoint save; checkpoint push
every generation as one rolling object, with a commit marker written last;
eval datasets fetched as a dependency from a bucket prefix rather than
shipped in a bundle format the image-baked bootstrap would have to learn;
the ABI record keyed by image tag; the venv decision for the GPU stage;
`--size-only` pulls of immutable prefixes; SIGTERM-grace probing in step 0;
the "one scheduler change" dropped (with the cursor seeded, `_next_index`
already floors at it).

Made moot by the revision: contended vCPU/GPU budget on one pod; volume
sizing as a dashboard control; slicing match eval out; the local kind of a
standalone role; the second bucket layout.

Open -- human call: whether unattended operation (laptop off) should be a
requirement after all. If so, the standalone-pod shape or an always-on
controller host is back on the table; the trainer contract in PR 2 is
common to every answer, so it is not blocked on this.

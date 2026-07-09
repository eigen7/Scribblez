# Training workloads on the master dashboard

How the training pipelines run as first-class workloads of the master dashboard
(docs/master_dashboard.md): self-play generation farmed out to any number of interchangeable
local/cloud workers, the GPU trainer running as a distinguished singleton worker consuming the
shared data, and the whole run — creation, worker control, live training analysis — driven from
the one web shell. Both training workloads (position_eval and max_move_per_lane) share this
shape; position_eval is described throughout, with the probe differing only in its parameters
and tabs.

## Summary

- **One layout**: every workload's tags live at `<mount>/tags/<workload>/<tag>/`
  (scribblez/paths.py `TagPaths`); kill_test included. `WorkloadSpec.data_dir` and
  `TagPaths.root` are the same thing.
- **One worker contract**: `worker_entrypoint.py` dispatches on `(SCZ_WORKLOAD, SCZ_ROLE)` to a
  runner declared by the workload spec. Roles carry their own deps, stats shape, and scheduling
  traits (parallel vs. singleton, interruptible or not).
- **Generation by staging + controller-side ingest**: generators are generation-agnostic — they
  produce whole `.slog` chunks into a flat staging area, exactly the kill-test shape. A per-task
  scheduler on the controller host assigns chunks to generation directories, maintains the
  manifests, and paces the fleet. Multi-producer completion falls out of single-writer ingest.
- **The trainer is a pure consumer**: it never generates. It waits for complete generations,
  trains the sliding window, checkpoints, and publishes a small cursor file the scheduler and the
  UI read. A single-machine run is "1 local generator + 1 trainer" on the same task.
- **One web shell**: the training tabs (Loss, Positions, Training, Controls, Info) are
  position_eval's workload tabs inside the master `TaskView`, registered client-side in
  `web/src/workloads.tsx`.

## The workload-spec contract

Declared in `scribblez/workloads/base.py`; one module per workload in `scribblez/workloads/`
(`kill_test.py`, `position_eval.py`, `max_move_per_lane.py`, registry in `__init__.py`).
Registry modules stay import-light — heavy code (runners, deps fetchers, schedulers) is
referenced by dotted path and imported only when it runs, so cloud CPU pods import the registry
without torch.

```python
@dataclass(frozen=True)
class StatsSpec:
    unit: str                     # what a cycle delivers: "pairs", "games", "rows"
    phases: dict[str, str]        # per-cycle timing keys -> display labels,
                                  # e.g. {"gen_s": "self-play", "sim_s": "sim", "upload_s": "upload"}

@dataclass(frozen=True)
class RoleSpec:
    name: str                     # "generate", "train"
    title: str                    # shown in the add-worker UI
    runner: str                   # dotted path to run(ctx: WorkerContext) -> int
    deps: str = ""                # dotted path to a fetch-runtime-deps callable
    singleton: bool = False       # at most one slot per task (the trainer)
    kinds: tuple[str, ...] = ("local", "cloud")
    interruptible: bool = False   # cloud rental mode for this role's pods
    stats: StatsSpec | None = None

@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    title: str
    params_cls: type              # frozen task params (params.py dataclass)
    roles: tuple[RoleSpec, ...]
    scheduler: str = ""           # dotted path to a controller-side per-task tick (optional)
    progress: str = ""            # dotted path to progress(spec, tag) -> list[(label, value)]
    sync_data_dirs: tuple[str, ...] = ()   # bucket prefixes cloud_sync pulls
```

Field by field, what the dashboard derives from it:

- **`params_cls`** — the single source of truth for the new-tag form, CLI flags, worker env vars,
  and validation. Frozen at task creation. `scribblez/params.py` supports int, float, str, and
  bool fields. The split of knobs is uniform across workloads: *task params* (frozen; define the
  corpus and the model), *live controls* (per-tag dashboard.db control table, adopted by the
  trainer at its natural cadence), and *worker-slot resources* (threads/vcpus, per slot, never
  frozen).
- **`roles`** — the worker-slot taxonomy. `WorkerRecord` carries a `role`; the add-worker UI
  renders one form per role, restricted to that role's `kinds`; `WorkerManager` enforces
  `singleton` (refuses a second trainer slot) and passes `role.interruptible` to pod creation.
- **`runner` / `deps`** — the entrypoint contract. `worker_entrypoint.py` owns process concerns
  (env config, sink construction, SIGTERM handling, the provenance manifest) and delegates the
  loop to the role's runner via a `WorkerContext` (spec, role, tag, params, threads, sink).
- **`stats`** — drives the generic Stats tab: the per-worker record
  (`scribblez/workloads/worker.py` `WorkerStats`) keeps cumulative totals plus a bounded window
  of `{t, <phase>_s..., units, bytes}` samples; the summary table, throughput, cycle-breakdown,
  and timeline figures (`dashboard/worker_stats_figures.py`) are all schema-driven. The API
  reports the schema alongside the rows (`/api/task/stats`, `/api/task/figure/<name>?role=`) so
  the client renders columns without workload knowledge.
- **`scheduler`** — a per-task tick run by the dashboard server's reconcile loop (the natural
  home: it is the one always-on controller-host process, already running sync watchers and pod
  reconciliation). It gets the task record plus `SchedulerHooks`: `gate(role, reason)` to
  park/unpark a role's workers distinctly from operator pause, and (for tasks with cloud
  workers) `mirror(chunk, dest)` to replay ingests in the bucket. kill_test declares none; the
  training workloads share `scribblez/generational/scheduler.py`.
- **`progress`** — the tag-listing and Overview counters (kill_test: pair count; training
  workloads: fill state, completed generations, trainer cursor, rows).

Client side, a parallel registry (`web/src/workloads.tsx`) maps workload name → extra tab
components. Generic tabs (Overview, Stats-when-declared) come free; interactive tabs (Positions,
Controls) are real React components registered per workload. The server never renders tabs; it
only serves the generic data endpoints they use.

### kill_test under the contract

One role: `generate` (parallel, local+cloud, interruptible), runner = the cycle loop in
`scribblez/workloads/kill_test.py`, deps = lexicon + Macondo strategy, stats =
`("pairs", {gen_s, sim_s, upload_s})`. No scheduler. Each worker cycles in a private work dir
(`data/work/<worker_id>/`) and delivers complete pairs to the tag's `data/slogs/` store with a
`-<worker_id>` stem suffix, so any number of workers share a tag without name collisions.

## The position-eval workload

### Roles

| Role | Cardinality | Kinds | Interruptible | Does |
|---|---|---|---|---|
| `generate` | N, interchangeable | local + cloud | yes | one cycle = one whole `.slog` chunk of self-play games, delivered to the staging area |
| `train` | singleton | local (the GPU box) | — | consume complete generations: train, checkpoint, export ONNX, write dashboard.db |

The trainer never generates and the generators never train. Generation capacity is the worker
fleet, tuned per slot (threads/vcpus); there is no game-generation thread control on the
trainer. A single-machine run attaches one local generator and the trainer to the same task; the
CPU split between them is the two slots' thread counts.

### Data flow: staging + controller-side ingest

```
generator (local)  ──chunk──►  tags/position_eval/<tag>/data/staging/     ─┐
generator (cloud)  ──chunk──►  R2: position_eval/<tag>/staging/  ──sync──► ─┤
                                                                            │ scheduler ingest
                                                                            ▼ (single writer)
                                              data/test/                *.slog   (filled first, frozen)
                                              data/generations/gen_000000/{manifest.json, *.slog}
                                              data/generations/gen_000001/...
                                                                            │
                                                              trainer: SlogDataset(window dirs)
```

Generators are **generation-agnostic** (`scribblez/workloads/selfplay_gen.py`, shared by both
training workloads). A cycle produces one whole `.slog` chunk (`games_per_chunk` games, one
file) in the worker's private work dir and hands it to the sink (`cloud/sinks.py`): the local
sink renames it into the tag's `staging/` dir; the R2 sink uploads it under the tag's `staging/`
bucket prefix, whence the task's sync watcher pulls it into the same local `staging/`. Chunk
names carry a `-<worker_id>` stem suffix for global uniqueness. The work dir is wiped on worker
start: play_game buffers a batch and writes the file in one shot, so a crash mid-cycle can leave
a truncated file, and leftovers are never delivered — a restart loses at most the in-flight
chunk.

The **scheduler** (`scribblez/generational/scheduler.py`, ticked per task by the dashboard
server's reconcile loop) is the *only* writer of generation structure. Each tick it moves staged
chunks into the open generation directory and flips the manifest to `complete` when the target
game count is reached. Because assignment is one `rename()` per whole file by a single process,
the hard invariants hold trivially:

- every `.slog` belongs to exactly one generation and arrives whole (the C++ loader trusts
  headers; the file was closed before it ever left the generator);
- completion is a recorded fact (`manifest.json`); committed counts are recomputed from .slog
  headers each tick rather than tracked incrementally, so a crash between a rename and a
  manifest write self-heals;
- an ingest ledger (`data/ingest_log.txt`, one chunk name per line, written before the rename)
  makes assignment idempotent: a chunk re-appearing in staging (a cloud sync racing an ingest)
  is deleted, not assigned twice, and a crash between ledger write and rename loses that one
  chunk rather than duplicating it.

For cloud chunks the scheduler mirrors the assignment in the bucket (server-side move
`staging/ → generations/gen_N/`, via the `mirror` hook the WorkerManager provides), so the
bucket remains the durable archive *and* its layout mirrors the local corpus — disaster recovery
is `rclone copy` of the tag prefix — and the sync watcher, which pulls only `staging/` (plus
`stats/`, `params/`), never re-downloads an ingested chunk after the local copy moved.

### Generation lifecycle and pacing

Manifests carry `{index, target_games, status, committed_games}`; at most one generation is open
at a time. The scheduler's open/close rule provides both **background refill** and **flow
control**:

- open generation `M` when `M ≤ trainer_cursor + open_ahead` (task param, default 1);
- close it (mark complete) when `committed_games ≥ target_games`;
- when nothing is open — the fleet is `open_ahead` generations in front of the trainer — **gate**
  the generate role: park local generator processes and stop generator pods, shown in the UI as
  `waiting (ahead of trainer)`, distinct from operator pause. Gates are recorded on the task
  record (`gates`) and enforced by the reconcile loop; ungating happens when the trainer
  advances.

This supersedes the `BackgroundRefiller` sketch in docs/generational_training.md: overlap comes
from generators running continuously while the trainer trains, and the CPU-vs-GPU arbitration
degenerates to slot sizing plus the gate. It is also the concrete realization of that doc's
"distributed game generation" section — the generation directory is the transport; workers bolt
on through staging without touching the trainer.

`trainer_cursor` comes from a small `train_state.json` the trainer writes atomically beside its
rolling checkpoint at every checkpoint (`{generation_index, epoch_in_generation, rows_trained,
checkpoint_index}`; lifecycle.py `write_train_state`). The scheduler and the progress/Overview
endpoints read it; nobody outside the trainer parses the torch checkpoint. Before a trainer has
ever run, the cursor reads as 0, so a fresh task immediately opens the test fill and generations
0..open_ahead — generation can run ahead of the trainer being attached at all.

Overshoot is bounded and harmless: chunks that land while the open generation is already at
target (or while everything is gated but a worker's cycle was in flight) simply wait in staging
for a later ingest; a generation may exceed its target by the in-flight chunks, which the window
trains on like any other games.

### The frozen test split

The held-out test set rides the same protocol: the scheduler fills `data/test/` first (a
manifest with `target_games = test_games`, ingesting staged chunks like any generation), marks
it complete, and only then opens generation 0. It is never evicted and never regenerated;
probe/calibration metrics stay comparable across the run, and the train/test boundary is
file-level by construction. `test_games = 0` (max_move_per_lane's default) skips it.

### The trainer role

`scribblez/position_eval/trainer.py` (and its max_move_per_lane sibling) owns orchestration; the
gradient machinery (`run_epoch`, model, losses, ONNX export, checkpoint format, per-checkpoint
evals, dashboard.db writes) is the per-task training code it always was:

```
run(ctx):                                  # invoked by worker_entrypoint, role=train
    params  = frozen task params (from env, same transport as every worker)
    resume model/optimizer/state from the rolling checkpoint
    loop until SIGTERM or max_rows:
        wait until generation[cursor] is complete (poll manifests; sleep between checks —
            the GPU idling here IS the "generation is the bottleneck" signal, visible in Stats)
        train reuse-derived epochs over window_dirs(cursor, window)   # SlogDataset per window,
                                                                      # built after completion, so
                                                                      # its one-shot glob is safe
        per epoch: checkpoint + ONNX + metrics + evals + train_state.json + a stats sample
        evict generations beyond the window; cursor += 1
```

- **Launch**: a local worker slot like any other — the dashboard spawns `worker_entrypoint` with
  `SCZ_ROLE=train`, `SCZ_SINK=local`; logs land under the tag's `logs/`. The runner lives with
  the training code (referenced by dotted path) so generator bundles never import torch.
  `scripts/position_eval/train.py` remains as a thin CLI over the same runner for headless
  debugging (something must still fill generations for it to consume).
- **Pause**: SIGTERM raises out of the loop; resume repeats the partial epoch from the last
  checkpoint (at most one epoch's work repeated).
- **Live controls**: `base_lr`, `dataloader_workers`, `torch_threads` are dashboard.db controls
  (single writer: the trainer), the LR seeded from the task's `lr` param.
- **Stats**: the trainer publishes the same per-worker stats record as everyone else
  (`unit="rows"`, phases `{train_s, eval_s}`), so the Stats tab shows fleet throughput and
  trainer liveness side by side; the deep metrics stay in dashboard.db.
- A fresh start is a fresh tag; there is no in-place run reset.

### Seeds

Generators always run `play_game` with seed 0 (the binary draws from `std::random_device` per
chunk): a fleet splitting a generation under any deterministic seed partition would duplicate
games, so distributed corpus reproducibility is deliberately not offered. If a reproducible
single-machine corpus is ever needed, `generate_data.py` still exists.

### Failure and restart matrix

| Failure | Effect | Recovery |
|---|---|---|
| generator crash / pod preemption | loses at most the in-flight chunk (work dir wiped on start) | reconcile respawns/restarts it |
| dashboard server down | no ingest, no gating; local workers (its children) die; pods keep producing into bucket staging | on restart: reconcile respawns locals, ingest drains staging; overshoot bounded by pod output during the outage |
| trainer crash | training halts; generation continues to the ahead-limit gate | respawn resumes from the rolling checkpoint |
| sync lag | chunks arrive late to staging | ingest is idempotent (ledger); late chunks join the currently open generation |
| chunk uploaded but not yet ingested at eviction time | none — ingest only ever targets open generations, never evicted indices | — |
| corrupt staged chunk (unreadable header) | quarantined as `.bad`, never assigned | — |

### Dashboard UI

- **Overview**: workers table has a `role` column and a `waiting (<reason>)` state for gated
  workers; one add-worker form per role (the trainer form offers only local and disables once
  its singleton slot exists); the Task card shows the workload's progress counters (fill state,
  completed generations, trainer cursor, rows trained).
- **Stats**: the generic schema-driven tab — one section per role (generator games/hour and
  phase breakdown; trainer rows/s), each with the timeline/throughput/breakdown figures.
- **Loss / Positions / Training / Controls / Info**: the training tabs, registered as
  position_eval's workload tabs in `web/src/workloads.tsx` (max_move_per_lane registers Loss /
  Lane analysis / Controls / Info). They work off task+tag query params against the same Tornado
  process. The trainers no longer spawn a dashboard of their own; `scripts/dashboard.py` is the
  single entrypoint and `VITE_TOOL=dashboard` always renders the master app.

## Open questions

- **Chunk size** (`games_per_chunk`): the pacing/latency quantum — small enough that gating
  reacts within a generation, large enough that per-chunk overhead (process spawn, upload,
  ingest) stays noise. Default 1000; measure under real parameters.
- **Trainer wait behavior**: plain sleep-poll on manifests (simple, no IPC). If the idle tail
  between generations ever matters, the scheduler could touch a sentinel the trainer
  inotify-waits on — not worth it until measured.
- **GPU trainer in the cloud**: the role abstraction admits `kinds=("local","cloud")` for the
  trainer later (CUDA worker image + GPU pod + bucket-side checkpoints), but nothing depends on
  it.
- **Neural self-play generations**: once generation needs the current model, chunks must be
  stamped with the model version they were produced under (filename or sidecar), the scheduler
  routes/rejects by version, and model distribution to workers piggybacks on the gate/ungate
  cycle. Staging-plus-ingest is the right substrate for that; the stamping format is deferred to
  the neural phase.

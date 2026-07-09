# Master dashboard

The React dashboard is the single web entrypoint for Scribblez work: pick a workload,
pick or create a tag, configure its parameters, attach workers (the local machine and/or
rented cloud pods), start/pause them, and watch progress and workload-specific analysis —
all from the browser. Registered workloads: kill-test data generation and the two
training workloads (position_eval, max_move_per_lane), whose generator-fleet + singleton-
trainer shape is described in docs/position_eval_workload.md.

## Concepts

- **Workload** — a kind of work (e.g. `kill_test`, `position_eval`). Declared in the
  workload registry (`py/scribblez/workloads/`) as a `WorkloadSpec`: display title, the
  params dataclass, worker roles, an optional scheduler, and progress counters. The
  params dataclass is the single source of truth for parameters: its fields (name, type
  — int/float/str/bool — default, help metadata) generate both the web form (with
  validation) and the CLI flags, so the CLI and dashboard cannot drift.
- **Task** — one (workload, tag) pair with a fixed parameter set, recorded in
  `task.json` at the tag's root (`<mount>/tags/<workload>/<tag>/`). Parameters are
  frozen at task creation because every worker on a tag must run with identical settings
  for the data to be analyzable as one corpus. Tags created outside the dashboard (no
  task.json) still appear in the tag list, read-only.
- **Role** — which of the workload's worker kinds a slot runs (`RoleSpec`): parallel
  interchangeable generators, or a singleton trainer restricted to the local GPU box.
  Each role declares its runner, runtime deps, allowed kinds (local/cloud), whether its
  cloud pods rent interruptible, and its stats schema.
- **Worker** — a durable *slot* attached to a task: a record in task.json holding
  `worker_id`, role, kind, its resource allocation, and a **desired state**
  (running/paused). Two kinds:
  - **local**: the slot is backed by a subprocess of the dashboard server running the
    same worker loop as the cloud, with a thread-count knob (CPU allocation) and a local
    results sink (files stay in the mount dir; no upload).
  - **cloud**: the slot is backed by exactly one Runpod CPU pod (vCPU count + flavor),
    running the worker image + bundle flow of docs/cloud_compute.md and uploading results
    to R2. While a task has cloud workers, the server keeps a `cloud_sync --watch`
    subprocess running so cloud results stream into the local mount automatically.

  A slot's *actual* state can diverge from its desired state — a pod the operator paused,
  a local process that exited, or an interruptible pod Runpod reclaimed — and the UI
  shows both. The server reconciles desired vs. actual: on startup (relaunching local
  workers that should be running) and periodically (restarting reclaimed interruptible
  pods when capacity returns).
- **Gates** — a workload's scheduler can *park* a role without touching the operator's
  desired state (e.g. the training workloads' generators once they are a generation
  ahead of the trainer). Gated workers show as `waiting (<reason>)` and resume
  automatically when the scheduler releases the gate; the reconcile loop enforces gates
  exactly like desired state.
- **Interruptible rentals** — `RoleSpec.interruptible` declares whether a role
  tolerates preemption; the generator roles do (atomic chunks/pairs, at most one
  in-flight cycle lost), so their cloud pods are created `interruptible: true` for the
  discount, and preemption is handled by the reconcile loop rather than by a human.

## The web flow

1. **Home page**: workload selector → tag list for that workload, sortable by name or
   last-activity, each row showing the workload's progress counters and running-worker
   counts. Selecting a tag opens its task view. **New tag** opens the schema-generated
   params form (typed inputs, validation, defaults prefilled); Create writes task.json
   and opens the task.
2. **Task view** — tabs:
   - **Overview**: the frozen params; task-agnostic info (created, progress counters,
     data location, live cloud $/hr); the workers table (worker, role, kind, resources,
     state, cost, ssh for pods); one add-worker form per role (local: threads; cloud:
     count × vCPUs × flavor from the Runpod flavor enum; a singleton role's forms
     disable once it has a slot); buttons: per-worker start/pause/remove and task-level
     Start all / Pause all. Pausing a cloud worker stops the pod (billing drops to
     disk-only); pausing a local worker interrupts the subprocess (the loops are
     interruption-safe by design). Only a non-running worker can be removed, so removal
     never silently discards an in-flight cycle. The home page's tag list offers
     per-tag deletion (local data dir only — the bucket archive is kept), refused while
     the tag has workers.
   - **Stats**: the generic per-role worker statistics — a summary table plus Bokeh
     figures (cumulative units over time, units/hour, per-phase cycle-time breakdown),
     all driven by the role's declared stats schema (unit noun + timing phases). A
     generator whose upload share dominates is network-bound; one whose sim share
     dominates is CPU-bound; the trainer section shows rows/s.
   - **Workload tabs** from the client registry (`web/src/workloads.tsx`): the training
     workloads add Loss / Positions (or Lane analysis) / Training / Controls / Info —
     the training-analysis views of docs/react_dashboard.md.

## Worker statistics (the data behind Stats)

Each worker maintains `stats/<worker_id>.json` under the tag — cumulative counters
(units, cycles) plus a bounded window of recent per-cycle samples
`{t, <phase>_s..., units, bytes}` and identity fields (role, kind, threads/vcpus, arch).
Cloud workers rcat it to the bucket after every cycle (it rides the same sync back);
local workers write it directly. The dashboard reads only the local mount, so the
Stats tab needs no live bucket access; freshness is bounded by the sync interval.

Workers deliver output files with a `-<worker_id>` stem suffix: output names are
per-machine nanosecond timestamps, so two workers could in principle mint the same
name. The suffix makes names globally unique while preserving the stem-based pair
matching downstream tools rely on.

## Server architecture

One Tornado process (the existing `scribblez.dashboard.api`) hosts a control plane
alongside the read-only training data plane:

```
GET  /api/workloads                      registry: params schemas + role declarations
GET  /api/workload_tags?workload=        tag list with progress/last-activity
POST /api/tasks                          create task.json (validates params)
GET  /api/task?workload=&tag=            params + progress + gates + workers with live status
POST /api/task/workers                   add worker(s) for a role (local or cloud)
POST /api/task/worker_action             {action: start|pause|remove, worker_id?};
                                         no worker_id = every slot (Start/Pause all)
GET  /api/task/stats?workload=&tag=      per-role stats schemas + per-worker summary rows
GET  /api/task/figure/<name>?workload=&tag=&role=   Bokeh json_items for the Stats tab
```

- `py/scribblez/dashboard/tasks.py` — task records, tag enumeration, progress.
- `py/scribblez/dashboard/workers.py` — the WorkerManager: local subprocesses (spawn,
  interrupt, respawn; logs under the tag's `logs/`), cloud pods via `py/cloud`
  (create/stop/start/terminate), the per-task sync watcher, scheduler ticks (with the
  gate and bucket-mirror hooks), and gate enforcement.
- `py/scribblez/dashboard/worker_stats_figures.py` — the schema-driven Stats figures.
- The API binds to localhost only: it holds cloud credentials and launches processes, so
  it must not be reachable off-machine (the container port-forward provides browser
  access).

The React shell is the master app (`web/src/components/master/MasterApp.tsx`): home page
+ task view, with workload-specific tabs registered in `web/src/workloads.tsx`.
`scripts/dashboard.py` starts it (optionally opening on a workload/tag).

## Worker loop changes

The worker entrypoint (`py/cloud/worker_entrypoint.py`) is the one worker loop for
both kinds and every role, parameterized by a **results sink** (`SCZ_SINK=r2|local`,
`py/cloud/sinks.py`) and dispatching on `(SCZ_WORKLOAD, SCZ_ROLE)` to the role's runner
from the workload registry. Runners cycle in a private per-worker work dir and deliver
whole output files through the sink; both sinks record the per-cycle timing samples.

## Later

Kill-test analysis results tab (render `kill_test.py`'s per-arm JSON output + a re-run
button), live log streaming, run history, GCP spot adapter, volunteer ingest.

# Master dashboard

The React dashboard is the single web entrypoint for Scribblez work: pick a
workload, pick or create a tag, configure its parameters, attach workers (the
local machine and/or rented cloud pods), start/pause them, and watch progress
and workload-specific analysis — all from the browser.

## Concepts

- **Workload** — a kind of work, declared in the workload registry
  (`py/scribblez/workloads/`) as a `WorkloadSpec`: display title, params
  dataclass, worker roles, an optional scheduler, and progress counters. The
  params dataclass is the single source of truth for parameters — its fields
  generate both the web form (with validation) and the CLI flags, so the CLI
  and dashboard cannot drift.
- **Task** — one (workload, tag) pair with a fixed parameter set, recorded in
  `task.json` at the tag's root (`<mount>/tags/<workload>/<tag>/`). Parameters
  freeze at task creation because every worker on a tag must run identical
  settings for the data to be analyzable as one corpus. Tags created outside
  the dashboard appear in the tag list, read-only.
- **Role** — which of the workload's worker kinds a slot runs (`RoleSpec`):
  parallel interchangeable generators, or a singleton trainer restricted to
  the local GPU box. Each role declares its runner, runtime deps, allowed
  kinds (local/cloud), whether its pods rent interruptible, whether it needs
  GPU hardware, and its stats schema.
- **Worker** — a durable *slot* attached to a task: role, kind, resource
  allocation, and a **desired state** (running/paused). A local slot is a
  subprocess of the dashboard server running the same worker loop as the
  cloud, with a local results sink; a cloud slot is exactly one Runpod pod
  running the image + bundle flow of [cloud_compute.md](cloud_compute.md).
  While a task has cloud workers, the server keeps a sync watcher running so
  cloud results stream into the local mount. A slot's *actual* state can
  diverge from desired (an operator-stopped pod, a dead subprocess, a
  reclaimed interruptible pod); the UI shows both, and the server reconciles
  desired vs. actual on startup and periodically.
- **Gates** — a workload's scheduler can *park* a role without touching the
  operator's desired state (e.g. the training workloads' generators once they
  are a generation ahead of the trainer). Gated workers show as
  `waiting (<reason>)` and resume automatically when released.
- **Interruptible rentals** — roles that tolerate preemption (the generators:
  at most one in-flight cycle lost) rent interruptible pods for the discount;
  preemption is handled by the reconcile loop, not a human.

## The web flow

Home page: workload selector → tag list with progress counters and
running-worker counts; **New tag** opens the schema-generated params form.
Selecting a tag opens its task view, with tabs:

- **Overview** — the frozen params, progress, live cloud $/hr, the workers
  table, one add-worker form per role (cloud forms offer a live instance
  selector fed by the cached Runpod catalog), and per-worker plus task-level
  start/pause/remove. Pausing a cloud worker stops the pod (billing drops to
  disk-only); only a non-running worker can be removed, so removal never
  silently discards an in-flight cycle. Tag deletion (local data dir only —
  the bucket archive is kept) is refused while the tag has workers.
- **Stats** — generic per-role worker statistics: a summary table plus Bokeh
  figures (cumulative units, units/hour, per-phase cycle-time breakdown),
  driven by the role's declared stats schema. Each worker maintains a stats
  JSON under the tag (cumulative counters plus a bounded window of per-cycle
  samples); cloud workers upload it every cycle and it rides the normal sync,
  so the dashboard reads only the local mount.
- **Workload tabs** from the client registry (`web/src/workloads.tsx`): the
  training workloads add the training-analysis views of
  [react_dashboard.md](react_dashboard.md).

Output files carry a per-worker stem suffix: names are per-machine nanosecond
timestamps, so the suffix makes them globally unique across workers while
preserving the stem-based pair matching downstream tools rely on.

## Server architecture

One Tornado process (`scribblez.dashboard.api`) hosts the control plane
alongside the read-only training data plane:

- `py/scribblez/dashboard/tasks.py` — task records, tag enumeration, progress.
- `py/scribblez/dashboard/workers.py` — the WorkerManager: local subprocesses
  (spawn/interrupt/respawn; logs under the tag's `logs/`), cloud pods via
  `py/cloud`, the per-task sync watcher, scheduler ticks, and gate
  enforcement.
- `py/scribblez/dashboard/worker_stats_figures.py` — the schema-driven Stats
  figures.
- `web/src/components/master/MasterApp.tsx` — the React shell (home page +
  task view); `scripts/dashboard.py` starts it.

The API binds to localhost only: it holds cloud credentials and launches
processes, so it must not be reachable off-machine (the container
port-forward provides browser access).

The worker entrypoint (`py/cloud/worker_entrypoint.py`) is the one worker
loop for both kinds and every role, parameterized by a results sink (R2 or
local; `py/cloud/sinks.py`) and dispatching to the role's runner from the
workload registry. Runners cycle in a private per-worker work dir and deliver
whole output files through the sink.

## Later

Kill-test analysis results tab, live log streaming, run history, GCP spot
adapter, volunteer ingest.

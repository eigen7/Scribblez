# Master dashboard

The React dashboard is the single web entrypoint for Scribblez work: pick a workload,
pick or create a tag, configure its parameters, attach workers (the local machine and/or
rented cloud pods), start/pause them, and watch progress and workload-specific analysis —
all from the browser. The first (and currently only) registered workload is kill-test
data generation; the training dashboards continue to work as before and fold into the
registry over time.

## Concepts

- **Workload** — a kind of work (e.g. `kill_test`). Declared in the workload registry
  (`py/scribblez/workloads.py`) as a `WorkloadSpec`: display title, the params dataclass,
  where its data lives, how to launch a worker locally, and the env-var mapping for cloud
  workers. The params dataclass is the single source of truth for parameters: its fields
  (name, type, default, help metadata) generate both the web form (with validation) and
  the CLI flags, so the CLI and dashboard cannot drift.
- **Task** — one (workload, tag) pair with a fixed parameter set, recorded in
  `<data_dir>/task.json` at creation. Parameters are frozen at task creation because
  every worker on a tag must generate with identical settings for the data to be
  analyzable as one corpus. Tags created before the dashboard existed (no task.json)
  still appear in the tag list, read-only.
- **Worker** — a durable *slot* attached to a task: a record in task.json holding
  `worker_id`, kind, its resource allocation, and a **desired state** (running/paused).
  Two kinds:
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
  pods when capacity returns). This maps one-to-one onto Runpod's pod model
  (stop/start/terminate + the `interruptible` rental flag).
- **Interruptible rentals** — `WorkloadSpec.interruptible` declares whether a workload
  tolerates preemption; kill-test generation does (atomic pairs, at most one in-flight
  cycle lost), so its cloud pods are created `interruptible: true` for the discount, and
  preemption is handled by the reconcile loop rather than by a human.

## The web flow

1. **Home page**: workload selector → tag list for that workload, sortable by name or
   last-activity, each row showing pair counts and running-worker counts. Selecting a tag
   opens its task view. **New tag** opens the schema-generated params form (typed inputs,
   int/bool validation, defaults prefilled); Create writes task.json and opens the task.
2. **Task view** — tabs:
   - **Overview**: the frozen params; task-agnostic info (created, pair count, data
     location, live cloud $/hr); the workers table; add-worker forms (local: threads;
     cloud: count × vCPUs × flavor from the Runpod flavor enum, with a console link for
     live availability/pricing); per-worker rows show kind, state, cost rate, and for
     cloud pods the ssh command (`ssh <podId>@ssh.runpod.io`, requires a registered
     Runpod SSH key); buttons: per-worker start/pause/remove and task-level Start all /
     Pause all (enabled only when they would do something). Pausing a cloud worker stops
     the pod (billing drops to disk-only); pausing a local worker interrupts the
     subprocess (the loop is interruption-safe by design). Only a non-running worker can
     be removed, so removal never silently discards an in-flight cycle. The home page's
     tag list offers per-tag deletion (local data dir only -- the bucket archive is
     kept), refused while the tag has workers.
   - **Stats** (kill-test): Bokeh figures built server-side and embedded via the existing
     `BokehFigure` path — cumulative pairs over time (per worker and total), per-worker
     throughput (pairs/hour), and a per-worker cycle-time breakdown (generate vs sim vs
     upload seconds) that makes bottlenecks legible: a worker whose upload share
     dominates is network-bound (the upload MB/s column gives the transfer rate, e.g.
     for spotting Asia↔US path problems), one whose sim share dominates is CPU-bound.

## Worker statistics (the data behind Stats)

Each worker maintains `stats/<worker_id>.json` under the tag — cumulative counters
(pairs, cycles) plus a bounded window of recent per-cycle samples
`{t, gen_s, sim_s, upload_s, bytes}` and identity fields (kind, threads/vcpus, arch).
Cloud workers rcat it to the bucket after every cycle (it rides the same sync back);
the local worker writes it directly. The dashboard reads only the local mount, so the
Stats tab needs no live bucket access; freshness is bounded by the sync interval.

Cloud workers upload each pair's files with a `-<worker_id>` stem suffix: output names
are per-machine nanosecond timestamps, so two workers could in principle mint the same
name, and a bucket collision would silently splice one worker's .slog with another's
.sobs. The suffix makes bucket (and synced-local) names globally unique while preserving
the stem-based pair matching downstream tools rely on.

## Server architecture

One Tornado process (the existing `scribblez.dashboard.api`) gains a control plane
alongside the read-only data plane:

```
GET  /api/workloads                      registry + params schemas (drives the form)
GET  /api/workload_tags?workload=        tag list with counts/last-activity
POST /api/tasks                          create task.json (validates params)
GET  /api/task?workload=&tag=            params + info + workers with live status
POST /api/task/workers                   add worker(s) (local or cloud) and start them
POST /api/task/worker_action             {action: start|pause|remove, worker_id?};
                                         no worker_id = every slot (Start/Pause all)
GET  /api/kill_test/stats?tag=           per-worker summary rows for the Stats table
GET  /api/kill_test/figure/<name>?tag=   Bokeh json_items for the Stats tab
```

- `py/scribblez/dashboard/tasks.py` — task records, tag enumeration, params
  schema/validation (shared with the argparse generator in workloads.py).
- `py/scribblez/dashboard/workers.py` — the WorkerManager: local subprocesses (spawn,
  interrupt, respawn; logs under the tag's `logs/`), cloud pods via `py/cloud`
  (create/stop/start/terminate), and the per-task sync watcher.
- The API binds to localhost only: it holds cloud credentials and launches processes, so
  it must not be reachable off-machine (the container port-forward provides browser
  access).

The React shell (`web/src/AppDashboard.tsx`) becomes the master app: home page +
task view. When launched with `VITE_TASK` set (the trainers' auto-spawned dashboards),
it renders the training-task view exactly as before; the master flow is the no-VITE_TASK
default. `scripts/dashboard.py` with no arguments starts the master dashboard.

## Worker loop changes

The cloud worker entrypoint (`py/cloud/worker_entrypoint.py`) is the one worker loop for
both kinds, parameterized by a **results sink** (`SCZ_SINK=r2|local`): r2 uploads each
pair and deletes the local copy; local leaves pairs in place. Both sinks record the
per-cycle timing samples. The generation cycle itself (`scribblez/kill_test_gen.py`,
moved out of scripts/ so library code does not import from scripts/) reports per-phase
timings; `generate_kill_test_data.py` remains as the thin CLI over the same cycle.

## Later (unchanged from the original proposal)

Kill-test analysis results tab (render `kill_test.py`'s per-arm JSON output + a re-run
button), folding the training workloads into the registry as launchable tasks, live log
streaming, run history, GCP spot adapter, volunteer ingest.

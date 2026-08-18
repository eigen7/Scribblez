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
  allocation, and a **desired state** (running/paused). A slot is added
  paused: nothing launches until its first start, which for a cloud slot is
  the moment the backing pod is created (a pod boots on creation), so an
  added-but-never-started slot costs nothing. A local slot is a
  subprocess of the dashboard server running the same worker loop as the
  cloud, with a local results sink; a cloud slot is exactly one Runpod pod
  running the image + bundle flow of [cloud_compute.md](cloud_compute.md); an
  ssh slot is that same image + bundle flow as a Docker container on a machine
  the operator owns (a spare laptop on the LAN, a home server), driven over
  SSH. Cloud slots deliver through the results bucket, and while a task has
  any, the server keeps a sync watcher running so their results stream into
  the local mount. An ssh slot skips the bucket entirely: it delivers into its
  own container and the reconcile pass reads finished output back over the
  control link (`py/cloud/ssh_transfer.py`), so a cycle on an operator's own
  machine contains no network at all. A slot's *actual* state can diverge from
  desired (an operator-stopped pod, a dead subprocess, a reclaimed
  interruptible pod, an ssh machine that is off the network); the UI shows
  both, and the server reconciles desired vs. actual on startup and every few
  seconds. That pass is the only thing that talks to a machine or the cloud
  API, and it does so off the event loop; every status request is served from
  what it last observed, so no slow host can stall the dashboard. A slot the
  pass has not reached yet reads `checking`. A container that is not running
  carries its reason -- exit code and last log line -- into the workers table,
  and one that keeps dying is retried with a growing delay rather than every
  pass.
- **Gates** — a workload's scheduler can *park* a role without touching the
  operator's desired state (e.g. the training workloads' generators once they
  are a generation ahead of the trainer). Gated workers show as
  `waiting (<reason>)` and resume automatically when released. Parking suspends
  what it cheaply can (an ssh container is paused, keeping its unpacked bundle
  and in-flight chunk) and stops what it must: a local worker, which restarts
  in a second, and a cloud pod, which bills while it idles.
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
  start/pause/remove. Adding a worker records it paused — review the slot,
  then start it. Pausing a cloud worker stops the pod (billing drops to
  disk-only); only a non-running worker can be removed, so removal never
  silently discards an in-flight cycle. Tag deletion (local data dir only —
  the bucket archive is kept) is refused while the tag has workers.
- **Stats** — generic per-role worker statistics: fleet-aggregate tiles
  (units/hour, totals, cycle time, worker health), Bokeh figures (units/hour
  over time, per-phase cycle-time breakdown), and a per-worker detail table,
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

## SSH worker machines

An ssh slot's machine is prepared once, by hand:

- **SSH**: reachable non-interactively from the dev container — key-based
  auth, no prompts. The key to authorize is the container's own
  `~/.ssh/id_ed25519.pub`; devenv_utils persists that identity host-side, so
  authorizing it once holds across container relaunches. The form's host
  string is passed to `ssh` verbatim, so `user@host` and `~/.ssh/config`
  aliases both work — and the config file persists alongside the key, so an
  alias defined once keeps working too.
- **Docker**: installed, with the SSH user able to run it (in the `docker`
  group).
- **Worker image**: pulled once — `docker login` (the image repo is private;
  see `registry.worker_image` in the credentials file) then
  `docker pull <worker image>`. Containers start with `--pull=never`, so a
  missing image is an instant, actionable error rather than a long pull
  blocking the dashboard.

Slots then behave like pods: the machine's CPU arch picks its bundle (generic
`x86-64` fallback), which the machine still fetches from the bucket at
startup, and the reconcile loop restarts a container that died (e.g. the
machine rebooted). Results go the other way -- collected over ssh rather than
uploaded -- so nothing but the bundle fetch touches R2. Delivered chunks are
deleted from the container only once they are safely on the controller's disk;
what a container has not yet handed over goes with it if the slot is removed. A scheduler gate parks the container by pausing it rather than
stopping it, so a gate that flips every minute costs nothing: no bundle
refetch, and the chunk in flight survives. An operator pause still stops it
(cleanly, flushing completed output). The bundle itself is deployed for you -- a task builds and pushes
the controller's tree when its first remote worker starts, and pins the
result, so every worker of one task runs identical code and editing code
mid-run does not change what the fleet is executing. The Overview badges the
tree having moved on and offers Redeploy, which repins the task and replaces
its containers (a pod's or container's bundle is fixed at creation). A machine that is off the network shows `unreachable`; the server
leaves it alone — its worker may well still be running — and resumes control
when SSH works again. Keep the machine from sleeping on lid-close if it is a
laptop.

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

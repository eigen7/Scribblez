# Master dashboard: one web UI to launch, monitor, and analyze all work

A proposal for evolving the React dashboard into the single entrypoint for all Scribblez
work: pick a workload, configure it, launch it onto local and/or cloud workers, watch it,
stop it, and view its workload-specific analysis tabs — from the browser.

## Where we start (already true today)

The dashboard is already a single umbrella, which makes this proposal an evolution rather
than a rewrite:

- **One React app** (`web/src/AppDashboard.tsx`, `VITE_TOOL=dashboard`) and **one Tornado
  data API** (`py/scribblez/dashboard/api.py`) serve both training tasks, with
  task-conditional tabs and a task-aware per-request DB connection.
- Clean extension points exist: new figure tabs are one entry in `FIGURES` (api.py) plus
  one in `FIGURE_TABS` (AppDashboard.tsx); interactive tabs are one component plus a
  `renderTab()` branch; new endpoints subclass `_Base` and register in `make_app()`.
- The cloud fleet machinery (`py/cloud/`: bundles, Runpod client, worker env contract) is
  already a library the CLI drives; a dashboard can drive the same library.

What's missing, in order of substance:

1. A **control plane**: nothing today can launch or stop work; the dashboard only reads.
2. The shell is **per-run**: `task` is fixed at launch (`VITE_TASK`), and trainers spawn a
   dashboard instance per training run rather than there being one standing dashboard.
3. **Kill-test results** appear nowhere: `kill_test.py` is terminal-only (it already
   writes per-arm JSON histories to its results dir — unread by anything).

## Proposed shape

```
 browser ─► React shell
             ├─ global nav: workload picker ▸ tag picker ▸ tabs
             ├─ Jobs tab (new, workload-agnostic): launch form · running jobs ·
             │    start/stop · logs · cloud fleet cost meter
             └─ workload tabs (existing): Loss · Positions · Training · Lane analysis · …
                     │
             Tornado API (same process as today)
             ├─ data plane: /api/figure, /api/lane/*, /api/post_move/*  (unchanged)
             └─ control plane (new): /api/workloads · /api/jobs · /api/jobs/<id>/stop|log
                     │
             JobManager ──► local jobs: subprocess (generate_kill_test_data, train, …)
                       └──► cloud jobs: py/cloud fleet up/status/down + cloud_sync watcher
```

### 1. The workload registry — one source of truth

A `WorkloadSpec` registry (new module, e.g. `py/scribblez/workloads.py`) describing each
launchable workload:

- name (`kill_test`, `post_move_value_train`, `max_move_per_lane_train`, …)
- its params dataclass (e.g. the existing `KillTestParams`) — field names, types, and
  defaults become both the CLI flags and the dashboard's auto-generated config form
- how to launch it locally (argv builder) and on the cloud (worker env-var mapping)
- which dashboard tabs it owns

This is the generalization already wanted on the CLI side (cloud_fleet.py is hardcoded to
kill_test): `cloud_fleet.py --workload <w>` and the dashboard's launch form both read the
same spec, so adding a workload is one registry entry, not N touchpoints.

### 2. The JobManager + jobs API

A job is `{job_id, kind: local|cloud, workload, tag, params, state, started_at, …}`,
persisted under `<mount>/jobs/<job_id>/` (a small JSON record + log file), so a dashboard
restart re-attaches to running work instead of orphaning it.

- **Local jobs**: `subprocess.Popen` of the workload's argv with stdout/stderr to the
  job's log file; stop sends the interrupt the tool already handles gracefully (the
  generators are interruption-safe by design). The local machine is just another worker.
- **Cloud jobs**: a "fleet" job — N pods created via the existing `py/cloud` library, with
  the job record holding pod ids. Status merges pod state (Runpod API), bucket progress
  (pair counts), and $/hr; stop = fleet down. Each cloud job also owns a background
  `cloud_sync` watcher so results stream into the local mount automatically while it runs.
- Endpoints: `GET /api/workloads` (registry + param schemas), `POST /api/jobs` (launch),
  `GET /api/jobs`, `POST /api/jobs/<id>/stop`, `GET /api/jobs/<id>/log?tail=N`.

### 3. Shell changes

- Promote **workload** to a runtime navigation dimension alongside tag (today: baked in as
  `VITE_TASK` at Vite launch). Tab lists are already task-conditional; they become
  workload-conditional via the registry.
- A standing **Jobs tab** independent of workload: launch form (workload picker →
  auto-generated params form → local/cloud toggle, worker count, vCPUs), the running-jobs
  table with start/stop, log tail, and a prominent fleet cost meter.
- `scripts/dashboard.py` (no args) becomes the master entrypoint: start once, leave
  running. Trainer auto-spawn becomes "reuse the healthy standing dashboard if one is up,
  else spawn" — and a dashboard-launched training run is just a local job, closing the
  loop.

### 4. Kill-test analysis tab (the one genuinely new data surface)

`kill_test.py` stays a CLI, but its analysis becomes displayable: running it (locally, as
a job — it's compute-heavy) already leaves per-arm JSON histories in the tag's results
dir; a small `/api/kill_test/results?tag=` endpoint reads those, and a tab renders the
4-arm comparison (and a "re-run analysis" button that launches the job). No live coupling
to the analysis internals — the JSON on disk is the interface.

## Safety and scope notes

- **Bind the API to localhost.** Today it listens on `0.0.0.0`; once it can spend money
  and run subprocesses, it must not be reachable off-machine (the container port-forward
  already provides browser access). Cloud credentials stay server-side; the browser never
  sees them.
- **Spend guardrails**: launching cloud workers shows the computed $/hr before confirm;
  the cost meter is on the Jobs tab permanently; `down` is one click.
- The C++ WebSocket tools in `web/` (`play_game`, `manual`, `board`) are untouched; they
  share only the component library.

## Phasing (each lands independently useful)

1. **Registry + generic CLI**: `WorkloadSpec` registry; `cloud_fleet.py --workload`
   generalization; shared fleet lib. Pure refactor, no UI.
2. **Jobs core**: JobManager + jobs API + Jobs tab, local jobs only; `dashboard.py` as
   standing master (workload picker in-app; trainer spawn made reuse-aware).
3. **Cloud jobs**: fleet launch/stop/status from the Jobs tab + per-job auto-sync.
4. **Kill-test tab**: results endpoint + comparison view + re-run button.
5. **Niceties, iteratively** (as they earn their keep): live log streaming, run history,
   bundle picker (LATEST vs pinned), per-flavor throughput/$ calibration display,
   notifications on fleet errors or arch-fallback warnings.

## Open questions

- Params forms: generate purely from dataclass fields (uniform but plain) vs. per-workload
  hand-tuned forms (nicer, more code). Proposal: generated, with an escape hatch.
- Should analysis runs (`kill_test.py`) be jobs too (proposed: yes — they're long-running
  compute like everything else), or stay CLI-only initially?
- Job identity across dashboard restarts: pidfile + process-group re-attach is proposed;
  the alternative (jobs die with the dashboard) is simpler but hostile to long runs.

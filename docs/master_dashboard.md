# Master dashboard

The React dashboard is the single web entrypoint for Scribblez work. From the
browser you pick a workload, pick or create a tag, set its parameters, attach
workers, start and pause them, and watch progress and workload-specific
analysis. Workers can run on the local machine, on your own machines over ssh,
or on machines the dashboard rents from a cloud provider.

Launch it with `./py/scripts/dashboard.py` (optionally `--workload W --tag T`
to open straight onto a tag).

## Concepts

**Workload.** A kind of work, declared once in the workload registry
(`py/scribblez/workloads/`) as a `WorkloadSpec`: a display title, a params
dataclass, worker roles, an optional scheduler, and progress counters. The
params dataclass is the single source of truth for parameters. Its fields
generate both the web form (with validation) and the CLI flags, so the two
cannot drift.

- A field that declares `choices` has a closed value set. Validation and the
  CLI enforce it, and the form renders a selector.
- `primary_params` names the few fields that define a run, in the order the
  form shows them. The rest sit at their defaults in the form's collapsed
  **Advanced** section.
- `profiles` are named partial value sets over the dataclass defaults (for
  example, one recipe per trunk), with a `default_profile`. Values resolve as
  defaults ← profile ← operator edits, in the form and in the CLI
  (`--profile`) alike, and a task records which profile it came from.
  Profiles are code: tuning results are promoted into them by PR, and
  changing a profile never touches an existing task's frozen params.

**Task.** One (workload, tag) pair with a fixed parameter set, recorded in
`task.json` at the tag's root (`<mount>/tags/<workload>/<tag>/`). Parameters
freeze at task creation because every worker on a tag must run identical
settings for its output to be analyzable as one corpus. Tags created outside
the dashboard appear in the tag list read-only.

**Role.** Which of the workload's worker kinds a slot runs (`RoleSpec`): for
example parallel, interchangeable generators, or a singleton trainer. A role
declares its runner, runtime deps, the worker kinds it allows (local, ssh),
whether it needs a GPU, which worker image it runs on, and its stats schema.

**Worker.** A durable *slot* attached to a task, with a role, a kind, and a
**desired state** (running or paused). A slot is created paused; nothing
launches until its first Start.

- A **local** slot is a subprocess of the dashboard server running the worker
  loop with a local results sink.
- An **ssh** slot runs the worker image plus a code bundle
  ([cloud_compute.md](cloud_compute.md)) as a Docker container on a machine
  reached over ssh: one you own (a spare laptop, a home server) or one the
  dashboard rented for the task (see [Machines](#machines)).

**Actual vs. desired state.** A slot's actual state can diverge from the
desired one: a local subprocess died, an ssh machine dropped off the network,
a rented machine is stopped. The UI shows both. A reconcile pass, run at
startup and every few seconds, drives actual toward desired. It is the only
code that talks to a machine or the provider's API, and it runs off the event
loop; status requests are served from what it last observed, so no slow host
can stall the dashboard. States worth knowing:

| State | Meaning |
|---|---|
| `checking` | the reconcile pass has not reached the slot yet |
| `starting` | its container does not exist yet; on a machine that has never run one, this lasts as long as pulling the image |
| `waiting (<reason>)` | parked by the workload's scheduler (see Gates) |
| `unreachable` | the machine does not answer ssh; the server leaves it alone and resumes control when it does |
| `finished` | the worker exited 0: it reached its role's terminal condition (see below) |

A container that is not running shows its reason in the workers table: the
exit code and last log line, or why its creation failed. One that keeps dying
is retried with a growing delay rather than on every pass.

**Finished workers.** A worker that exits 0 has reached its role's terminal
condition (a trainer's `max_rows`, a generator's cycle cap). Its slot flips to
paused and reads `finished` instead of being restarted forever; Start clears
it. The worker entrypoint exits non-zero when SIGTERM ended the run, so a stop
or a machine reboot never reads as completion.

**Gates.** A workload's scheduler can *park* a role without touching the
operator's desired state, for example the training workloads' generators once
they are far enough ahead of the trainer. Gated workers show
`waiting (<reason>)` and resume automatically when released. Parking suspends
what it cheaply can and stops the rest: an ssh container is paused, keeping
its unpacked bundle and in-flight chunk, while a local worker is stopped and
restarts within a second of release.

## The web flow

The home page shows a workload selector and that workload's tags, with
progress counters and running-worker counts. **New tag** opens the params form
(`primary_params` up front, the rest under **Advanced**). With profiles, the
form starts from the default one; ◆ marks a value the selected profile sets
and ✎ one you edited. Edits belong to the profile they were made under:
switching profiles shows the other recipe with its own edits, switching back
restores yours, and edits survive a reload (per browser). Create freezes the
selected profile's current values.

Selecting a tag opens its task view:

- **Overview.** The frozen params (with the profile they came from and how
  they depart from it), progress, what the task's rented machines bill right
  now, the workers table, one add-worker form per role, the Machines card,
  and per-worker and task-level start/pause/remove. Only a non-running worker
  can be removed, so removal never silently discards an in-flight cycle.
  Deleting a tag removes the local data dir only (the bucket archive is kept)
  and is refused while the tag has workers.
- **Stats.** Generic per-role worker statistics driven by the role's stats
  schema: fleet-aggregate tiles (units/hour, totals, cycle time, worker
  health), Bokeh figures (units/hour over time, per-phase cycle-time
  breakdown), and a per-worker table. Each worker maintains a stats JSON under
  the tag (cumulative counters plus a bounded window of per-cycle samples).
  Bucket-delivering workers upload it every cycle and it arrives with the
  normal sync, so the dashboard reads only the local mount.
- **Workload tabs**, from the client registry `web/src/workloads.tsx`. The
  training workloads' analysis tabs are described in
  [react_dashboard.md](react_dashboard.md).

Output files carry a per-worker suffix on their stem. Names are per-machine
nanosecond timestamps, so the suffix makes them globally unique across workers
while keeping the stem-based pair matching downstream tools rely on.

## Machines

An ssh slot names its machine one of two ways: a bare host string typed into
the slot's form, or one of the task's **machines**. A machine is a record in
the task's `task.json` (`task.machines`) carrying the address, an optional
private key with its own known_hosts file, and its GPU count. Register one on
the Overview's Machines card (name, host, key file, GPUs); it can host any
number of the task's slots.

The reconcile pass probes a machine (ssh, then Docker) before its slots, and
leaves the slots alone until the machine is `up`. A GPU role is refused at add
time on a machine known to have no GPU; a bare host is not checked. GPU slots
on one machine share its GPUs (every container runs under `--gpus all`), as
local workers share the controller's. Removing a machine removes its slots,
subject to the same only-when-not-running rule as removing a slot directly.

### Renting a machine

The Machines card's **Rent** form launches an AWS instance for the task and
records it as a machine. The one-time account setup (IAM user, quotas, and
the key pair and security group `py/scripts/aws_setup.py` creates) and the
provider design are in [plans/cloud_machines.md](plans/cloud_machines.md).

- **Launch.** The type comes from a curated catalog (`py/cloud/providers/aws.py`,
  listing vCPUs, GPU, bundle arch and on-demand price). The instance boots
  AWS's stock Deep Learning GPU image with a 100 GB root volume, tagged as the
  task's, and runs a first-boot script that logs in to the image registry and
  pulls both worker images. The machine reads `launching` until ssh answers,
  `preparing` until that script finishes, then `up`.
- **Delivery.** Every slot on a rented machine delivers through the results
  bucket rather than being collected over ssh. The machine has a datacenter
  link to the bucket, and the controller downloads each chunk once instead of
  hauling it home and uploading it again.
- **Idle stop.** A rented machine on which nothing has run for ten minutes is
  **stopped**: its disk is kept and its hourly rate stops. A gated generator
  is a paused container and a finished trainer an exited one, so a run that
  ends stops its machine. Starting a slot on a stopped machine starts it
  again; it reads `launching` until it answers on its (possibly new) address.
- **Refusals.** When AWS refuses a launch or start (the account's vCPU quota
  for the family, no capacity in the zone), the machine's row shows the reason
  and what to do, and the request is retried with a growing delay.
- **Spot.** The **spot** box rents spare capacity at its market rate (shown
  beside each type, and recorded as the machine's rate). AWS may stop such a
  machine when it wants the capacity back and start it again later, so it
  behaves like an on-demand machine that was stopped and started. A trainer on
  it resumes from its own checkpoint, losing at most the generation in flight.
- **Remove** cancels any spot request, **terminates** the instance, and rolls
  its spend into the task's total. A machine whose instance is gone
  (terminated in the console, or a spot interruption that terminated it)
  reads `gone`, and its slots can be removed outright, since their containers
  went with the disk.
- **Orphans and the burn strip.** Every pass lists the instances tagged as
  ours. One that no task's machines name appears on the Machines card as an
  orphan with a Terminate button; it is never terminated automatically. The
  same listing feeds the **burn strip** pinned to the top of every dashboard
  page: the fleet's current hourly bill and, per instance, its type, state,
  uptime, rate and owning task (or "orphan"). It shows every instance
  whichever tag you are viewing, so a machine left running under a tag you
  moved on from stays visible. The strip is quiet when nothing bills, and
  turns amber when the listing has stopped refreshing or failed, rather than
  showing a zero it cannot vouch for.

### Preparing your own machine

A machine you own is prepared once, by hand:

- **SSH.** It must be reachable non-interactively from the dev container:
  key-based auth, no prompts. Authorize the container's own
  `~/.ssh/id_ed25519.pub`; devenv_utils persists that identity host-side, so
  authorizing it once holds across container relaunches. The form's host
  string is passed to `ssh` verbatim, so `user@host` and `~/.ssh/config`
  aliases both work, and the config file persists alongside the key.
- **Docker.** Installed, with the ssh user able to run it (in the `docker`
  group). A GPU role also needs the NVIDIA container toolkit, since its
  container runs with `--gpus all`; without it the container fails to start
  and the slot's exit reason says so.
- **Worker image.** Optional, but it turns a first Start from minutes into
  seconds: `docker login` (the image repo is private; see
  `registry.worker_image` in the credentials file), then
  `docker pull <worker image>`. Otherwise the dashboard pulls the image itself
  as a separate step before creating the first container, which is why that
  slot sits in `starting` for minutes. Containers themselves are created with
  `--pull=never`, so a missing image is an immediate, actionable error rather
  than a long pull blocking the dashboard.

Keep a laptop from sleeping on lid-close.

### Container lifecycle

**Bundles.** Code reaches a container as a bundle picked by the machine's CPU
arch (generic `x86-64` as fallback), fetched from the bucket at container
start ([cloud_compute.md](cloud_compute.md)). The dashboard deploys it for
you: when a task's first remote worker starts, it builds and pushes the
controller's tree and pins the task to the result. Every worker of a task
therefore runs identical code, and editing code mid-run does not change what
the fleet executes. The Overview badges the tree having moved on and offers
**Redeploy**, which repins the task and replaces its containers (a
container's bundle is fixed at creation). A paused slot the task has moved on
from is replaced the same way when next started. The reconcile loop restarts
a container that died (for example on a machine reboot), pulling the current
worker image as part of creating one, so a rebuilt image reaches the machine
without anyone logging in.

**Collection on your own machines.** A slot on a machine you own skips the
bucket. It delivers into its own container, and the reconcile pass reads
finished output back over the control link (`py/cloud/ssh_transfer.py`); only
the bundle fetch touches the bucket. Each pass collects a bounded batch, so a
backlog drains at a steady rate instead of each attempt moving everything
that has piled up, and the workers table shows what a container still holds.
Delivered chunks are deleted from the container only once they are safely on
the controller's disk.

**Draining before replacement.** Because output lives in the container until
collected, a container is replaced (after a redeploy) only once it has handed
everything over. One still holding output is started so the next passes can
drain it, then stopped and replaced once empty, with a final sweep of the
stopped container: stopping gracefully is what makes the worker flush its
last finished output. A container that will not stay up cannot be drained,
so it is restarted rather than replaced, however stale its bundle. Recovering
such a slot means discarding what it holds; that is the operator's decision,
taken in the workers table, where Remove states what would be lost. Removing
any slot discards whatever its container still holds, and the dashboard says
how much first.

**Bucket-delivering containers** (rented machines, and remote trainers) are
replaced outright: their outputs are already in the bucket, so a generator
loses only its in-flight chunk and a trainer its in-flight generation, coming
back on the bucket's last committed checkpoint. While a task has any such
slot, the server runs a sync watcher that streams their results into the
local mount.

**Pauses.** A scheduler gate pauses the container rather than stopping it, so
a gate that flips every minute costs nothing: no bundle refetch, and the
chunk in flight survives. An operator pause stops it cleanly, flushing
completed output.

## Server architecture

One Tornado process (`scribblez.dashboard.api`) hosts the control plane
alongside the read-only training data plane:

- `py/scribblez/dashboard/tasks.py`: task records, tag enumeration, progress.
- `py/scribblez/dashboard/workers.py`: the `WorkerManager`. Local subprocesses
  (spawn, interrupt, respawn; logs under the tag's `logs/`), ssh containers
  and rented machines via `py/cloud`, the per-task sync watcher, scheduler
  ticks, and gate enforcement.
- `py/scribblez/dashboard/worker_stats_figures.py`: the schema-driven Stats
  figures.
- `web/src/components/master/MasterApp.tsx`: the React shell (home page and
  task view). `py/scripts/dashboard.py` starts it alongside the API
  (`scribblez.dashboard.react_server`).

The API binds to localhost only, because it holds cloud credentials and
launches processes. The browser reaches it through the Vite dev server's
proxy, which the dev-container gateway serves.

The worker entrypoint (`py/cloud/worker_entrypoint.py`) is the one worker loop
for both kinds and every role. It is parameterized by a results sink (bucket
or local; `py/cloud/sinks.py`) and dispatches to the role's runner from the
workload registry. Runners cycle in a private per-worker work dir and deliver
whole output files through the sink.

## Not yet built

Live log streaming, run history, a second cloud provider, and volunteer
ingest.

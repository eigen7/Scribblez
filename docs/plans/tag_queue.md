# Plan: a tag queue over a machine pool

**Status: proposed and plan-reviewed (2026-09-25); three calls remain open
for the operator (see the dissent log).** Nothing here is built.
**Prerequisite:** #271 (match eval on rented machines, and finishing match
eval once training ends) must be merged first. It is open.

**Decision.** The dashboard gets one **machine pool** and one global,
ordered **tag queue**.

- The pool **owns** its machines: registered laptops, and instances it rents
  itself up to a cap the operator sets.
- A queued tag runs on a pool machine under a **lease**. The lease is
  exclusive: one tag per machine at a time.
- Each pass matches queued tags to free machines in queue order. A placed
  tag's slots come from its workload's **layout**, and a machine qualifies
  only if it passes the layout's resource checks.
- When every slot of the tag has finished and its output has been drained,
  the lease closes and the machine takes the next tag.

End conditions stay optional. "Run forever" becomes -1 in every workload,
and the dashboard warns when the queue holds a tag without an end
condition.

## Why

Tags are placed by hand today. The operator adds a machine (rented, or
registered by ssh) to one tag, adds its slots, starts them, and later
notices the tag has finished and does the same for the next one. A
campaign of twelve 10-hour runs on three machines, one of them rented,
means a human awake at every hand-off. There were two more problems on
the first day of the tuning campaign:

- A tag's trainer was put on asus-laptop, whose GPU has 3.7 GiB. The
  transformer trainer needs about 10 GiB at batch 256 without activation
  checkpointing, and about 4.1 GiB with it. It crashed with CUDA
  out-of-memory. Nothing checks that a machine fits the slots put on it.
- AWS quota allows one g6.2xlarge today. As more quota arrives, the number
  of machines grows, and hand placement becomes more work.

## Design

### 1. End conditions: optional, uniform, and warned about

A tag that never ends holds its machine until the operator releases it. An
open-ended run is legitimate, so an end condition stays optional. The
changes make it uniform and visible:

- **Workloads that have an end parameter declare it** (`param(end=True)`).
  Every role then finishes on its own once the end is reached. A trainer
  exits at its budget. Generators are finished by the scheduler once the
  trainer is done or the store is full (#270). Match eval is finished once
  nothing is owed and the trainer is done (#271). Workloads without an end
  parameter, such as kill_test, keep running until paused, and they cannot
  be enqueued until they have one.
- **"Run forever" is -1 in new tags, and it is the default.** The create
  form shows a "run forever" checkbox beside each end parameter.
- **Stored 0 stays a valid alias for -1. Nothing is migrated.** A stored 0
  already means "never" to every bundle in the field. Rewriting it to -1
  would hand -1 to containers still pinned to an older bundle, and older
  code reads -1 as "stop now". Accepting both avoids that and the flag day
  a migration would need.
- **Every end-parameter comparison goes through one helper.** The helpers
  are `params.unbounded(limit)` (true for -1 or 0) and
  `params.reached(value, limit)`. Today's call sites test 0 by truthiness
  or `== 0`, and each of them misreads -1 as "stop now". The worker then
  exits 0 and looks finished, so a missed call site turns a 10-hour run into
  a zero-row "success". PR 1 lists every call site (`max_rows`,
  `target_pairs`, `target_positions`, `train_epochs`, including the CLI
  mirrors and tests). A test per workload drives its run-until-end loop
  with -1 and checks that it does not exit.

| workload | end parameters | change |
|---|---|---|
| position_eval | `max_rows`, default 0 | default -1 |
| max_move_per_lane | `max_rows`, default 0 | default -1 |
| move_set_eval | `target_pairs` 600, `train_epochs` 20 | -1 accepted; defaults kept |
| evidence_trajectories | `target_pairs` default 0, `train_epochs` 20 | `target_pairs` default -1; its 15-idle-minutes "corpus final" rule applies to -1 and 0 |
| blind_spots | `target_positions` 100 | -1 accepted |
| kill_test | none | none for now; it is not queueable (§3) |
| match_arms | `pairs_per_arm`, finite | none |

**A tag is complete when every one of its slots has finished.** A tag
without an end condition completes only when the operator **releases** it
from its task view, which finishes all its slots.

**The warning.** Enqueueing asks for confirmation when the new tag, or any
tag already queued or placed, has no end condition. The confirmation lists
those tags and says that each will hold its machine until released by
hand. The queue view marks them with ∞.

### 2. The pool owns its machines

This reverses a deliberate rule. `MachineRecord` says a machine "belongs to
the task, living and dying with it like a slot, so nothing outside
task.json has to agree with it." That rule fits a machine used by one tag
for its whole life. A queue exists to move machines between tags, and
moving a task-owned machine means ownership transfers. A rented instance
would have to be retagged (the Provider protocol has no retag), its record
moved between two task.json files, and its known_hosts moved too. Between
release and the next placement it would belong to no task, so no idle
timer, spend accrual or orphan check would cover it.

So pool machines are owned by the pool:

- `pool.json` under the mount root holds the entries: **localhost**,
  **registered** machines (ssh host, key, arch, and the host aliases they
  answer to), and **rental capacity** (instance type, spot or on-demand,
  and a count cap). An instance rented for a capacity entry is recorded in
  the pool under a pool owner tag, `pool/<entry>/<n>`, with its known_hosts
  under `MACHINES_DIR/pool/`.
- **Instance lifecycle for pool instances is reconciled once, at pool
  level**: spend accrual, host refresh, idle-stop, termination, and orphan
  detection. `_owned()` also counts pool instances.
- **A lease binds a pool machine to one tag.** It is stored in `pool.json`
  (`lease: {workload, tag, phase, since}`). The tag's machine-backed slots
  name the pool machine. `MachineRecord` lookup resolves a slot's machine
  from the task's own list or from the pool, so the ssh, bundle, delivery
  and collection code keeps working on a `MachineRecord` either way.
- **Spend** accrues on the pool entry. The share accrued during a lease is
  added to the leasing task's spend, so a task's total stays right.
- **Task-owned machines remain** for everything outside the queue: a
  machine rented by hand for one tag works exactly as it does today.

Each entry records what eligibility needs: vCPUs, GPU count, **GPU
memory**, and a per-machine **GPU reserve** (memory taken by things no
slot represents, such as dashboard inference and dev-container tests on
localhost). For rented types these come from the catalog, which gains a
numeric `gpu_memory_gb`. For localhost and registered machines, one
`nvidia-smi` query runs when the entry is added. An entry can override the
generator thread count its layout would pick. There is no on/off switch:
the operator keeps the laptops available and manages that by hand.

**Busy.** A pool machine without a lease counts as busy while any slot of
any task on it is either desired `running` or observed alive. Paused and
finished slots do not count. This lets the operator keep hand-placing work
on a laptop without the queue double-booking it. A slot's host is matched
to a pool entry by its aliases, normalized by stripping `user@` and
resolving with `ssh -G`. That matters because today's tags spell
asus-laptop both as `asus-laptop` and as `dshin@asus-laptop`. A test runs
the busy rule against the current mount's tags.

### 3. Layouts, resources and eligibility

A queued tag has no slots until it is placed, because its slots depend on
the machine. A queueable workload supplies a **layout**:
`layout(params, machine) -> [SlotPlan(role, threads)]`. Only position_eval
gets one in this plan. Any other workload is refused at enqueue as "not
queueable yet", and gains its layout in its own PR when it is first queued.
The position_eval layout is:

- a trainer;
- a generator with the machine's vCPUs minus the trainer's reserve (its
  data-loader workers and the main process);
- match eval, if `match_every_generations > 0`.

A machine is **eligible** for a tag only when both checks pass:

- **Kinds and roles.** Every planned slot passes `_check_role` for the
  machine's kind (localhost is `local`; registered and rented machines are
  `ssh`), checked before anything is rented or created.
- **GPU memory.** The **sum** of the planned GPU roles' requirements must
  fit in the machine's GPU memory minus its reserve. The trainer and match
  eval share one GPU under `--gpus all`, so checking each role alone is not
  enough.

Requirements come from a **measured table**. A role's requirement is
keyed by the params that move its memory: trunk, batch size, widths,
`activation_checkpointing`, and optimizer (Muon keeps momentum). The value
is a measured peak plus headroom. A configuration with no measurement is
**not placed**: the queue shows "no memory figure for this config"
instead of guessing, and a queue entry can carry an explicit per-tag
override. PR 3 starts by measuring the table for every position_eval
configuration the campaign queues, using the same method as the
checkpointing benchmark.

**The same check guards hand placement.** When an operator adds a slot by
hand to a machine whose GPU memory is known, `_check_role` sums that
machine's GPU slots against their requirements. The asus-laptop OOM
therefore cannot recur either way.

A queue entry can narrow eligibility to named pool machines. The default is
any eligible machine.

### 4. The queue and placement

`queue.json` is an ordered, global list of `{workload, tag, machines,
memory_override}`, FIFO, which the operator reorders by dragging. A tag
enters from the create form ("Create & enqueue") or its task view
("Enqueue"). Enqueueing requires a queueable workload and a tag with no
slots, and it gives the warning of §1. Dependent tags, such as a student
whose `finalize` needs its teacher's export, are out of scope: they are
enqueued once their source has completed. An `after:` field is a possible
follow-up.

**Matching.** Each pass matches queued tags to free pool machines, in
queue order, with augmenting paths. An earlier tag may move to another
machine it is eligible for if that lets a later tag start. That way two
free machines never leave an eligible tag waiting. Suppose tag A is
eligible for both the laptop and the rental, and tag B only for the rental:
greedy placement gives A the rental and leaves B waiting. Matching gives A
the laptop and B the rental. Owned machines are tried before rental
capacity. A new instance is rented only for a tag that no free owned
machine can take, and only within the cap. The cap counts instances by
owner tag from the provider's listing, not only from `pool.json`.

**Placement is a journaled state machine** run on its own executor, not on
the reconcile pass's single blocking thread (the same reasoning as the
bundle-build executor). Renting and booting take minutes, and running them
inline would stall every Pause and Remove meanwhile. The phases are:

1. `reserved` is written to the lease before any provider call;
2. `renting` (capacity entries only) records the instance id as soon as the
   provider returns it;
3. `starting`: the slots are created from the layout and set running;
4. `running`.

On startup, recovery resumes or undoes each phase. A crash never leaves an
unclaimed billing instance, or a tag that is both queued and placed.
`pool.json` and `queue.json` use the `tasks.py` discipline: one shared
object per process, mtime-checked, atomically replaced.

**Code pinning.** A queued tag's bundle is built and pinned **at enqueue**,
for the archs of every eligible pool machine. Remote slots then run the
code the operator enqueued, not whatever the tree holds hours later.
Local slots run the live checkout today and would still do so (see the
dissent log).

### 5. Release, failure and hand-over

**Release drains before it removes.** When the tag is complete, the lease
enters a `releasing` phase:

1. For each stopped ssh container on the local sink, its last output is
   swept, and release requires `undelivered == 0`.
2. For bucket-delivering slots, one non-watching `cloud_sync` pass for the
   tag runs to completion, so the remote trainer's final export,
   checkpoint and records land locally.
3. Only then are the slots removed and the lease closed. The machine's
   lease-period spend goes to the task.

A tag whose final output cannot be drained keeps its lease, and the queue
view shows why.

**Failure is a state.** Today a crashed worker restarts forever, with
backoff for ssh slots and none for local ones. A slot becomes `failed`
after N consecutive non-zero exits within a window. A failed trainer marks
the placement failed, with the exit reason (for example the CUDA OOM
line), and pauses the tag's other roles. The release drain then runs, and
the machine goes to the next tag, so one broken tag cannot stall an
unattended queue. The failed tag leaves the queue with its data intact.
**Requeue** puts it back at the head, optionally with narrowed eligibility.

**Hand-over.** Because the pool owns the instance, handing a running
rented machine to the next tag is only a lease change. It keeps the booted
machine and the worker images on its disk. With nothing left to place, the
pool's idle rule stops the instance, then terminates it after the idle
timeout (a stopped instance still pays for its disk). The pool rents again
on demand.

A tag stays on its machine until it completes. Moving a running tag to
another machine is left out.

### 6. Dashboard

A **Queue** view shows:

- the pool: each entry's kind, GPU memory, lease and phase, and spend, with
  add, edit and remove forms;
- the queue: drag to reorder, eligibility, and ∞ marks;
- rental capacity: the cap, what is rented, and the cumulative spend.

The create form gains "Create & enqueue". The task view gains "Enqueue",
"Requeue" and "Release".

## Alternatives

- **Chaining** ("start B on A's machines when A finishes"). This is
  trivial, but it fixes each tag's machine in advance, which is the manual
  pattern that motivated this plan.
- **Task-owned machines moved between tags** (this plan's first draft).
  It was rejected in review because of the ownership transfer described in
  §2.
- **Packing several tags onto one machine** by CPU, GPU and memory. An
  earlier review of the unmerged machine-model draft (2026-09-10) rejected a
  general cross-task machine model because packing had no present value;
  `cloud_machines.md` itself only defers cross-task machines, without a
  design. Packing still has no value here: a position_eval tag saturates its
  machine's GPU with the trainer and its CPUs with the generator. The pool
  shares machines over time, never at the same moment.
- **An external scheduler (Slurm, Ray, Kubernetes).** It would duplicate
  the dashboard's slot, bundle and machine lifecycle, and it could not
  honour its per-kind delivery rules. That is a large dependency for a
  queue of a dozen tags.

## Landing

| PR | Content |
|---|---|
| 0 | #271 merged (prerequisite). |
| 1 | Uniform end conditions: `param(end=True)`, -1 as the new default with 0 as an accepted alias, `params.unbounded`/`reached` routed through every enumerated call site, a run-until-end test with -1 per workload, and the form checkbox. No migration. Independent of the rest. |
| 2 | Pool ownership: `pool.json` with the tasks.py discipline, localhost and registered entries with aliases, the GPU-memory probe and reserve, machine lookup through the pool, pool-level instance lifecycle and orphan accounting, the busy rule (tested against the current mount), the GPU-memory check in `_check_role`, and the catalog's `gpu_memory_gb`. |
| 3 | The measured requirement table, the position_eval layout, eligibility, `queue.json`, matching, the journaled placement on its own executor, pinning at enqueue, drain-then-release, the failed state, requeue and release. For localhost and registered machines: the laptops can then run the campaign unattended. |
| 4 | Rental capacity: renting on demand for tags no owned machine can take, the cap counted from the provider listing, lease hand-over of running instances, and stop-then-terminate when idle. |

## Review record

**Operator's comments on the first draft (PR #272):**

- **End conditions are optional, not required.** The default is -1 (run
  forever), and the dashboard warns when a tag without one is queued (§1).
- **One machine per placement is enough for now.** A tag that wants more
  (a rented trainer plus extra generator machines) takes hand-added slots
  alongside its placement.
- **The queue is global**, with eligibility doing the filtering.

**Plan review (2026-09-25).** The panel had four seats:

- hidden complexity (Claude, session tier);
- rival designer (codex, cross-vendor, with repo access);
- scope and integration (Claude Sonnet).

Every blocking or serious critique and its resolution:

| # | Critique (seat, severity) | Resolution |
|---|---|---|
| 1 | Flipping "never" to -1 breaks every truthiness check; -1 then reads as "stop now", and the worker exits 0 as if finished (hidden, blocking) | **Revised**: one helper, enumerated call sites, a -1 test per workload (§1). |
| 2 | Migrating stored 0s to -1 hands -1 to containers on old bundles, and the migration tool has no conditional map (hidden, blocking) | **Revised**: no migration; 0 stays an accepted alias (§1). |
| 3 | Release discards a finished ssh slot's last output, and stops the bucket watcher before a remote trainer's final artifacts are pulled (hidden, blocking) | **Revised**: drain-then-release with a `releasing` phase (§5). |
| 4 | Pool machines reverse the documented "the machine belongs to the task" invariant without saying so (integration, blocking) | **Revised**: the pool owns its machines, and tags hold leases (§2). |
| 5 | "Owner tag moves with it" needs a retag the Provider protocol lacks (integration, blocking) | **Resolved by 4**: pool instances carry a pool owner tag, so hand-over never retags. |
| 6 | Make the pool the permanent owner and tags lessees (rival, serious) | **Adopted** (§2). |
| 7 | A released rented machine belongs to no task, so nothing idles it, accrues its spend or tracks it (hidden, serious) | **Resolved by 4**: pool-level lifecycle (§2). |
| 8 | Greedy machine-order placement can leave an eligible tag waiting beside a free machine (rival, serious) | **Adopted**: queue-order matching with augmenting paths (§4). |
| 9 | Minutes-long placement would run on the single blocking reconcile thread (integration, serious) | **Revised**: its own executor, journaled (§4). |
| 10 | GPU-memory eligibility guards only queued tags, not hand placement (integration, serious) | **Revised**: the same check in `_check_role` (§3). |
| 11 | The busy rule makes both laptops permanently busy (paused slots), and host identity across tasks is unspecified (hidden, serious) | **Revised**: running or alive slots only; aliases normalized with `ssh -G`; tested on the current mount (§2). |
| 12 | Per-role memory ignores co-located GPU roles, and unmeasured configs are undefined (hidden, serious) | **Revised**: summed requirements minus a reserve, a measured table, and refusal of unmeasured configs (§3). |
| 13 | Eligibility ignores `RoleSpec.kinds` and the dispatch-on-rented refusal; the plan depends on unmerged #271 (hidden, serious) | **Revised**: `_check_role` runs on every planned slot before renting; #271 is PR 0. Verified: #271 is still open. |
| 14 | A queued tag runs whatever code the tree holds at placement time (hidden, serious) | **Revised for remote slots**: the bundle is pinned at enqueue. Local slots remain an **open human call** (below). |
| 15 | "Failed" is undefined; crashed workers restart forever (hidden, serious) | **Revised**: a durable failed state that releases the machine (§5). Whether to release or hold is an **open human call**. |
| 16 | queue.json and pool.json need the tasks.py concurrency discipline, and placement spans three files and a provider call (hidden, serious) | **Revised**: the tasks.py pattern, a journal, and the cap counted from the provider (§4). |
| 17 | Layouts for all seven workloads is speculative (scope, serious) | **Adopted**: position_eval only; others are "not queueable yet" (§3). |
| 18 | Adding kill_test's end condition only to satisfy an invariant (scope, serious) | **Adopted**: dropped, along with the rule that every workload must name one (§1). |

Minor critiques:

- **Adopted:**
  - the rent policy (owned machines first; rent only for tags no free owned machine takes);
  - dependent tags declared out of scope;
  - the misattributed citation to the cloud-machines review, corrected.
- **Rejected:**
  - *Defer rented hand-over* (scope). Under the lease model it is only a lease change and costs nothing extra.
  - *Hard-code the rental cap to 1* (scope). The operator asked for a cap they set, and quota is expected to grow; the cap is one number.
  - *Hand-edit pool.json and queue.json instead of forms* (scope). The operator drives the campaign from the dashboard, and reordering is the feature he asked for. The forms are small next to the placement logic.

**Open — human calls:**

1. **Code pinning of local slots.** Remote slots of a queued tag run the
   bundle pinned at enqueue. Local slots on localhost run the live
   checkout, as they do today, so a queued local arm picks up whatever is
   in `/workspace/repo` when it is placed. The options:
   - accept that and show a warning (cheap);
   - make local slots run from an unpacked copy of the pinned bundle (a
     real change to local spawning).
2. **What a failed tag does to its machine.** The plan releases it to the
   next tag, so the queue keeps moving unattended. The alternative is to
   hold the machine so the failure can be inspected in place.
3. **Scope growth.** The lease model makes PR 2 larger than the first
   draft's: machine lookup through the pool, and pool-level instance
   lifecycle. The rival designer and two other seats judged it necessary.
   The cheaper path is the first draft's ownership transfer, whose failure
   modes are critiques 4, 5 and 7.

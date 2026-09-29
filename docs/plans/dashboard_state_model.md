# Plan: one state model for the dashboard's control plane

**Status: proposed, not reviewed; operator's calls recorded 2026-09-29**
(§Operator's calls): a SQLite control store; machines rented and registered
only on the Machine pool page; Requeue and Release keep their names; moving a
tag's training state is an automatic step of placement.
Written after the first live run of the tag queue (#276-#280) needed eleven
follow-up PRs (#281-#291). The running campaign is unaffected. Each PR here
lands between campaign arms, with a dashboard restart.

**Decision.**

- Every thing the dashboard manages (a tag, a slot, a machine) has **one
  stored lifecycle state**. It changes only through named transitions, each
  with a guard and an effect. Nothing infers a state from several stores that
  can disagree.
- **One writer.** Only the reconcile thread changes control state. Request
  handlers submit commands to it. Helper threads return results for it to
  apply. Status reads come from a snapshot and change nothing.
- **Intent before action.** Before the dashboard acts on the world (stops a
  process, pauses a container, terminates an instance), it saves why. What it
  observes afterwards is read against that record, and the record survives a
  restart.
- **The pool owns every machine.** Machines are rented and registered only on
  the Machine pool page. A tag run by hand holds a *manual* lease on a pool
  machine, which the queue never touches. The task-owned machine model, with
  its separate lifecycle, goes away.
- **A tag's training state has a recorded home**, local or bucket. The queue
  moves it, as a step of placement, when it places a local tag on a rental.
- **Operator verbs that each say what happens to the machine,** offered only
  in the tag states where they apply.
- **A simulation test** drives random event sequences through the real
  control code, including dashboard restarts at any step, and checks the
  invariants in §7 after every step. It is the argument that the design has no
  races and survives restarts, instead of prose.

## Why

### What the first live run found

| Cause | PRs | Example |
|---|---|---|
| Tests reached live state through module-wide default paths, and subprocesses resolved paths on their own | #281, #287 | Deleted tag `a` kept coming back |
| A decision made once and never revisited | #286, #283 | A bundle decided only at enqueue, so capacity added later went unused |
| An observation read without knowing its cause | #288 | A gate stop (exit 143) counted as a crash, failing every tag on localhost after ~8 minutes |
| Where a tag's data lives was not modeled | #290 | Localhost-trained tags placed on rentals, which started over and would overwrite the local checkpoint |
| Two authorities over one thing | #284, #291 | Hand-added slots on a queued tag; Release freed a rental and the queue re-leased it at once |
| One serialized thread for everything | #289 | A tag's generation uploads froze every request for minutes |
| The operator could not see the model | #282, #285, parts of #284/#290/#291 | Which roles a tag gets; why it waits; what Release means; how to reach $0 |

The 33 findings from the review of #276-#280 were local correctness issues.
None of the rows above came up. They appear only when the pieces run
together, over time, with an operator acting on them.

### The concurrency today has no simple argument

The intended rule was that all mutation is serialized on the one blocking
thread (`WorkerManager.offload`). The code does not keep to it:

- **Handlers that change state on the event-loop thread:** `TaskCreateHandler`,
  `WorkerAddHandler`, and `TaskHandler`, the tag page's 3-second poll. The poll
  calls `worker_status`, which can mark a slot finished (`_note_finished`) or
  zero its undelivered count (`_holds_nothing`) on the shared cached record
  (workers.py `worker_status`).
- **A helper thread that writes shared records:** the bundle build thread
  sets `m.machine.arch` on pool machine objects that the blocking thread later
  saves (tag_queue.py `_submit_build`).
- **Shared cached objects:** `tasks.load_task` hands every thread the same
  cached `TaskRecord` object, and the drain thread reads it while reconcile
  changes it.
- **Several processes writing one tag dir:** local workers, `cloud_sync` (which
  pulls a bucket trainer's checkpoint and cursor), and the scheduler. Nothing
  states which process owns which file. The `tune-wd0.1` near-miss was a
  bucket pull about to replace a local checkpoint.

### What a restart forgets

These are held only in memory:
- **WorkerManager:** `_crashes`, `_stopped_local`, `_down_since`,
  `_idle_since`, `_stop_now`, `_exits`, `_restarts`, `_probes`,
  `_machine_states`, `_publishing`, `_pending_builds`.
- **TagQueue:** `_builds`, `_drains`, `_rent_refused`.
- **PoolRentals:** `_idle_since`.

Each was judged fine to lose when it was added, one at a time. No one has
judged the whole set. `_stop_now` is one lapse: after a restart, Stop all
cloud spending falls back to the 10-minute idle stop. `_stopped_local` is
correct only because a process started before the restart has no exit code
to read.

### A tag's state is inferred from three stores

Whether a tag is queued comes from `queue.json`. Whether it is placed comes
from a lease in `pool.json`. Whether it runs by hand comes from its slots in
`task.json`. They disagreed live: `tune-wd0.01` was "queued #2" while running
by hand. A tag's "failed" existed only as the lease's reason text, and
vanished when the lease closed.

## Design

### 1. Lifecycles, stored and explicit

**Tag.** One stored `state`, the only answer to "what is this tag doing":

```
            enqueue             place
    idle ───────────▶ queued ─────────▶ running ──── all slots finished, or Release ──▶ releasing(release) ──▶ done
    ▲ ▲ ◀── dequeue ──┘ ▲                 │  │
    │ │                 │                 │  └── Requeue ───▶ releasing(requeue) ───▶ queued (at the head)
    │ │                 └─────────────────┼───── (the machine goes to the next fitting tag)
    │ │                                   └── 3 crashes in 30 min ──▶ releasing(fail) ──▶ failed (machine held)
    │ └──────────────── Dismiss ─────────────────────────────────────────────────────────────────┘
    └── Unmanage ── manual ◀── Place by hand (from idle)
```

`placing` (the current lease phase `reserved`) is a sub-state of `running`
until the slots exist. Pause parks a running tag's slots without changing
its state, and keeps its machine. `releasing` records why it was entered
(release, requeue, fail, stop-all), and that reason decides where it goes
next. Invariants tie the stores together:

- A tag is `queued` if and only if it is in the queue order.
- A tag is `running`, `releasing` or `failed` if and only if exactly one lease
  names it.
- A `manual` tag has no queue entry. Any lease it holds is a manual one.

**Slot.** `desired_state` becomes `wanted ∈ {run, park, stop}` plus the
**intent** of the last stop the dashboard issued (§3). Its end is a stored
`outcome ∈ {finished, crashed, failed}`, set only by the classification rule:

- **finished:** it exited 0 with no stop intent recorded.
- **crashed:** it exited any other way with no stop intent recorded. A crash
  feeds the failure count.
- **An exit that matches a recorded intent** is the expected result of that
  action (a gate park, an operator pause, a requeue). It sets no outcome.

**Machine.** Every machine is a pool machine:

```
 registered ───────────────────────────────┐
 renting ──▶ booting ──▶ up ◀──▶ stopped (held: disk only)
                          │
                          ├── retire (stop-all, remove) ──▶ retiring ──▶ terminated
                          └── lease: none | queue(tag) | manual(tag)
```

A retiring machine takes no lease, and is terminated once no lease holds it.
The lease phases stay as they are today.

### 2. One writer

- **Every command goes through the reconcile thread.** Each handler that
  changes anything becomes a command, submitted and awaited, as `offload`
  already does for most of them. This includes task creation and adding a
  worker. The web server's thread never touches a record.
- **Status reads come from a snapshot.** At the end of every pass and command,
  the writer publishes an immutable snapshot (plain dicts). Reads serve it
  without waiting and without side effects. `worker_status` loses
  `_note_finished` and `_holds_nothing`, which move into reconcile.
- **Helper threads (build, upload, drain, provider calls) get immutable
  inputs and return results.** The writer applies each result on its next
  step. Arch detection returns the arch, and the writer records it.
- **Every tag file has one writer process,** listed in the doc next to the
  file:
  - `staging/` belongs to the workers.
  - `generations/` and their manifests belong to the scheduler.
  - The checkpoint and cursor belong to the local trainer when the tag's
    home is local, or to `cloud_sync` when it is bucket. It is never both,
    because the home decides (§5).

With one writer and snapshot reads, control state has no data races by
construction. What remains is the ordering of actions on the world, which §3
covers.

### 3. Intent before action

Every action on the world first saves an intent record `{action, target,
reason, at}` in the control store. Then it acts. The action is either
idempotent or re-checked against the intent when retried. This generalizes
what already works: the lease is written before the rental, the rental
carries an owner tag, and a generation is marked published only after
upload.

- **Exits** are classified against intents (§1). That is the general form of
  the #288 fix, and it survives a restart.
- **Stop all cloud spending** saves `retire` on each machine and `stop-now`
  on each task rental. It is correct across restarts.
- **In-memory state is allowed only if it is safe to lose:** observation
  caches and backoff timers. Each such field says what losing it costs, and
  the simulation test (§7) restarts at every step to check that claim.

### 4. The pool owns every machine

- **Machines come from one place, the Machine pool page:** rent, register,
  and rental capacity. The tag page's Machines card and rent form go. Instead,
  **Run by hand on…** picks any free pool machine (or rents one there and
  then) and gives the tag a *manual* lease on it. The queue never places on a
  manually leased machine, and never moves a manual tag. Every machine and
  every dollar is then on one page.
- **What goes away:**
  - task-owned `MachineRecord`s and `_reconcile_machines`;
  - the second idle policy (stop after 10 minutes, versus the pool's
    terminate when idle);
  - the task-versus-pool split in orphans, the burn strip and Stop all cloud
    spending.

  One lifecycle (§1) and one idle policy remain:
  - an unleased rental that nothing wants is terminated;
  - a held one is stopped, keeping its disk.
- **Adding slots by hand** is allowed only for a `manual` tag, on its manually
  leased machine. That makes the #284 refusal structural, instead of a check.

### 5. Where a tag's training state lives

- **Recording the home.** The tag's stored `home ∈ {local, bucket}` replaces
  `trainer_sink` (#290). It is set when the first trainer slot is created, and
  changed only by **move home**.
- **Move home: local to bucket, done by the queue.** When the queue places a
  local tag on an ssh machine, placement first moves its home: upload the
  checkpoint, the cursor and the current window of generations, confirm, then
  flip `home`, and only then start the slots. Measured on `tune-wd0.01`, that
  is about 170 MB (a 117 MB checkpoint plus four generations of about 12 MB).
  That is seconds to minutes once per tag, against an arm of about four
  hours. The queue row shows the move while it runs. The move is a command
  with an intent (§3), so an interrupted move resumes or rolls back. Moving
  from bucket to local is the pull `cloud_sync` already does, followed by the
  flip.
- **Why automatic.** The capacity cap is already the operator's decision to
  rent. A second, manual step to let a tag use a rental would bring back the
  "why won't my tag go there" confusion the live run hit. A tag the operator
  wants kept on localhost says so with its queue entry's machine list, which
  already exists.
- **Pulls never regress progress.** `cloud_sync` refuses to replace a
  checkpoint whose cursor is ahead of the incoming one. This guards the
  failure mode behind the `tune-wd0.1` backup.

### 6. Operator verbs

| Verb | From state | Tag goes to | Machine |
|---|---|---|---|
| Enqueue / Dequeue | idle / queued | queued / idle | — |
| Pause / Resume | running | running (slots parked) | kept |
| Requeue | running | queued, at the head | freed for the next tag |
| Release | running | done (slots finished) | freed for the next tag |
| Dismiss | failed | idle (data kept, re-enqueue when fixed) | the held machine is freed, or retired if rented |
| Place by hand / Unmanage | idle / manual | manual / idle | manual lease taken / released |
| Remove machine | — | — | only when unleased; a rental is retired |
| Stop all cloud spending | any | tags on rentals requeued | every rental retired; caps to 0 |

Requeue and Release keep their names (operator's call). Release is the one
that misled in the live run: it reads as "release the machine", but means
"this tag is done, give its machine to the next tag". So both buttons get a
tooltip and a confirmation that say what happens to the tag and to the
machine, and the Machine pool page points at Stop all cloud spending for
"I want this machine gone". The tag page shows the tag's one `state`, and
offers only the verbs valid in it.

### 7. The simulation test

A harness runs the real `TagQueue`, `WorkerManager`, `PoolRentals` and
scheduler against fakes:
- a clock;
- a provider (launch, stop, terminate, spot interruption, listing failure);
- ssh and containers;
- local processes (exit codes, SIGTERM handling);
- the bucket (slow uploads, failed uploads).

A seeded driver draws events: a pass, an operator verb, a worker exit (clean,
crash, OOM), a gate flip, an instance vanishing, a listing failure, a slow
upload, and a **dashboard restart**, which drops all in-memory state and
rebuilds from disk. After every step it checks:

- **I1.** Every tag is in exactly one state, and the queue, lease and slot
  stores agree with it (§1).
- **I2.** No machine holds more than one lease. A retiring machine never
  gains one.
- **I3.** Every billing instance is a pool machine or a reported orphan.
  After Stop all cloud spending, the burn reaches $0 within K steps unless the
  operator acts.
- **I4.** No slot is `crashed` without an exit that lacks an intent. No slot
  is `finished` without exit 0 and no intent.
- **I5.** A tag's cursor (rows trained) never decreases at its home.
- **I6.** A queued tag with a fitting free machine is placed within K steps.
  Otherwise its queue row states a reason.
- **I7.** Status reads leave the stored state byte-for-byte unchanged.
- **I8.** A run with a restart inserted at any step reaches the same states
  as the same run without it, allowing for timing.

A failing seed reproduces the failure exactly. A few hundred sequences of a
few hundred steps should run in seconds in CI. Most of the fakes exist in
`test_pool_rentals.py`, `test_tag_queue.py` and `test_worker_manager.py`. Each
bug from #281-#291 would have broken at least one invariant: #286 breaks I6,
#288 breaks I4, #290 breaks I5, #291's trap breaks I3, and #284 breaks I1.

### 8. Smaller items

- **One paths context.** No path defaults to the real mount root: the
  dashboard builds one context from `--mount-root` and passes it down, and
  subprocesses get it as an argument. This removes the class of leak behind
  #281 and #287, instead of fencing it off in `conftest.py`.
- **A stale-code banner.** The dashboard records the source hash it started
  from, and says so when the checkout has moved on ("restart to pick up merged
  changes"). Every fix from #289 on waited on a restart nobody knew was needed.
- **Enqueue refuses a finished tag,** one whose end condition is reached.

## Alternatives

- **Keep fixing incrementally.** It costs least now. But each feature adds
  interactions, and nothing checks them together: this session's bugs were
  all interactions. Even if everything else here is rejected, **the
  simulation test (§7) is worth landing on its own**, against the current
  code, with the known violations marked as expected failures.
- **Control store: JSON files (rejected) or SQLite (chosen).** Transitions
  that span stores (placement writes a lease, then a tag state, then the
  queue) need either ordered writes plus a startup repair pass, or
  transactions. JSON would keep today's files and tools
  (`migrate_tag_params.py`, hand inspection), at the cost of a repair pass for
  each transition that spans stores. One SQLite control database (tags,
  slots, leases, queue, machines, intents) makes each transition one
  transaction, and the repair pass disappears. `migrate_tag_params.py` moves to
  the database. Per-tag data (generations, checkpoints, metrics in
  `dashboard.db`) stays where it is.
- **Actors or locks instead of one writer.** Finer-grained concurrency buys
  nothing at this scale (dozens of tags, a pass every few seconds). A single
  writer is the model that fits in one sentence.
- **A workflow engine (Temporal and the like).** Too much machinery for one
  operator and one controller.

## Landing

Each PR keeps the simulation green on the invariants it claims, and lands
between campaign arms with a dashboard restart.

1. **The simulation harness and invariants**, against the current code. Known
   violations are marked as expected failures. This is the baseline.
2. **One writer:** commands, snapshots, reads with no side effects, helper
   threads returning results. Fixes I7, and the race sites listed under "Why".
3. **Intents:** stored, with exits classified against them. Fixes I4, and the
   restart gaps.
4. **Explicit tag state and the SQLite control store,** with a one-time
   migration from `task.json`, `queue.json` and `pool.json`. Fixes I1.
5. **The pool owns every machine:** the tag page's Machines card and rent form
   go, Run by hand on… arrives, and existing task machines are migrated into
   the pool with manual leases. This is the largest PR.
6. **Operator verbs and the tag page:** one `state`, and only valid verbs.
7. **Tag home, and the queue's automatic move home,** plus the no-regress rule
   for pulls. Fixes I5 fully, and lets localhost tags continue on rentals.
8. **The paths context, the stale-code banner, and refusing to enqueue a
   finished tag.** These are independent and can go any time.

## Operator's calls (2026-09-29)

1. **The control store:** one SQLite database (§Alternatives).
2. **Renting from a tag's page:** left to this plan, which removes it.
   Machines come only from the Machine pool page, and a tag page runs a tag by
   hand on a pool machine (§4).
3. **Verb names:** Requeue and Release stay. Both get a tooltip and a
   confirmation that say what happens to the machine (§6).
4. **Move home:** automatic, as a step of placement (§5), confirmed after
   weighing the measured cost (about 170 MB per tag) against a manual step.

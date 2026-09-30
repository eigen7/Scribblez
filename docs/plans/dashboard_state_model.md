# Plan: one state model for the dashboard's control plane

**Status: plan-reviewed 2026-09-29 and revised (§Review record); every
operator's call settled (§Operator's calls); being built: PR 0 (the paths
context) landed as #294; PR 1 (the schema and the shadow import) as #295 and
#297; PR 2 (the fake world and the simulation) as #299; PR 3a (one writer)
as #301; PR 3b (the control store) in review.**

- **Review.** Four independent panelists: hidden complexity, a rival design
  (a Codex seat), scope, and integration. They raised 2 blocking, 16 serious
  and 8 minor critiques; all were revised in, and none was rejected outright.
- **The largest change from the draft:** control state is now normalized
  records under SQLite constraints, with an outbox of operations. The tag
  state the operator sees is a projection of those records, not one stored
  enum.
- **Background.** Written after the first live run of the tag queue
  (#276-#280) needed eleven follow-up PRs (#281-#291). The running campaign
  is unaffected, and each PR here lands between campaign arms, with a
  dashboard restart.

**Decision.**

- **One SQLite control database, normalized, with constraints.** It holds
  five kinds of record:
  - **Tag:** what the operator wants, the terminal result, and the home of
    its training state.
  - **Machine:** the provider identity and observed condition.
  - **Assignment:** links a tag to a machine, either `queue` or `manual`.
  - **Slot:** what it should do, what was observed, and its outcome.
  - **Operation:** a durable record of every action on the world.

  Constraints make the key invariants impossible to break: at most one queue
  assignment per machine and per tag, and never a queue assignment beside a
  manual one. The tag states the operator sees (queued, running, releasing,
  ...) are a documented, deterministic projection of these records, specified
  by the diagram in §1. Nothing else infers a state.
- **One writer.** Only the reconcile thread writes the database. Request
  handlers submit commands. Helper threads return results for the writer to
  apply. Status reads come from a snapshot of control state and change
  nothing. The constraints hold even if a second writer slipped in, and a
  check fails any write made off the writer thread.
- **Operations before action.** Every action on the world is recorded as an
  operation before it runs: stop a process, pause a container, terminate an
  instance, move a tag's home. The operation has a status and a result, and
  whatever the dashboard observes later is read against it. A local worker
  reports its exit durably, so that reading works across restarts.
- **The pool owns every machine.** Machines are rented and registered only
  on the Machine pool page. A tag run by hand holds a *manual* assignment,
  which can share a machine with other manual assignments. A *queue*
  assignment is exclusive. Idle policy follows who rented the machine: the
  queue's rentals are terminated when idle, and the operator's are stopped.
- **A tag's training state has a recorded home**, local or bucket. The queue
  moves it, as a phase of placement, when it places a local tag on a rental.
  Pulls never replace newer training state with older.
- **Requeue and Release keep their names.** Every verb says what happens to
  the machine, and is offered only in the tag states where it applies.
- **A simulation over a fake world** (processes, containers, provider,
  bucket, clock) drives random events and dashboard restarts through the real
  control code, and checks the invariants in §7 after every step. It checks
  orderings and restart behavior. Freedom from data races comes from the
  one-writer structure.

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
- **WorkerManager:** `_crashes`, `_down_since`,
  `_idle_since`, `_stop_now`, `_exits`, `_restarts`, `_probes`,
  `_machine_states`, `_publishing`, `_pending_builds`.
- **TagQueue:** `_builds`, `_drains`, `_rent_refused`.
- **PoolRentals:** `_idle_since`.

Each was judged fine to lose when it was added, one at a time. No one has
judged the whole set. `_stop_now` is one lapse: after a restart, Stop all
cloud spending falls back to the 10-minute idle stop.

### A tag's state is inferred from three stores

Whether a tag is queued comes from `queue.json`. Whether it is placed comes
from a lease in `pool.json`. Whether it runs by hand comes from its slots in
`task.json`. They disagreed live: `tune-wd0.01` was "queued #2" while running
by hand. A tag's "failed" existed only as the lease's reason text, and
vanished when the lease closed.

## Design

### 1. Records, constraints, and the projection the operator sees

**Records** (one SQLite database; §9 lists what stays in the tag directory):

| Record | Holds | Constraints |
|---|---|---|
| Tag | workload, name, operator desire (`idle`, `queued` with a queue position, `manual`), terminal result (`done`, `failed` with its reason), `home` (`local` or `bucket`) | one row per tag; it decides whether the tag exists |
| Machine | name, kind, provider identity, owner (`pool` capacity or `operator`), observed condition, `retiring` | name unique |
| Assignment | tag, machine, kind (`queue` or `manual`), phase (`reserved`, `moving`, `running`, `releasing`, `held`), release reason | queue assignments: at most one per machine and one per tag; no machine has a queue and a manual assignment at once; manual assignments may share a machine |
| Slot | tag, role, machine, `wanted` (`run`, `park`, `stop`), observations, `outcome` (with its source) | its machine is one its tag is assigned to |
| Operation | kind, target, reason, status (`pending`, `running`, `done`, `failed`), created and finished times, result | its target exists |

**The operator's tag state is a projection,** computed the same way
everywhere from these records:

```
            enqueue             place
    idle ───────────▶ queued ─────────▶ running ──── all slots finished, or Release ──▶ releasing(release) ──▶ done
    ▲ ▲ ◀── dequeue ──┘ ▲                 │  │
    │ │                 │                 │  └── Requeue ───▶ releasing(requeue) ───▶ queued (at the head)
    │ │                 └─────────────────┼───── (the machine goes to the next fitting tag)
    │ │                                   └── 3 crashes in 30 min ──▶ releasing(fail) ──▶ failed (machine held)
    │ └──────────────── Dismiss ─────────────────────────────────────────────────────────────────┘
    └── Unmanage ── manual ◀── Run by hand on… (from idle)
```

- **The projection rules:**
  - `queued` means the desire is queued, with no assignment.
  - `running` means a queue assignment in `reserved`, `moving` or `running`.
  - `releasing` means an assignment in `releasing`, and shows its reason.
  - `failed` means a `held` assignment or a failed result.
  - `manual` means the desire is manual.
- **Pause** parks slots (`wanted = park`), and the projection shows it as
  "running, paused". Sub-conditions like this are not further states.
- **The constraints above make the old cross-store disagreements
  impossible.** "Queued #2 while running by hand" would need a queued desire
  and a manual desire at once. "Two tags on one rental" would need two queue
  assignments on one machine.

**Machine lifecycle:**

```
 registered ────────────────────────────────┐
 renting ──▶ booting ──▶ up ◀──▶ stopped (disk only: held, or idle and operator-rented)
                          │
                          ├── retire (Stop all cloud spending, Remove) ──▶ retiring ──▶ terminated
                          └── assignments: none | one queue | manual ones
```

A retiring machine gets no new assignment, and is terminated once none
holds it.

**Slot outcome.** Each outcome is set by exactly one source (§3):
- `exited`: finished (exit 0) or crashed.
- `scheduler`: the role finished at its end condition.
- `released`: Release finished the tag.
- `lost`: the worker disappeared with no exit on record.

### 2. One writer

- **Every command goes through the reconcile thread.** Each handler that
  changes anything becomes a command, submitted and awaited, as `offload`
  already does for most of them. That includes `TaskCreateHandler`,
  `WorkerAddHandler`, and the mutation hidden inside `TaskHandler`'s poll.
  `worker_status` loses `_note_finished` and `_holds_nothing`, which move into
  reconcile.
- **Snapshots cover control state only.** After each command's transaction
  and each pass, the writer publishes an immutable snapshot of the control
  records it changed. It is incremental, not a full rebuild, so the snapshot
  cannot bring back the #289 freeze. File-derived, read-only data stays
  computed by the read handlers directly: progress, last activity, bundle
  drift, stats.
- **Helper threads get immutable inputs and return results.** This covers
  builds, uploads, drains and provider calls. The writer applies each result
  in a transaction. Arch detection, for example, returns the arch.
- **Every tag file has one writer process:**
  - `staging/` belongs to the workers.
  - `generations/` and their manifests belong to the scheduler, and so do
    their uploads (§5).
  - The trainer outputs (checkpoints, cursor, records, models) belong to the
    local trainer when home is `local`, and to `cloud_sync` when it is
    `bucket`.
- **The argument.** One writer, snapshot reads and database constraints give
  control state no data races by construction. A debug check fails any
  database write made off the writer thread. What remains is the ordering of
  actions on the world, which §3 covers and §7 tests.

### 3. Operations: record before acting

Every action on the world is an **operation** row, written in the same
transaction as the decision to act. Only then is it carried out, outside the
transaction, by the writer or a helper thread. Its result is recorded in a
second transaction. An operation either is idempotent, or is checked against
the world before a retry. This is the outbox pattern. It generalizes what
already works: the lease is written before the rental launch, and a
generation is marked published only after its upload. It also gives
multi-step work (move home, §5) an explicit protocol.

- **Exit classification.** An exit matches a stop operation only if the
  exit happened after that operation was created. The exit's time comes from
  the container's `FinishedAt`, or from the local exit record below. A stop
  operation is closed by the next `wanted = run`. An exit that matches a stop
  operation is the expected result of that action (a gate park, a pause, a
  requeue), and sets no outcome. Otherwise, exit 0 is `finished` and anything
  else is `crashed`, from source `exited`.
- **A durable exit channel for local workers.** Today a local worker's exit
  code exists only in the `Popen` of the process that spawned it. It is
  unreadable after a restart (`_local_exit_code` returns None). Instead, the
  worker entrypoint writes `{code, at}` to a per-slot exit file as it exits. A
  worker gone with no exit file is `lost`.
- **Dashboard shutdown** records a stop operation for each local slot before
  it signals them. The restart that every PR here requires is then an
  expected stop, not a crash or a loss.
- **Crash history is a table,** so a restart neither forgets it nor
  double-counts it.
- **In-memory state is allowed only if it is safe to lose:** observation
  caches and backoff timers. Each such field says what losing it costs.

### 4. The pool owns every machine

- **Machines come from one place, the Machine pool page:** rent, register,
  and rental capacity. The tag page's Machines card and rent form go.
  **Run by hand on…** picks a pool machine (or rents one there and then), and
  gives the tag a manual assignment.
- **Manual assignments share.** Several hand-run tags can use one machine,
  as several did on localhost and asus-laptop before the queue. The queue
  never places on a machine that has a manual assignment, and never moves a
  manual tag. The old co-tenant concept (`occupants`, `_holds_machine`)
  becomes "the machine's manual assignments".
- **Idle policy by owner:**
  - A rental made under a capacity entry is terminated when unassigned and
    wanted by nothing.
  - A rental the operator made is stopped when idle, disk kept. The operator
    chose its lifetime.
  - A registered machine is left alone.
- **Migrating task-owned machines (PR 5a)** is a port, not a deletion:
  - **Retag instances.** A task rental's instance carries the owner tag
    `<workload>/<tag>/<machine>`, and the pool finds its own by `pool/<name>`.
    PR 5a adds `Provider.retag` (EC2 `CreateTags`), and retags each migrated
    instance to `pool/<name>`, owned by `operator`.
  - **Port the task path's lifecycle into the pool:** stop and start, a
    moved host on restart (`_moved_host`), and spend accrual across stops.
    Today `PoolRentals` only launches and terminates.
  - **Bare-host slots** (`host=` strings with no machine record) become pool
    machines, matched through `canonical_host` aliases. Slots that already
    share a machine become manual assignments on it.

### 5. Where a tag's training state lives, and moving it

- **The tag's `home`** replaces `trainer_sink` (#290). It is set when the
  first trainer slot is created, and changed only by the move below.
- **New work: pulls never regress progress.** This is not current behavior:
  `cloud_sync.sync_once` copies every target unconditionally. `cloud_sync`
  must compare the local and bucket cursors (`rows_trained` in
  `train_state.json`) before pulling the checkpoint, the cursor, `records/`
  or `models/`. When the bucket's cursor is behind, it pulls none of them and
  reports why. A missing local cursor counts as behind. On a tie it pulls,
  since the same run produced both.
- **Move home, local to bucket, as a placement phase.** Today `_place` leases
  a machine and starts slots in one pass. A queue assignment for a local tag
  on an ssh machine instead enters `moving` between `reserved` and
  `running`. `_advance_lease` polls the move operation each pass (§2's result
  pattern) and calls `_start_slots` only once the move is done. `_start_slots`
  stays idempotent, so a restart mid-move resumes the operation. The steps,
  each a sub-step recorded on the operation:
  1. **Fence the bucket's stale trainer outputs.** Move any earlier bucket
     run's `records/`, `models/`, `checkpoints/` and `train_state.json` to an
     archive prefix. Without this, a pull would bring back stale records,
     and the `rclone sync` of `models/` would delete local exports the bucket
     lacks.
  2. **Upload the whole trainer tree:** checkpoints, cursor, `records/` and
     `models/`.
  3. **Publish generations through the scheduler's path,** not a second
     uploader. The publish hook is enabled when `home = bucket` or a move is
     pending, not only when bucket slots exist. The scheduler's published
     markers are then the only record of what was uploaded.
  4. **Verify** that the bucket's trainer tree matches the local one.
  5. **Flip `home`, and start the slots.**

  A failed step leaves the assignment `moving`, shows the step and error on
  the queue row, and retries. Dequeue or Requeue abandons the move. The home
  flips only at step 5, so the local tree stays authoritative until then.
- **Size.** On `tune-wd0.01` the window plus trainer tree is about 170 MB
  (a 117 MB checkpoint plus four generations of about 12 MB). A tag whose
  earlier generations were never published uploads more. PR 7b measures that
  case before landing.
- **Placement stops refusing ssh machines for local tags.** The #290 branch
  in `placement.refusal` (a local home on an ssh machine) becomes eligibility
  with a cost: "would move its training state (N MB)". The queue plan panel
  and `_note_refusals` show that line instead of a refusal. PR 7b checks all
  eight `refusal` call sites in `tag_queue.py` against the new meaning.
- **Moving from bucket to local** is the pull `cloud_sync` already does,
  under the no-regress rule, followed by the flip.

### 6. Operator verbs

| Verb | From state | Tag goes to | Machine |
|---|---|---|---|
| Enqueue / Dequeue | idle / queued | queued / idle | — |
| Pause / Resume | running | running, paused | kept |
| Requeue | running | queued, at the head | freed for the next tag |
| Release | running | done (slots finished) | freed for the next tag |
| Dismiss | failed | idle (data kept; re-enqueue when fixed) | the held machine is freed, or retired if the queue rented it |
| Run by hand on… / Unmanage | idle / manual | manual / idle | manual assignment taken / dropped |
| Remove machine | — | — | only when unassigned; a rental is retired |
| Stop all cloud spending | any | tags on rentals requeued | every rental retired; caps to 0 |

- **Requeue and Release keep their names** (operator's call).
- **Release** misled in the live run: it reads as "release the machine", but
  means "this tag is done, give its machine to the next tag". Both buttons
  get a tooltip and a confirmation that say what happens to the tag and to
  the machine. The Machine pool page points at Stop all cloud spending for
  "I want this machine gone".
- **The tag page shows the projected state,** and offers only the verbs
  valid in it.
- **The tooltip and confirmation copy doesn't gate any invariant,** and can
  land on its own.

### 7. The simulation

**A fake world, which is new work.** Today's tests mostly stub the code
under test (they monkeypatch `all_tasks`, `_submit_build`, `_advance_lease`,
`worker_pid_alive`), and none simulates a world. PR 2 builds the pieces:
- a fake `Popen` and `/proc` pid table, since `worker_pid_alive` reads
  `/proc`, plus local exit files;
- a fake `SshMachine`, whose containers change state and report
  `FinishedAt`;
- a fake provider with instance lifecycles, spot interruptions and listing
  failures;
- a fake `rclone` bucket;
- a clock, threaded through the bare `time.time()` calls;
- deterministic executors in place of the three thread pools, so a seed
  replays.

**The driver** draws seeded events: a pass, an operator verb, a worker exit
(clean, crash, OOM, or vanishing with no exit record), a gate flip, an
instance vanishing, a listing failure, a slow or failed upload, a stale
earlier run left in the bucket, and a **dashboard restart**. A restart drops
all in-memory state (the stores' live objects and readers' copies, and the
module-level `pool._canonical`), then rebuilds from the control database and
the tag trees.

**What it checks, after every step:**
- **I1.** The projection gives each tag exactly one state. The database
  constraints hold.
- **I2.** No machine has more than one queue assignment, or a queue
  assignment beside a manual one. A retiring machine never gains an
  assignment.
- **I3.** Every billing instance is a pool machine or a reported orphan.
  After Stop all cloud spending, the burn reaches $0 within K steps unless
  the operator acts.
- **I4.** Each slot outcome has a source that allows it. `finished` or
  `crashed` from `exited` requires an exit after no matching stop operation.
  `scheduler` and `released` require that finish. `lost` requires a vanished
  worker with no exit record.
- **I5.** No training state regresses at its home. That means the cursor
  (rows trained), and no local export or record lost or replaced by an older
  one.
- **I6.** A queued tag with a fitting free machine is assigned within K
  steps. Otherwise its queue row states a reason.
- **I7.** Status reads leave the database byte-for-byte unchanged.
- **I8.** Convergence after a restart. After a restart and K quiescent steps,
  a named projection equals the same run's projection without the restart:
  tag state, the assignment holder, home, and slot outcomes. The events are
  drawn independently of timing, so a lost backoff cannot shift the event
  stream.

**What the simulation checks, and what it doesn't.** It checks orderings and
restarts at step granularity. It does not see thread interleavings, because
its executors are deterministic. Freedom from data races rests on §2's
structure and its writer-thread check.

**What each earlier bug would have broken:**
- #286 breaks I6.
- #288 breaks I4.
- #290 breaks I5.
- #291's trap breaks I3.
- #284 breaks I1.

### 8. Smaller items

- **A stale-code banner.** The dashboard records the source hash it started
  from, and says so when the checkout has moved on ("restart to pick up
  merged changes"). Every fix from #289 on waited on a restart nobody knew
  was needed.
- **Enqueue refuses a finished tag,** one whose end condition is reached.

### 9. What moves to SQLite, and what stays in the tag directory

- **Stays in `task.json`:** the frozen params, the profile, the bundle pin,
  and the source hash. These are read outside the dashboard, by
  `backfill_placement_eval.py`, `match_arms.py`, `trajectories_api.py`, and
  the per-worker `params/*.json` copies that `migrate_tag_params.py` edits
  together with `task.json`. That tool keeps working unchanged.
- **Moves to SQLite:** slots, gates, retired spend, desire, result, home,
  assignments, machines, the queue, operations and crash history.
- **The database row decides whether a tag exists.** `delete_tag` is a
  transaction that removes the row, followed by removing the directory. If
  the directory removal fails, the next pass finishes it. A tag directory
  restored from a copy is re-registered by a command; it is never picked up
  implicitly.

### 10. Migrating the three JSON stores

PR 1 imports the stores into the database in **shadow mode**. The JSON files
stay authoritative. Each pass re-imports them and compares the projection
with the current behavior, logging any disagreement. The decision table for
one tag:

| Queue entry | Queue lease | Slots | State |
|---|---|---|---|
| yes | no | none | queued |
| no | yes | any | running (or releasing, held, from the lease phase) |
| no | no | some | manual (a manual assignment on each machine its slots use) |
| no | no | none | idle |
| yes | yes | any | running; the queue entry is dropped (the repair `_drop_stale_entries` already does) |
| yes | no | some | the `tune-wd0.01` case: refused, and shown to the operator to settle (dequeue, or remove the slots) before the switch |

**Before the switch:**
- The table runs in dry-run mode against a copy of the live mount before
  PR 3 flips the authority.
- PR 3 ships a database-to-JSON export, so the switch can be rolled back to
  the last good state.

## Alternatives

- **One stored lifecycle enum per tag, slot and machine** (the draft). It was
  replaced after review. Its "one state" was several independent facts in
  disguise: running also carried placing and paused sub-states, and a machine
  had both a lifecycle and a lease phase. The normalized records with
  constraints (§1) represent those facts directly. The operator still sees
  one state per tag, as a documented projection, so the draft's clarity
  survives where it matters.
- **Keep fixing incrementally.** It costs least now. But each feature adds
  interactions, and nothing checks them together: this session's bugs were
  all interactions.
- **Control store: JSON files (rejected) or SQLite (chosen).**
  - JSON would need ordered writes plus a startup repair pass for each
    transition that spans the three files.
  - SQLite makes each transition one transaction, and lets the key
    invariants be written as constraints.
- **Actors or locks instead of one writer.** Finer-grained concurrency buys
  nothing at this scale (dozens of tags, a pass every few seconds). The
  constraints mean correctness no longer depends on the writer being the only
  one, but one writer is still the simplest model.
- **A workflow engine (Temporal and the like).** The operation outbox (§3)
  is the small subset this needs.

## Landing

Each PR lands between campaign arms with a dashboard restart. It keeps the
simulation green on the invariants it claims, from PR 2 on.

0. **The paths context.** No path defaults to the real mount root: the
   dashboard builds one context from `--mount-root` and passes it down, and
   subprocesses get it as an argument. Everything later, including the
   database file and the simulation's scratch roots, takes its paths from it.
   This removes the class of leak behind #281 and #287.
1. **The SQLite schema, and a shadow import.** JSON stays authoritative.
   Each pass imports into the database and compares the projection with
   current behavior (§10). This PR also runs the migration's dry run against
   a copy of the live mount. Nothing changes behavior.
2. **The fake world and the simulation (§7),** driving the current
   controller. The invariants are read from the shadow projection, so they
   carry through the switch unchanged. Known violations are marked as
   expected failures.
3. **The switch,** in two parts:
   - **3a. One writer, over the JSON stores.** Every change a handler makes
     is a command on the writer thread; status reads run on the event loop
     against each store's last committed copy and change nothing; a save off
     the writer thread fails, and so does saving a reader's copy. Helper
     threads (builds, drains) get copies and return results. The committed
     copy is the file as last saved, decoded once per save for readers: a
     per-store snapshot, never a full rebuild. Fixes I7 and the race sites
     under "Why".
   - **3b. SQLite authoritative.** The records move into the control
     database, one row each (the pool, the queue, each tag's control state)
     with the frozen params left in `task.json`; a transition that spans
     records (placing a tag, finishing a release, a requeue) commits in one
     transaction; readers read over a connection of their own. The dashboard
     imports the JSON files on its first start, and a database-to-JSON
     export arrives for rollback.
   - **3c. The constraints enforced.** The normalized tables are rebuilt in
     each committing transaction, and a commit that would break a
     constraint is refused. Fixes I1.
4. **Operations:** the outbox, the local exit file in the worker entrypoint,
   stop operations recorded at shutdown, exit classification, and crash
   history as a table. Fixes I4 and the restart gaps.
5. **The pool owns every machine,** in three parts:
   - **5a.** Migrate task-owned machines into the pool, with no UI change.
     This retags their instances (`Provider.retag`), ports stop, start, a
     moved host and spend into the pool, and turns bare-host slots and
     co-tenant hand tags into manual assignments. Gated by I2 and I3.
   - **5b.** The tag page's Machines card and rent form go, and Run by hand
     on… arrives.
   - **5c.** Delete the task-machine code paths: `_reconcile_machines`, the
     task-versus-pool split in orphans, the burn strip and Stop all cloud
     spending.
6. **The tag page shows the projected state,** and offers only valid verbs.
   The tooltip and confirmation copy can land separately.
7. **The tag's home, and moving it,** in two parts:
   - **7a.** `home`, and the no-regress rule for pulls (§5). The #290 refusal
     stays for now. This closes the failure the incident exposed.
   - **7b.** The automatic move home: the `moving` phase, the full-tree
     protocol, publishing through the scheduler, and `refusal` turned into
     "would move". Measure the unpublished-generations case first.
8. **The stale-code banner, and refusing to enqueue a finished tag.** These
   are independent and can go any time.

## Operator's calls (2026-09-29)

1. **The control store:** one SQLite database.
2. **Renting from a tag's page:** left to this plan, which removes it.
   Machines come only from the Machine pool page, and a tag page runs a tag by
   hand on a pool machine (§4).
3. **Verb names:** Requeue and Release stay, and both get a tooltip and a
   confirmation that say what happens to the machine (§6).
4. **Move home:** automatic, as a phase of placement (§5). Confirmed after
   weighing the measured cost (about 170 MB per tag) against a manual step.

5. **An operator-rented machine left idle:** stopped, disk kept (§4). The
   operator chose to rent it, and a later "Run by hand on…" would want it
   back. Rentals the queue made under a capacity entry are still terminated
   when idle.

## Review record (2026-09-29)

**The panel:**
- hidden complexity (session-tier subagent);
- rival design (Codex, `gpt-5.6-sol`, full repo access);
- scope (Sonnet subagent);
- integration (Sonnet subagent).

Each saw only the plan, the repo and its lens. Every blocking and serious
critique is below with its resolution. None is left open. Call 5 above is
new policy the review surfaced, not a disputed critique.

| Critique (panel) | Severity | Resolution |
|---|---|---|
| Move home skips `records/` and `models/`; the bucket's `models/` sync would delete local exports; stale bucket records overwrite newer ones (hidden complexity) | blocking | **Revised.** The move fences stale bucket outputs, uploads the whole trainer tree and verifies it before the flip (§5). I5 now covers exports and records, and the simulation gains a stale-bucket event (§7). |
| The no-regress pull rule is written as existing behavior; it does not exist (integration) | blocking | **Revised.** It is marked as new work, with its comparison, the missing-file and tie cases, and its targets specified (§5). It is PR 7a's acceptance criterion. |
| One lifecycle state hides independent facts; normalize into records with constraints plus an operation outbox, and project the states the operator sees (rival) | serious | **Revised, adopted in substance.** Records and constraints (§1), the outbox (§3), and the projection as the operator's state. One writer is kept, without correctness depending on it. |
| Foundations are built before the store and would be done twice; make the schema first, with a shadow import (rival) | serious | **Revised.** The landing now runs paths, then schema and shadow import, then the simulation, then the switch, then operations. The simulation reads the shadow projection, so it survives the switch (§Landing). |
| Move home needs placement split into lease and start, with a `moving` phase that is polled (integration) | serious | **Revised.** The `moving` assignment phase, polled by `_advance_lease`, with `_start_slots` still idempotent (§5). |
| The #290 refusal would make the automatic move dead code (integration) | serious | **Revised.** The refusal becomes "would move its training state (N MB)", and all eight call sites are checked in PR 7b (§5). |
| Generations uploaded by both the move and the publish hook (hidden complexity) | serious | **Revised.** The move publishes through the scheduler's path. The hook is enabled for `home = bucket` or a pending move. The published markers are the only record of uploads (§5). |
| Exclusive leases forbid today's shared hand-run use; bare-host slots not migrated (hidden complexity) | serious | **Revised.** Manual assignments share; queue assignments are exclusive; I2 is amended; `occupants` is replaced; bare hosts are migrated through `canonical_host` (§1, §4). |
| Migrated rentals keep task owner tags; pool rentals have no stop and start (hidden complexity) | serious | **Revised.** `Provider.retag`, and stop, start, moved host and spend ported into the pool, as PR 5a's explicit scope (§4). |
| Local exit codes don't survive a restart; shutdown stops workers with no intent (hidden complexity) | serious | **Revised.** An exit file from the worker entrypoint, the `lost` outcome, and stop operations recorded at shutdown (§3). |
| Non-exit finishes break I4; intents have no timing or expiry (hidden complexity) | serious | **Revised.** Outcome sources, and matching only exits after the operation's creation (`FinishedAt` or the exit file), with the operation closed by the next run (§1, §3, I4). |
| The paths context must come first (hidden complexity) | serious | **Revised.** It is PR 0 (§Landing). |
| The simulation cannot show freedom from races; the harness is new work (hidden complexity) | serious | **Revised.** The fake world is scoped explicitly. The claim is narrowed to orderings and restarts, with data races left to the structure plus the writer-thread check (§7). |
| I8 has no oracle (hidden complexity) | serious | **Revised.** Convergence of a named projection within K quiescent steps, events independent of timing, and crash history stored (§3, §7). |
| The migration must resolve store disagreements and has no rollback (hidden complexity) | serious | **Revised.** The decision table, a dry run on a copy of the live mount, the shadow period, and a database-to-JSON export (§10). |
| PR 5 bundles migration, UI and deletion (scope) | serious | **Revised.** Split into 5a, 5b and 5c. |
| Automatic move-home bundled with the fix it is justified by (scope) | minor | **Revised.** Split into 7a (home and no-regress) and 7b (the automatic move). The operator's call stands. |
| Store and new transitions in one PR (scope) | minor | **Revised.** The shadow import (PR 1) comes before the switch (PR 3). |
| The verb copy rides with the correctness PR (scope) | minor | **Revised.** It is marked independent (§6). |
| Is hand-run use heavy enough to keep manual leases in v1? (scope) | minor | **Kept.** Hand-running was the norm before the queue, and is how the live incidents were worked around. Sharing is what makes it cheap (§4). |
| What stays in `task.json`; outside readers (hidden complexity) | minor | **Revised.** The field split, the readers named, and the database row deciding existence (§9). |
| Snapshot rebuilds on the writer thread (hidden complexity) | minor | **Revised.** Snapshots cover control state only and are incremental; file-derived data stays in the read handlers (§2). |
| The integration panel confirmed the plan's other claims about the current code (in-memory fields, event-loop mutations, the cached-object sharing) | — | Noted. |

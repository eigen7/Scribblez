# Training workloads on the master dashboard

How the training pipelines run as first-class workloads of the master
dashboard ([master_dashboard.md](master_dashboard.md)): self-play generation
farmed out to any number of interchangeable local/cloud workers, the GPU
trainer running as a distinguished singleton worker consuming the shared data,
and the whole run driven from the one web shell. Both training workloads
(position_eval and max_move_per_lane) share this shape; position_eval is
described throughout, with the probe differing in its parameters and tabs and
lacking position_eval's third role (the match_eval singleton below).

## The workload-spec contract

Declared in `scribblez/workloads/base.py`; one module per workload in
`scribblez/workloads/`, registry in `__init__.py`. A `WorkloadSpec` names the
params dataclass, the roles (`RoleSpec`: runner and deps as dotted paths,
singleton or parallel, allowed kinds, interruptible, stats schema), an
optional controller-side scheduler, a progress callable, and the bucket
prefixes cloud_sync pulls. Registry modules stay import-light — heavy code is
referenced by dotted path and imported only when it runs, so cloud CPU pods
import the registry without torch.

Knobs split uniformly across workloads: **task params** (frozen at creation;
define the corpus and the model), **live controls** (per-tag dashboard.db
control table, adopted by the trainer at its natural cadence), and
**worker-slot resources** (threads/vcpus, per slot, never frozen).

The scheduler tick runs inside the dashboard server's reconcile loop (the one
always-on controller-host process) and gets `SchedulerHooks`:
`gate(role, reason)` to park a role's workers distinctly from operator pause,
and `mirror(chunk, dest)` to replay ingests in the bucket. The same loop runs
a role's `dispatch` tick, if it declares one: the controller-side half of a
role whose work it assigns rather than the worker choosing it (match_eval,
below), which gets one handle per running slot onto that slot's filesystem
(`scribblez/dashboard/slot_files.py`). Client-side, a
parallel registry (`web/src/workloads.tsx`) maps workload name → extra tab
components; the server never renders tabs, only generic data endpoints.

kill_test under the contract: one parallel, interruptible `generate` role and
no scheduler; each worker cycles in a private work dir and delivers complete
pairs to the tag's data store.

## Roles

| Role | Cardinality | Kinds | Interruptible | Does |
|---|---|---|---|---|
| `generate` | N, interchangeable | local + cloud | yes | one cycle = one whole `.slog` chunk of self-play games, delivered to the staging area |
| `train` | singleton | local (the GPU box) | — | consume complete generations: train, checkpoint, export ONNX, write dashboard.db |
| `match_eval` | singleton | local + ssh (needs a GPU) | — | play sequential-test-checked paired matches for the checkpoint the controller assigns it, against a fixed opponent (position_eval only; docs/roadmap.md A1) |

The trainer never generates and the generators never train; match_eval only
consumes exported ONNX checkpoints, so the training loop is never blocked. A
single-machine run attaches one local generator and the trainer (plus,
optionally, the match_eval worker) to the same task; the CPU split between
them is the slots' thread counts. Putting the match_eval slot on a second
machine (kind `ssh`) is how the eval matches stop competing with training for
the GPU — see the match-eval roundtrip below.

## Data flow: staging + controller-side ingest

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

Generators are **generation-agnostic** (`scribblez/workloads/selfplay_gen.py`,
shared by both training workloads): one cycle produces one whole `.slog` chunk
in the worker's private work dir and hands it to the results sink, which lands
it in the tag's `staging/` (locally by rename; from the cloud via the bucket's
staging prefix and the sync watcher). The work dir is wiped on worker start
and chunks are written in one shot, so a crash loses at most the in-flight
chunk and leftovers are never delivered.

The **scheduler** (`scribblez/generational/scheduler.py`) is the *only*
writer of generation structure: each tick it moves staged chunks into the open
generation directory and flips the manifest to `complete` at the target game
count. Single-writer, whole-file renames make the invariants trivial:

- every `.slog` belongs to exactly one generation and arrives whole;
- completion is a recorded fact (`manifest.json`), and committed counts are
  recomputed from `.slog` headers each tick, so crashes self-heal;
- an ingest ledger written before each rename makes assignment idempotent — a
  chunk re-appearing in staging (a cloud sync racing an ingest) is deleted,
  not assigned twice.

For cloud chunks the scheduler mirrors the assignment in the bucket
(server-side move via the `mirror` hook), so the bucket remains the durable
archive with the same layout as the local corpus — disaster recovery is an
`rclone copy` of the tag prefix — and the sync watcher never re-downloads an
ingested chunk.

## Generation lifecycle and pacing

At most one generation is open at a time. The scheduler opens generation `M`
when `M ≤ trainer_cursor + open_ahead` (task param), closes it at
`target_games`, and when nothing is open **gates** the generate role — local
processes parked, pods stopped, shown as `waiting (ahead of trainer)` —
ungating when the trainer advances. `trainer_cursor` comes from a small
`train_state.json` the trainer writes atomically at every checkpoint; nobody
outside the trainer parses the torch checkpoint. Before a trainer has ever
run the cursor reads 0, so generation runs ahead of a trainer being attached
at all. Overshoot from in-flight chunks is bounded and harmless — they join
an open generation like any other games.

## The trainer role

`scribblez/position_eval/trainer.py` (and its max_move_per_lane sibling) owns
orchestration: resume from the rolling checkpoint; wait for the cursor's
generation to complete (sleep-poll on manifests — GPU idle here *is* the
"generation is the bottleneck" signal, visible in Stats); train one epoch
over the window; checkpoint + ONNX export + metrics + `train_state.json`
under the generation's index; evict generations beyond the window; advance
the cursor.

Launched as a local worker slot like any other; the runner lives with the
training code (referenced by dotted path) so generator bundles never import
torch. `scripts/position_eval/train.py` remains a thin CLI over the same
runner for headless debugging. SIGTERM pauses; resume repeats at most one
generation from the last checkpoint. Live controls (LR, loader threads) come from
dashboard.db. A fresh start is a fresh tag; there is no in-place run reset.

## The match_eval roundtrip

The match-eval worker does not choose its own work: the controller assigns it
a generation and ingests what comes back
(`scribblez/match_eval/dispatch.py`, ticked per task by the reconcile loop).

```
controller picks the newest export with no match row
        │
        ▼  put in the slot's inbox (symlink locally, a push over ssh)
data/match_inbox/<worker_id>/model_epoch_NNNN.onnx
        │
        ▼  worker plays it: SPRT-checked paired rounds, then marks it .done
data/match_results/gen_NNNNNN-<worker_id>.json
        │
        ▼  controller ingest: a match_eval row + match_* metrics, keyed by generation
           (and the .done mark is cleared, freeing the slot)
```

That split is what lets the slot sit on another machine. The database and the
exports both live on the controller, neither reachable from an ssh worker's
container; what crosses the link is one model in and one small JSON out, over
the control connection the dashboard already holds open
(`py/cloud/ssh_transfer.py`). A local slot takes the identical path — its
"link" being a symlink into `models/` — so there is one runner and one set of
rules rather than one per kind.

**The inbox is the ledger**, and it holds a generation until that generation is
recorded — not until the worker is done with it. Nothing durable records what
is in flight, because the directory already says. The distinction matters
because a container's result reaches the controller by collection, a separate
step that can fail or time out for passes on end: a ledger that emptied when
the worker finished would offer the same generation up again in that gap and
replay a match the machine had already played. Ingest is idempotent anyway (a
row is keyed by its generation), and a file that is not a result is
quarantined as `.bad` rather than retried forever.

Shared external-data blobs beside the exports (the frozen-lexicon blob, when
the model has one) go with the first assignment and stay: a model does not
load without them, and every generation references the same bytes.

## Seeds

Generators always run `play_game` with seed 0 (the binary draws from
`std::random_device` per chunk): any deterministic seed partition across a
fleet would duplicate games, so distributed corpus reproducibility is
deliberately not offered. If a reproducible single-machine corpus is ever
needed, `generate_data.py` still exists.

## Failure and restart matrix

| Failure | Effect | Recovery |
|---|---|---|
| generator crash / pod preemption | loses at most the in-flight chunk | reconcile respawns/restarts it |
| dashboard server down | no ingest, no gating; local workers die; pods keep producing into bucket staging | on restart: reconcile respawns locals, ingest drains staging |
| trainer crash | training halts; generation continues to the ahead-limit gate | respawn resumes from the rolling checkpoint |
| sync lag | chunks arrive late to staging | ingest is idempotent; late chunks join the open generation |
| corrupt staged chunk | quarantined as `.bad`, never assigned | — |
| match_eval worker or container dies mid-match | the model is still in its inbox unmarked, so the match counts as unplayed | it replays from the same fixed seeds on the next start |
| a push is cut off mid-model | the size check fails, so nothing lands under the name the worker polls for | the next pass re-pushes |
| match_eval machine unreachable | no matches; training is unaffected | reconcile resumes assigning when it answers again |

## Dashboard UI

Overview gains a `role` column, `waiting (<reason>)` states, one add-worker
form per role, and the workload's progress counters (fill state, completed
generations, trainer cursor, rows). Stats is the generic schema-driven tab.
The training tabs (Loss / Positions / Match / Training / Controls / Info;
max_move_per_lane registers Loss / Lane analysis / Controls / Info) are
registered in `web/src/workloads.tsx`; `scripts/dashboard.py` is the single
entrypoint and always renders the master app.

## Open questions

- **Chunk size** — the pacing/latency quantum; fixed at 1000 games
  (play_game's kGamesPerFile), measure under real parameters.
- **Trainer wait behavior** — sleep-poll on manifests is fine until the idle
  tail between generations is measured to matter.
- **GPU trainer in the cloud** — the role abstraction admits it later;
  nothing depends on it.
- **Neural self-play generations** — once generation needs the current model,
  chunks must be stamped with the model version that produced them, the
  scheduler routes/rejects by version, and model distribution piggybacks on
  the gate/ungate cycle. Staging-plus-ingest is the right substrate; the
  stamping format is deferred to the neural phase.

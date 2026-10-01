# Training workloads on the master dashboard

How the generational training pipelines run as workloads of the master
dashboard ([master_dashboard.md](master_dashboard.md)). Self-play generation
is farmed out to any number of interchangeable workers, the GPU trainer runs
as a singleton worker consuming their shared output, and the whole run is
driven from the web UI.

position_eval is described throughout. max_move_per_lane has the same shape
with its own parameters and tabs, a local-only trainer, and no match_eval
role. For why training is organized into generations at all, see
[generational_training.md](generational_training.md).

## The workload-spec contract

The contract is declared in `py/scribblez/workloads/base.py`, one module per
workload in the same package, with the registry in `__init__.py`. A
`WorkloadSpec` names:

- the params dataclass;
- the roles (`RoleSpec`): runner and deps as dotted paths, singleton or
  parallel, allowed worker kinds, whether it needs a GPU, which worker image
  (`runtime`) it runs on, optional controller-side `dispatch` and `ingest`
  ticks, out-of-tag `inputs`, and a stats schema;
- an optional controller-side scheduler;
- a progress callable;
- the `data/` subdirectories workers deliver into (`collected_dirs`), which
  is what a collection from an ssh container takes.

Registry modules stay import-light. Heavy code is referenced by dotted path
and imported only when it runs, so a CPU-only worker container can import the
registry without torch.

Knobs split the same way across workloads:

| Kind | Where it lives | When it changes |
|---|---|---|
| **Task params** | `task.json`, frozen at creation | never; they define the corpus and the model |
| **Live controls** | the tag's `controls.json`, written by the Controls tab | adopted by the trainer at its natural cadence |
| **Slot resources** | the worker slot (threads) | per slot, any time |

The scheduler tick runs inside the dashboard server's reconcile loop, the one
always-on controller process. It receives `SchedulerHooks`:

- `gate(role, reason)` parks a role's workers, distinct from an operator
  pause;
- `mirror(chunk, dest)` replays a local ingest in the bucket;
- `publish(dest)` uploads a completed generation for a trainer running
  elsewhere.

The same loop runs a role's `dispatch` tick, if it declares one. That is the
controller-side half of a role whose work the controller assigns rather than
the worker choosing it (match_eval, below); it gets one handle per running
slot onto that slot's filesystem (`scribblez/dashboard/slot_files.py`).

On the client, a parallel registry (`web/src/workloads.tsx`) maps a workload
name to its extra tab components. The server never renders tabs; it serves
only generic data endpoints.

## Roles

| Role | Cardinality | Kinds | Does |
|---|---|---|---|
| `generate` | N, interchangeable | local, ssh | one cycle = one whole `.slog` chunk of self-play games, delivered to the staging area |
| `train` | singleton | local, ssh (a rented GPU machine) | consume complete generations: train, checkpoint, export ONNX, deliver the generation's record |
| `match_eval` | singleton | local, ssh (needs a GPU) | play a match against a fixed opponent for each exported checkpoint |

The trainer never generates and the generators never train. match_eval only
consumes exported ONNX models, so it never blocks the training loop.

A single-machine run attaches one local generator and the trainer (and
optionally the match_eval worker) to the same task, and splits the CPU
between them through the slots' thread counts. Putting the match_eval slot
on a second machine is how the eval matches stop competing with training for
the GPU (see [The match_eval roundtrip](#the-match_eval-roundtrip)).

## Data flow: staging and controller-side ingest

```
generator (local)  ──chunk──►  tags/position_eval/<tag>/data/staging/     ─┐
generator (remote) ──chunk──►  bucket: position_eval/<tag>/staging/ ─sync─► ─┤
                                                                            │ scheduler ingest
                                                                            ▼ (single writer)
                                              data/test/*.slog          (filled first, then frozen)
                                              data/generations/gen_000000/{manifest.json, *.slog}
                                              data/generations/gen_000001/...
                                                                            │
                                                              trainer: SlogDataset(window dirs)
```

Generators are **generation-agnostic** (`scribblez/workloads/selfplay_gen.py`,
shared by both workloads). One cycle writes one whole `.slog` chunk in the
worker's private work dir and hands it to the results sink, which lands it in
the tag's `staging/`: by rename locally, or into the container's own tree for
a remote worker, from where the controller collects it over ssh. The work dir
is wiped on worker start and chunks are written in one shot, so a crash loses
at most the in-flight chunk and leftovers are never delivered.

The **scheduler** (`scribblez/generational/scheduler.py`) is the only writer
of generation structure. Each tick it moves staged chunks into the open
generation directory and marks the manifest `complete` once the generation
holds `games_per_generation` games. A single writer doing whole-file renames
makes the invariants easy to hold:

- every `.slog` belongs to exactly one generation and arrives whole;
- completion is a recorded fact (`manifest.json`), and committed counts are
  recomputed from `.slog` headers each tick, so crashes self-heal;
- an ingest ledger written before each rename makes assignment idempotent: a
  chunk that reappears in staging (a collection or relay repeated after its
  delete failed) is deleted, not assigned twice;
- a chunk whose header cannot be read is quarantined as `.bad`.

Where the scheduler runs is the tag's *data plane* (`TaskRecord.data_plane`).
On the legacy plane the controller ticks it over its own tag tree. On a
*data home* it runs beside the trainer, on the trainer's machine
(`scribblez/generational/data_home.py`), and the controller only gates the
generators from the home's published gate and heartbeat. Generators on that
machine deliver into its staging by rename; for a home on an ssh machine, the
controller relays in the chunks it collects from generators elsewhere.

On the legacy plane, when the tag has an ssh trainer, the scheduler also
*publishes* each completed generation to the bucket (the `publish` hook):
first the chunks, then the manifest, so a manifest in the bucket means the
whole generation is there. That is what the trainer reads
([plans/cloud_training.md](plans/cloud_training.md)). The manifest records
publication, so a failed upload is retried on the next tick.

## Generation lifecycle and pacing

At most one generation is open at a time. The scheduler opens generation `M`
only when `M ≤ trainer_cursor + open_ahead` (a task param) and closes it at
`games_per_generation`. When nothing may be opened, it **gates** the generate
role (local workers stopped, containers paused, both shown as
`waiting (ahead of trainer)`) and releases the gate when the trainer
advances.

`trainer_cursor` comes from a small `train_state.json` the trainer writes
atomically at every checkpoint; nothing outside the trainer parses the torch
checkpoint. Before any trainer has run, the cursor reads 0, so generation can
start before a trainer is attached. Chunks still in flight when a generation
closes simply join the next open generation.

## The trainer role

`scribblez/position_eval/trainer.py` (and its max_move_per_lane sibling) owns
the loop:

1. resume from the rolling checkpoint;
2. wait for the cursor's generation to complete, sleep-polling its manifest
   (GPU idle time here is the "generation is the bottleneck" signal, visible
   in Stats);
3. train one epoch over the window;
4. write the checkpoint, the ONNX export, `train_state.json` and the
   generation's record, all under the generation's index;
5. evict generations that have left the window, and advance the cursor.

**The trainer never writes `dashboard.db`.** Its metrics, eval predictions and
control events leave as generation-keyed records through the worker's results
sink (`records/` under the tag; `scribblez/generational/records.py`). The
dashboard's reconcile loop ingests them into the database (`RoleSpec.ingest`,
`scribblez/generational/train_ingest.py`). The database therefore has a single
writer, and the trainer is the same process wherever it runs. A generation's
record is written last, after the ONNX export and the checkpoint, so its
existence is the commit: the dashboard lists a generation only once
everything it stands for is on disk.

Live controls travel the other way as one file. The Controls tab writes every
control's value to the tag's `controls.json`, which the trainer reads through
the same sink once per generation.

**Everything crosses the sink**, which is what lets the same trainer run on
the controller's machine or on a rented one:

- Before waiting on a generation's manifest, the trainer fetches the
  generation through the sink (a pull of the published generation from the
  bucket; nothing under the local sink, whose mount dir already holds it).
- After writing each generation's outputs (the ONNX export, the rolling
  checkpoint and cursor as one state pair, then the record), it delivers them
  through the sink.
- Every start restores the newest state pair the sink holds when it has
  trained more rows than this machine's copy (the cursor rule,
  `generational/state_pair.py`), then the window's generations. A fresh
  machine therefore takes the sink's state; one holding fresher state keeps
  its own.

A trainer's records sink is local wherever it runs, so all of this writes
into its own tag tree. For an ssh trainer (an ssh slot on a rented or owned
GPU machine, on the torch worker image) the controller supplies the other
half over the control link:
- each pass collects its `models/`, `records/` and newest complete state pair,
  installing the pair under the cursor rule;
- a new container is seeded with the controller's checkpoint and cursor
  (`SCZ_STATE_SEED`), which the trainer installs under the rule before
  resuming;
- `controls.json` is pushed into its container whenever the Controls tab
  rewrites it.

On a data home it assembles its own generations; on the legacy plane they
still come through the bucket, assembled and published by the controller.
Match eval runs locally or over ssh against the collected exports, and the
tabs read what collection brought home. Several tags with remote trainers can run side by side from one
dashboard.

The runner lives with the training code and is referenced by dotted path, so
generator bundles never import torch. `py/scripts/position_eval/train.py` is
a thin CLI over the same runner for headless debugging; with no server
running to ingest records, `py/scripts/ingest_train_records.py` does it by
hand. SIGTERM stops the trainer, and a resume repeats at most one generation
from the last checkpoint. There is no in-place run reset: a fresh start is a
fresh tag.

## The match_eval roundtrip

The match-eval worker does not choose its own work. The controller assigns it
a generation and ingests what comes back (`scribblez/match_eval/dispatch.py`,
ticked per task by the reconcile loop).

```
controller picks the newest export with no match row
        │
        ▼  put in the slot's inbox (a symlink locally, a push over ssh)
data/match_inbox/<worker_id>/model_epoch_NNNN.onnx
        │
        ▼  worker plays match_pairs paired games, then marks the model .done
data/match_results/gen_NNNNNN-<worker_id>.json
        │
        ▼  controller ingest: a match_eval row + match_* metrics, keyed by generation
```

This split is what lets the slot sit on another machine. The database and the
exports both live on the controller, out of reach of an ssh worker's
container. What crosses the link is one model in and one small JSON out, over
the control connection the dashboard already holds open
(`py/cloud/ssh_transfer.py`). A local slot takes the identical path, its
"link" being a symlink into `models/`, so there is one runner and one set of
rules for both kinds.

**The inbox is the ledger.** It holds a generation until the controller has
accounted for it, not merely until the worker is done with it; nothing else
records what is in flight. The distinction matters because a container's
result reaches the controller by collection, a separate step that can fail or
time out for many passes in a row. A ledger that emptied when the worker
finished would offer the same generation again during that gap and replay a
match already played. Ingest is idempotent anyway, since a row is keyed by
its generation.

"Accounted for" means recorded, or delivered and found unreadable. A result
file that does not parse is quarantined as `.bad`, and that settles its
generation: nothing more is coming for it, so the mark stops holding the slot,
the generation falls due again, and the match is replayed. Any terminal
outcome that did not release the slot would strand it, and the readout would
silently stop.

The `.done` mark is not cleared at ingest but the next time that slot is
offered work, the only moment its presence matters. A mark can therefore
outlive its row while nothing new is due, which is harmless.

Shared external-data files beside the exports (the frozen-lexicon blob, when
the model has one) go with the first assignment and stay: a model does not
load without them, and every generation references the same bytes.

## Seeds

Generators always run `play_game` with seed 0, which makes the binary draw
from `std::random_device` per chunk. Any deterministic seed partition across
a fleet would duplicate games, so distributed corpora are deliberately not
reproducible. A reproducible single-machine corpus would need an explicit
seed: `run_games` in `py/scribblez/selfplay.py` accepts one, but no script
currently exposes it.

## Failure and restart matrix

| Failure | Effect | Recovery |
|---|---|---|
| generator crash or machine loss | loses at most the in-flight chunk | reconcile respawns or restarts it |
| dashboard server down | no ingest, no gating; local workers die; remote containers keep producing into their own trees | on restart, reconcile respawns local workers and collection drains the containers |
| trainer crash | training halts; generation continues up to the ahead-limit gate | respawn resumes from the rolling checkpoint |
| collection lag | chunks reach staging late | ingest is idempotent; late chunks join the open generation |
| corrupt staged chunk | quarantined as `.bad`, never assigned | none needed |
| match_eval worker or container dies mid-match | the model is still in its inbox unmarked, so the match counts as unplayed | it replays from the same fixed seeds on the next start |
| a push is cut off mid-model | the size check fails, so nothing lands under the name the worker polls for | the next pass re-pushes |
| match_eval machine unreachable | no matches; training is unaffected | reconcile resumes assigning when it answers again |

## Open questions

- **Chunk size.** The pacing and latency quantum, fixed at 1000 games
  (`kGamesPerFile` in `engine/src/arena/game_runner.cpp`). Not yet measured
  under real parameters.
- **Trainer wait behavior.** Sleep-polling manifests is fine until the idle
  tail between generations is measured to matter.
- **Neural self-play generations.** Once generation needs the current model,
  chunks must be stamped with the model version that produced them, the
  scheduler must route or reject by version, and model distribution can ride
  the gate/ungate cycle. Staging plus ingest is the right substrate; the
  stamping format is deferred until then.

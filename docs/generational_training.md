# Generational training

The training loop shared by the trainers of the position_eval and
max_move_per_lane workloads
([scribblez/position_eval/trainer.py](../py/scribblez/position_eval/trainer.py),
[scribblez/max_move_per_lane/trainer.py](../py/scribblez/max_move_per_lane/trainer.py)),
and the machinery it is designed to grow into. This document covers the
rationale; the protocol (roles, staging, scheduler, failure handling) is in
[position_eval_workload.md](position_eval_workload.md), and the data pipeline
underneath is in [architecture.md](architecture.md).

Built: the rows-clock, generations with a sliding window, one epoch per
generation, restart reconciliation, live controls, and distributed generation
through the master dashboard. Not built: the game-pool producer and the
resource-contention manager.

## Motivation

Two simpler designs sit at opposite extremes:

- **Streaming.** C++ self-play feeds rows straight into the GPU loop through
  an in-process ring buffer. Each game yields one position, used once. This
  wastes most of every generated game and starves the GPU on generation.
- **One-shot disk.** Generate everything, then run epochs over it. Full reuse,
  but a rigid lifecycle with no stop-and-resume.

The lever that matters is gradient signal per generated position: sample more
turns per game, and reuse each position across passes. It matters even more
once self-play moves from HastyBot to a much more expensive neural agent.
Generational training keeps the disk design's reuse, adds stop-and-resume,
and extends to neural self-play and remote generation workers.

## Core concepts

- **The rows-clock.** Everything that must survive a restart or index the
  dashboard is keyed on cumulative rows trained, never on wall-clock time. It
  is the dashboard's x-axis and the restart cursor stored in the rolling
  checkpoint.
- **Generations and the sliding window.** A generation is a batch of
  self-play games in its own directory
  (`data/generations/gen_NNNNNN/{manifest.json, *.slog}`). The trainer trains
  over a sliding window of the most recent `W` generations (`SlogDataset`
  takes a list of directories).
- **Decorrelation and multi-sampling come from the data pipeline.**
  `SlogDataset.iter_batches` shuffles the whole loaded set each pass. With
  `turns_per_game K` and the generation index passed as `epoch_index`, it
  draws a fresh set of K turns from each game on every pass. Over a game's
  `W` passes through the window that yields up to `W·K` distinct positions,
  rather than the same K rows repeated `W` times.
- **Epoch and generation are the same clock.** Each generation is trained
  exactly once: one epoch over the window it completes. A game's lifetime
  reuse is therefore fixed at `window · turns_per_game` passes by
  construction. This bound is load-bearing. At ~40 passes per game, the
  positions of a game all share one WLD target, and the model learned to
  memorize game outcomes: training accuracy kept climbing while held-out
  quality and play strength decayed from about 1M rows on. At 4 passes per
  game, the same row budget keeps improving.

## The lifecycle

The trainer is a pure consumer. Generator workers stage whole-file chunks,
the generation scheduler assigns them to generation directories, and the
trainer repeatedly waits for the cursor's generation to complete, trains one
epoch over the window, checkpoints, exports and publishes under the
generation's index, evicts generations beyond the window, and advances.
Generation overlaps training through the scheduler's ahead-limit: the fleet
runs continuously up to `open_ahead` generations ahead of the trainer's
published cursor, then is gated.

**Restart is "run it again."** What has been done is recorded by the
per-generation manifests plus the rolling checkpoint (`rows_trained`,
`generation_index`). On startup the trainer reconciles disk against that
state: it either waits for a filling generation or advances. Chunks are whole
files assigned by a single writer, so counting committed games against
targets is reliable.

**Learning rate.** An open-ended, stop-and-resume run has no known horizon,
so a single end-of-run decay has no trigger point. The trainers use one of
two horizon-free optimizer arms, selected by the frozen `optimizer` task
param (`scribblez/generational/optim.py`):

- `schedule_free`: AdamWScheduleFree, which replaces the schedule with an
  averaged iterate, so every generation's export is equally deployable. The
  default for position_eval and move_set_eval.
- `wsd`: AdamW on a cyclic warmup-stable-decay schedule over the rows-clock,
  which leaves a well-annealed checkpoint at the end of every cycle. The only
  schedule for max_move_per_lane and evidence_trajectories.

Both are pure functions of `rows_trained` plus the checkpointed optimizer
state, so a resume continues exactly where it stopped. The WSD design and its
sizing are in [wsd_lr_schedule.md](wsd_lr_schedule.md).

**Live controls** (DataLoader workers, torch threads) share one mechanism: a
per-tag controls file the dashboard writes and the trainer reads once per
generation, with no IPC into the hot loop. Values persist in the dashboard's
database and are restored on restart. The trainer's own output travels the
opposite way: it delivers records and the dashboard ingests them (see "The
trainer role" in [position_eval_workload.md](position_eval_workload.md)).

## Why a window, not a wipe

Wipe-and-regenerate is the degenerate `W = 1` case. For neural self-play a
sliding window is the right default: data from an older, weaker model ages
out gradually, without a hard distribution reset. For stationary HastyBot
data either works, so a modest window is a safe universal default.

A related choice, discrete generations with a fixed number of passes versus a
continuously advancing row window (fresher, and better suited to neural
data), sits on the same rows-clock and window abstraction. It is a parameter,
not a rewrite.

## Future: the game-pool producer (C++)

Today each self-play worker thread owns an agent pair and plays one game from
start to finish ([game_engine.h](../engine/include/arena/game_engine.h)), with
the thread count fixed at construction. The proposed pool decouples in-flight
games from threads: G active game slots with G ≫ T workers, each worker
advancing one game by one unit of work at a time. That buys two things:

1. **Live thread tuning.** Workers park and wake between units of work, with
   no game abandoned and no pool rebuild. This is the actuator a resource
   controller needs.
2. **Batched GPU inference.** A unit of work becomes "advance until the game
   needs a network evaluation, then yield", so many in-flight games gather
   into one batched forward pass. One game per thread cannot express this,
   and neural self-play depends on it.

## Future: relaunch per chunk vs. run forever

Generators relaunch the `play_game` subprocess once per chunk, so
thread-count changes apply at chunk boundaries. A run-forever producer
(long-lived, retuned live, weights refreshed in place, games straddling model
versions) is the AlphaZeroArcade design. Its motivations are specifically
neural: long games, expensive model loads, GPU tail latency. None apply to
HastyBot, whose games take milliseconds and load no model, so relaunching
costs nothing.

Beyond the game pool, a run-forever producer needs:

- flow control: produce about one generation ahead, then park;
- **drain-and-flush at generation boundaries**: a `.slog` file must never
  straddle two generation directories, because the loader trusts file
  headers and the lifecycle counts committed games per directory (relaunching
  gets this for free, since process exit flushes);
- crash supervision;
- a control path. The simplest local form runs the producer in-process via
  the FFI, so control is direct calls, with a subprocess and socket only for
  isolation or remote workers.

The neural-specific interfaces (the evaluation-batching hook, the
weight-refresh API) are defined by the neural agent that will use them and
cannot be designed well before it exists. This design is recorded here to be
implemented at step 3 of the table below, once that agent makes the
interfaces concrete.

## Future: resource contention

Three consumers compete for CPU: game generation, the C++ DataLoader, and
PyTorch's own threads. In the neural regime, self-play also competes with
training for the GPU. The abstraction is a contention manager over resources,
domains and priorities, modeled on AlphaZeroArcade's `GpuContentionTable`:

- divisible resources (CPU cores), split by a feedback controller that reads
  producer/consumer blocked-time counters, with GPU utilization as the target;
- exclusive locks (the GPU), awarded TRAINING before SELF_PLAY and yielded
  cooperatively.

The orchestrator never special-cases neural vs. HastyBot generation; HastyBot
generation simply never requests the GPU lock.

## Build steps

| Step | Build | Status | GPU contention |
|------|-------|--------|----------------|
| 1 | Discrete-generation lifecycle on the rows-clock; per-directory manifests; shared `run_epoch`; rows-clock LR | built | none (HastyBot) |
| 2 | Distributed generation: generator fleet and generation scheduler (staging ingest, pacing gate) on the master dashboard | built | none (HastyBot) |
| 3 | Game-pool producer (G ≫ T, live thread target); contention manager; continuous sliding row-window as a parameter | future | abstraction only |
| 4 | Neural self-play: batched evaluation on the game pool; GPU priority lock; model-stamped chunks and model distribution to generators | future | yes |

Keeping "who fills a generation" behind the staging and manifest interface is
what makes steps 3 and 4 additive rather than rewrites.

## Deliberately out of scope

AlphaZeroArcade machinery not needed at this scale, and not to be copied
preemptively: the multi-database schema, the ratings and self-eval domains,
fork-run retrain windows, and the two-filesystem cloud syncer. The ideas
adopted are narrow: the rows-clock, the sliding window, manifest-based commit
tracking, and the game pool.

## Open questions

- **Generation size.** Large enough that shuffling within a generation
  decorrelates batches, small enough that refill keeps pace. Not yet measured.
- **Where the bottleneck lands.** If reuse makes the GPU the constraint, the
  CPU controller is moot. Blocked-time instrumentation is worth adding early,
  because it tells which regime we are in.
- **Reuse.** Per-game passes are `window · turns_per_game`. Whether the right
  setting differs for expensive neural data, where each game costs far more to
  generate, is unknown.

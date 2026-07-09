# Generational training

This is the training pipeline — implemented as the train roles of the
position_eval and max_move_per_lane workloads
([scribblez/position_eval/trainer.py](../py/scribblez/position_eval/trainer.py),
[scribblez/max_move_per_lane/trainer.py](../py/scribblez/max_move_per_lane/trainer.py))
— plus the C++ and coordination machinery it grows into. The core lifecycle
(rows-clock, generations, the sliding window, reuse-driven epochs, restart
reconciliation, the live dashboard controls, and distributed game generation
via the master dashboard's generator workers and generation scheduler — see
[docs/position_eval_workload.md](position_eval_workload.md)) is built; the
game-pool producer and the resource-contention manager described below are
forward-looking. For the data pipeline it builds on, see
[docs/architecture.md](architecture.md).

## Motivation

Two simpler pipeline shapes sit at opposite extremes of the design space:

- **Streaming**: C++ self-play feeds sampled rows straight into the GPU loop
  through an in-process ring buffer (`StreamingGameProducer` /
  `StreamingRowBuffer` — the substrate the neural game pool will reuse; see
  [The game-pool producer](#the-game-pool-producer-c)). One position per game,
  no shuffle, no disk; each position is used once and dropped. Attractive when
  generation is cheap and abundant (HastyBot self-play), but it wastes the
  expensive part of every game — a game is played to completion to yield a
  single training row — and is heavily CPU-bound: the GPU starves waiting for
  game generation.

- **One-shot disk** ([generate_data.py](../py/scripts/generate_data.py) + a
  fixed-epoch trainer): generate all `.slog` data up front, then epoch over it.
  Every position is reused across epochs and many turns are sampled per game,
  but the lifecycle is rigid — generate everything, then train everything —
  with no stop-and-resume ergonomics.

The lever that matters is extracting more gradient signal from each generated
position — sample more turns per game, and reuse each position across several
passes — which the disk shape allows and the streaming shape cannot. It matters
even more once self-play graduates from HastyBot to a neural agent: generating
a game becomes far more expensive, so squeezing maximum training value out of
each one stops being optional.

**Generational training** combines the disk shape's data reuse with
stop-and-resume ergonomics, and its structure extends cleanly to neural
self-play and, eventually, remote game-generation workers.

## Core concepts

### The rows-clock

Every quantity that must survive a restart or index the dashboard is keyed on
**cumulative rows trained** (equivalently, positions), not on an epoch or
generation index. This single monotonic counter is:

- the dashboard x-axis (comparable across restarts and generation boundaries),
- the input to the learning-rate schedule (see below), and
- the restart cursor (stored in the rolling checkpoint).

This mirrors AlphaZeroArcade, which keys its entire training loop on cumulative
committed rows rather than wall-clock or epoch count.

### Generations and the sliding window

A **generation** is a batch of self-play games written to its own directory:

```
tags/<task>/<tag>/data/
  staging/                   # generator chunks awaiting assignment
  test/                      # frozen ONCE at run start, never regenerated
  generations/
    gen_000042/
      manifest.json          # {index, target_games, committed_games, status}
      *.slog
    gen_000043/ ...
```

The trainer trains over a **sliding window of the most recent W generations**.
`SlogDataset` already accepts a list of directories and reads their union
([dataset.py](../py/scribblez/dataset.py)), so the window is
`SlogDataset([gen_dirs...])`. W = 1 is the simplest "wipe and replace each
generation" behavior; W > 1 keeps a rolling corpus and is the preferred default
(see [Why a window, not a wipe](#why-a-window-not-a-wipe)).

The **held-out test/eval set is generated exactly once** and never wiped. Only
the train generations churn. This keeps the probe and calibration metrics
comparable across the entire run — regenerating the eval set each generation
would make every metric curve non-comparable and risks train/test leakage.

### Decorrelation and overfitting come from the data pipeline

Two mechanisms the pipeline needs are supplied by the data pipeline:

- **Decorrelation within a generation** is free: `SlogDataset.iter_batches`
  shuffles the whole loaded set each epoch via the C++ loader's `epoch_start`.
  As long as a generation holds many games, batches are decorrelated without a
  separate shuffle pass.

- **Multi-sample-per-game with anti-overfit** is `--turns-per-game K` with a
  per-epoch `epoch_index`: it draws a *fresh* window of K turns each epoch, so E
  epochs over one generation yield up to `E*K` *distinct* positions per game —
  not the same K rows hammered E times.

So what this pipeline itself owns is the **lifecycle** (generate → train a few
passes → slide the window → repeat, restartable at any point) and the **producer
and resource-management** machinery below; the sampling and shuffling come free.

### The overfitting knob is reuse, not epochs

The quantity to bound is **gradient passes per unique position**, not epoch
count. A generation of G unique positions trained for E epochs at K turns/game
gives `E * min(K*games, G)` gradient samples over G uniques. Expose a target
**reuse factor** and derive the epochs-per-generation from the generation's
size, rather than exposing raw epochs — otherwise a small generation overfits at
the same E where a large one is fine.

## The game-pool producer (C++)

Today each self-play worker thread **owns an agent pair and plays one whole game
start to finish** ([self_play_engine.h](../engine/include/selfplay/self_play_engine.h),
[streaming_game_producer.h](../engine/include/selfplay/streaming_game_producer.h)),
with `next_game_` handing out game indices. The thread count T is fixed at
construction because agent pairs are preallocated per thread; changing T means
tearing down and rebuilding the pool.

The proposed model decouples **in-flight games** from **worker threads**: a
circular buffer of G active game slots (G ≫ T), and T workers that each advance
*one game by one unit of work*, then move to the next slot in round-robin order.
A worker parks on a condition variable when its index exceeds the live target
thread count.

```
class GamePool {
  struct Slot { Game game; AgentState agents; enum {RUNNABLE, NEEDS_EVAL, DONE} phase; };
  std::array<Slot, G> slots_;          // G in-flight games, G >> T
  std::atomic<int> target_threads_;    // live knob; workers park on a CV above this
  // worker: grab next RUNNABLE slot, advance one unit; on DONE route to the
  //         GameSink and refill the slot with a fresh game.
};
```

This buys two things:

1. **Smooth, live thread tuning.** Raising or lowering `target_threads_` parks or
   wakes workers between work-units. No game is abandoned mid-play, no pool
   rebuild. T becomes a knob the resource manager (below) turns continuously.

2. **Batched GPU inference — the neural-self-play substrate.** Once self-play
   uses a neural agent, a "unit of work" becomes *advance this game until it
   needs a network evaluation, then yield*. With G ≫ T games in flight, many are
   simultaneously blocked on an eval, and those requests gather into one batched
   GPU forward instead of T tiny per-thread inferences. The current
   one-whole-game-per-thread structure cannot express this. Building the pool now
   is what makes neural self-play efficient later.

For the HastyBot (GPU-free) case a "unit of work" can be an entire game — the
pool structure is identical; only the eval-batching layer differs. The disk
`GameRunner` path and the streaming ring-buffer path both ride the same pool.

## Producer process model: relaunch-per-chunk vs. run-forever

Generation runs as a fleet of generator workers
([scribblez/workloads/selfplay_gen.py](../py/scribblez/workloads/selfplay_gen.py)),
each of which **relaunches the `play_game` subprocess once per chunk**: a cycle
runs `play_game --games N --threads T` into the worker's work dir and the
process exits at N games. The thread count is a per-worker-slot setting applied
at the next spawn — **chunk-boundary granularity**.

A **run-forever** producer — one long-lived process (or in-process game-pool)
that never exits, is told which generation directory to write to, and is retuned
live — is the alternative, and the shape AlphaZeroArcade uses. It exists there for
reasons that are **specifically neural**:

- **Game length.** Neural self-play games are long (MCTS runs many GPU
  evaluations per move). A fixed-game-count-then-exit batch leaves the GPU
  underutilized at the tail, while the last few long games finish and the rest of
  the pool sits idle.
- **Weight refresh beats reload.** Updating the weights of an already-loaded
  TensorRT engine is far cheaper than loading a fresh ONNX on a restart. A
  run-forever process swaps weights in place; a relaunch pays the full load each
  generation.
- **Straddling boundaries.** Because games are long and weight-refresh is cheap,
  AZA lets a game **start under model N and finish under model N+1 (or later)**:
  when a new model is ready, the C++ pauses all in-flight games, refreshes the
  weights, and unpauses. No game is discarded at a boundary, so there is no tail
  to waste.

**For HastyBot self-play none of this applies.** Games are ~milliseconds, there is
no model to load or refresh, and there is no GPU tail. Relaunch-per-generation
costs only a process spawn plus agent construction per generation — negligible
against a 20k-game generation. So the run-forever machinery buys essentially
nothing in the current regime; its value arrives with neural agents.

### What a run-forever producer needs beyond the game-pool

The [game-pool](#the-game-pool-producer-c) (G ≫ T, live `target_threads_`) is the
prerequisite — without it "live threads" and "run-forever" are the same problem.
But it is not sufficient; a forever producer also needs:

- **Flow control (pause/resume).** A forever producer outruns a slower trainer and
  floods disk. It must produce ~a generation ahead, then park until the window
  slides — the same pause/resume AZA drives on its workers. This is the largest
  functional gap beyond the pool.
- **Drain-and-flush at boundaries.** Rolling from gen N to N+1, in-flight games
  must be resolved and the current `.slog` file **flushed and closed** so that
  every file belongs to exactly one generation. The DataLoader trusts each file's
  header (`num_sample_positions`, game count) and the lifecycle counts committed
  games per directory, so a file straddling N/N+1 corrupts both. (This is distinct
  from AZA's *game*-level straddling above: a single game may span model versions,
  but a single *file* must not span generation directories.) Relaunch gets this
  free — process exit flushes everything.
- **Crash supervision.** Relaunch-per-generation makes a crashed producer a
  self-healing partial generation (manifest still `generating`, count < target →
  regenerate). A forever process must be supervised for liveness and, on death,
  relaunched and re-pointed at the current generation with its already-committed
  count. You trade *routine* restarts for *crash* restarts, not zero restarts.
- **A control path.** Relaunch needs none. Forever needs Python→C++ messages
  (switch-directory, set-threads, pause/resume, and later refresh-weights). The
  simplest local form removes the problem entirely: run the producer **in-process
  via FFI** (as the C++ streaming ring buffer already supports), so these become
  direct calls rather than a wire protocol. A **subprocess + socket** is warranted only for
  process isolation (a C++ crash not taking the trainer down) or the path to
  remote workers; if taken, design the channel to also carry the weight-refresh
  message the neural phase will need.

### Sequencing: capture now, implement with the neural work

The neural-specific pieces — straddling, in-place weight refresh, and the
GPU-tail motivation — **cannot be designed correctly before the TensorRT neural
agent exists**, because their interfaces (the eval-batching hook on the pool, the
weight-refresh API, GPU contention) are defined by that agent. Building them now
against HastyBot means writing stubs against guessed APIs that cannot be exercised
(no tail, no weights) and will likely be re-cut when the real agent lands. Even
the game-pool's reusable *skeleton* (circular buffer, CV-parked workers,
round-robin) has one neural-dependent seam — the "unit of work / yield-for-eval"
granularity — best cut once the MCTS structure is known.

The durable move is therefore to **record this design now** (this section) and
implement the run-forever mechanics as part of [Roadmap](#roadmap) step 3, when
the neural agent makes those interfaces concrete. The one present-value exception
is mid-generation live-T for a HastyBot auto-balancer (see
[Resource contention](#two-regimes)): if that is built, the game-pool pays for
itself independent of neural — but the balancer is itself deferred, and
boundary-granularity thread tuning already covers most of the need.

## The lifecycle orchestrator (Python)

The orchestrator is the trainer role and drives its per-minibatch step
through a shared per-task `run_epoch`
([position_eval/train_loop.py](../py/scribblez/position_eval/train_loop.py),
[max_move_per_lane/train_loop.py](../py/scribblez/max_move_per_lane/train_loop.py)),
so the gradient step is isolated from the lifecycle. The learning rate is set per
step from a pure function of the rows-clock (see below).

The trainer is a **pure consumer**: it never generates. Generator workers stage
whole-file chunks, and the generation scheduler on the controller host assigns
them to generation directories and marks completion in the manifests (see
[docs/position_eval_workload.md](position_eval_workload.md) for the protocol).
The trainer's loop is

```python
# the train role (skeleton)
state = resume_generational(paths, model, optimizer)   # rows_trained, gen_idx, epoch_in_gen
publish_train_state(paths, state)                      # the scheduler's pacing cursor

while params.max_rows == 0 or state.rows_trained < params.max_rows:
    wait_for_generation(paths, state.gen_idx)          # poll manifests until complete
    window = lifecycle.window_dirs(paths, state.gen_idx, params.window)
    ds     = SlogDataset(window, num_workers=cpu.dataloader_workers)

    for e in range(state.epoch_in_gen, epochs_for_reuse(...)):
        result = run_epoch(model, optimizer, ds, device, ...,
                           epoch_index=global_epoch(state, e),
                           rows_trained=state.rows_trained, lr_fn=lr_fn)
        checkpoint_eval_and_export(...)                # keyed on rows_trained
        publish_train_state(paths, state)

    lifecycle.evict_beyond_window(paths, state.gen_idx, params.window)
    state.gen_idx += 1
    state.epoch_in_gen = 0
```

Knobs (frozen task params): `games_per_generation`, `test_games`, `open_ahead`,
`reuse_per_position` (derives epochs-per-generation), `window` (W generations
kept), `turns_per_game`, `warmup_rows`, `max_rows`.

### Overlap: continuous generation with an ahead-limit

A strictly serial *generate a generation, then train on it* idles the GPU during
generation — the exact resource waste we are trying to escape. Instead the
generator fleet runs continuously while the trainer trains: the scheduler keeps
the next generation open while its index is within `open_ahead` of the trainer's
published cursor, and **gates** (parks) the fleet beyond that. When the trainer
finishes its passes, the next generation is already on disk. This gets the
streaming shape's CPU/GPU overlap *and* keeps the reuse benefit; how many cores
generation gets versus the DataLoader is the generator slots' thread counts
versus the trainer's DataLoader control.

### Learning rate: a persisted manual control

A per-epoch cosine annealing schedule (`CosineAnnealingLR(T_max=epochs)`) is
incompatible with an open-ended, stop-and-resume loop: it assumes a known total
epoch count and anneals to ~0 at the end, but an open-ended, moving-target run has
no fixed annealing horizon. Rather than compute the rate from a schedule, the **base
learning rate is a live control value** managed from the dashboard's
[Controls tab](#the-controls-tab), persisted to the control table and restored on
restart. This follows KataGo and LeelaChessZero, which run a fixed learning rate
that operators step down by hand when the loss (or Elo) plateaus.

Two refinements on top of a bare constant:

- **Startup warmup on the rows-clock.** The effective rate is
  `base_lr * min(1, rows_trained / warmup_rows)` for the first `warmup_rows`, then
  `base_lr`. Warmup prevents early instability and is standard even in the
  fixed-LR systems above. On a mid-run restart it is a no-op, since `rows_trained`
  is already past `warmup_rows`. The rows-clock thus remains the LR *time axis*
  (warmup clock and the key for the events below); only the LR *value* is manual.

- **LR changes are logged as events keyed on the rows-clock**, so the loss plots
  can annotate where the rate was dropped — how operators actually read these
  curves, and what keeps a manual run auditable even though the LR history is run
  state rather than a function of the args.

Caveat: the manual step-down wisdom from KataGo/LC0 comes from
SGD-with-momentum training. This project uses AdamW, whose per-parameter
adaptation already absorbs much of what a manual LR drop provides, so expect the
effect of a drop to be smaller than in those systems (a late drop still typically
yields a final loss improvement). Following their playbook closely would mean
revisiting the optimizer; staying on AdamW is fine, with tempered expectations
for manual annealing.

## Restart and state

Restart must be a plain "run the script again." The authority for what has been
done is a small **commit-tracking DB** (SQLite, or a per-generation
`manifest.json`), not a fragile file-count heuristic:

- Per generation: `{generation, target_games, committed_rows, status}` where
  `status ∈ {generating, complete}`.
- The rolling checkpoint carries `rows_trained`, `generation_index`, and
  `epoch_in_generation` alongside the model/optimizer/scheduler state.

On startup the trainer reconciles disk against this state:

- Cursor generation `status: generating` → wait for the generation scheduler to
  finish filling it (chunks are whole files assigned by a single writer, so
  counting committed games against the target is reliable — there are no
  half-written rows and nothing to regenerate).
- Checkpoint's `epoch_in_generation < epochs_per_generation` → resume passes over
  the current window.
- Otherwise → advance to the next generation.

This is the local-single-process version of AlphaZeroArcade's crash tolerance,
where a `staged/unstaged` commit state machine discards a dead worker's
unstaged rows while committing staged ones, and all state is reconstructed from
on-disk DBs on restart.

## Resource contention

Three components compete for the CPU, and — in the neural regime — for the GPU:

| Consumer | Pool knob | Hot when |
|----------|-----------|----------|
| Game generation | `GamePool::target_threads_` | refilling |
| DataLoader (decode/shuffle/augment) | `NativeDataLoader(num_workers)` ([ffi.py](../py/scribblez/ffi.py)) | training |
| PyTorch CPU side | `torch.set_num_threads` | backward + host↔device collation |

The right abstraction is a **contention manager** over resources, domains, and
priorities — a generalization of a CPU-core budget. Each consumer is a domain
that requests resources; a resource is either divisible (CPU cores) or an
exclusive lock (a GPU). This is deliberately modeled on AlphaZeroArcade's
`GpuContentionTable`, where domains (TRAINING, SELF_PLAY, RATINGS) contend for a
per-GPU lock awarded by priority.

### Two regimes

- **GPU-free generation (HastyBot).** Generation never requests the GPU lock, so
  only the CPU split matters. A feedback controller partitions a core budget
  `C = gen + dataloader + torch` using the signals already emitted:
  `producer_blocked_ns` / `consumer_blocked_ns` from the C++ streaming ring
  buffer ([streaming_row_buffer.h](../engine/include/data/streaming_row_buffer.h)),
  an analogous `loader_stats()` to be surfaced from the C++ DataLoader, and the GPU-busy
  fraction as the north star. Consumer-blocked → shift cores toward data;
  producer-blocked → reclaim them for generation; GPU busy > ~90% → hold. The
  game-pool's live `target_threads_` is precisely the actuator this controller
  needs; reconfiguration happens at generation boundaries (rebuild the dataset,
  retarget the pool, `set_num_threads`).

- **Neural generation (shared GPU).** Self-play now competes with training for
  the GPU and the two cannot both run full-tilt on one device. The contention
  manager awards the GPU lock by priority (TRAINING > SELF_PLAY); the current
  holder yields cooperatively when a higher-priority domain asks. AlphaZeroArcade
  additionally supports a "hijack" that fully evicts self-play, and a
  "switcheroo" that moves training to a second GPU rather than leaving it idle.
  When self-play runs on a *different* GPU or host, there is no contention at all
  — which is the whole point of remote workers.

The orchestrator does **not** special-case neural versus HastyBot. Every consumer
is a domain requesting resources; HastyBot generation simply never requests the
GPU lock.

### The Controls tab

Live operator knobs share one mechanism: a **control table** of values the
trainer polls at each generation boundary, written by a dashboard **Controls
tab** and read by the trainer — no live IPC into the hot loop. Every control is
persisted and its last value restored on restart, so a run resumes exactly as it
was left. The tab holds:

- **Base learning rate.** Stepped down by hand off the loss plots (see
  [Learning rate](#learning-rate-a-persisted-manual-control)); each change is
  logged as a rows-clock event that annotates the metric curves.
- **DataLoader workers** and **torch intra-op threads** — the trainer's CPU
  pools. Generation capacity is not a control here: it belongs to the generator
  worker slots (threads/vcpus per slot in the master dashboard). A future
  "auto-balance" toggle could hand the trainer-side split to the feedback
  controller, which the tab then merely visualizes.
- Later, **per-domain GPU priorities** for the neural regime's contention lock.

This rides the per-tag dashboard DB the metrics already live in (the `control`
and `control_event` tables).

## Distributed game generation

Game generation is farmed to interchangeable generator workers — local
subprocesses and rented cloud pods — through the master dashboard; see
[docs/position_eval_workload.md](position_eval_workload.md) for the built
protocol. The key simplification, inherited from AlphaZeroArcade's localhost
special case, is that **the on-disk generation directory is the transport**:
workers produce whole `.slog` chunks (into a staging area, locally or via the
results bucket), and the generation scheduler on the controller host assigns
them to generation directories with plain renames. No worker ever needs to know
which generation is open, and no networking exists beyond the bucket sync the
cloud scaffolding already provides.

The controller/worker shape, mapped onto this system:

| AlphaZeroArcade | Here |
|-----------------|------|
| LoopController (central: trainer, model, generation clock, contention manager) | the trainer role + the generation scheduler |
| SelfPlayServer (thin per-machine launcher) | the worker entrypoint running the generate role |
| self-play worker (C++ binary, own TCP connection) | `play_game` chunks delivered to staging |
| Controller pushes new model per generation | (neural phase) chunks stamped with their model version; workers pull before producing |

The neural regime adds the missing piece: generation that needs the current
model must learn which weights to run. Model distribution can piggyback on the
scheduler's existing gate/ungate cycle rather than a bespoke push flow.

## Why a window, not a wipe

The initial instinct is to wipe each generation and regenerate. AlphaZeroArcade
never wipes: generations are delimited by a cumulative committed-row threshold,
and training runs over a *sliding window of the most recent rows*. Wipe is just
the W = 1 degenerate case. For neural self-play the sliding window is the correct
default, not an optimization: data generated by an older, weaker model ages out
of the window automatically as fresh data arrives, without a hard reset of the
data distribution. For stationary HastyBot data either works, so a window with
modest W is the safe universal default.

A related dial is **discrete versus continuous**:

- **Discrete generations + reuse-factor passes** (this proposal's default):
  simpler bookkeeping, coarser cadence. Correct for stationary HastyBot data,
  where reuse is the whole point and freshness does not matter.
- **Continuous sliding window** (AlphaZeroArcade's approach): the window advances
  and the trainer steps as rows accumulate, maximizing overlap and freshness.
  Preferred for neural self-play, where the data distribution is a moving target
  and stale rows actively hurt.

Both sit on the same rows-clock and window abstraction, so this is a parameter,
not a rewrite.

## Roadmap

| Step | Build | Status | GPU contention |
|------|-------|--------|----------------|
| 1 | Discrete-generation lifecycle on the rows-clock; per-directory manifests; shared `run_epoch`; rows-clock LR | built | none (HastyBot) |
| 2 | Distributed generation: generator worker fleet + generation scheduler (staging ingest, pacing gate) on the master dashboard | built | none (HastyBot) |
| 3 | Game-pool producer (G ≫ T, live `target_threads_`); `ContentionManager` (resource/domain/priority); continuous sliding row-window as a parameter | future | abstraction only |
| 4 | Neural self-play: batched eval on the game-pool; GPU lock (TRAINING > SELF_PLAY + hijack); model-stamped chunks + model distribution to generators | future | yes (priority lock) |

The game-pool underpins all later generation, and keeping "who fills a
generation" behind the staging/manifest interface is what makes steps 3 and 4
additive rather than rewrites.

## Deliberately out of scope

AlphaZeroArcade carries substantial machinery this system does not need at
current scale, and which should not be copied preemptively: the multi-database
schema (clients/self-play/training/ratings), the ratings / self-eval /
eval-vs-benchmark domains, fork-run retrain windows, and the two-filesystem
scratch↔persistent cloud syncer. The load-bearing ideas to adopt now are narrow:
the rows-clock, the sliding window, the commit-tracking DB, and the game-pool.
Everything else is a later bolt-on.

## Open questions

- **Generation size.** Large enough that in-generation shuffling decorrelates
  batches, small enough that background refill keeps pace with training. Needs
  measurement once step 1 exists.
- **Where the new bottleneck lands.** Reuse may relax the CPU bottleneck enough
  that the GPU becomes the constraint, at which point the three-way CPU controller
  is moot and the answer is a bigger model or batch. Build step 1, measure, then
  decide how much of step 2's controller is worth it. The block-ns instrumentation
  is worth adding early regardless, because it tells us which world we are in.
- **Reuse factor.** The right passes-per-position for the HastyBot data
  distribution is unknown; it likely differs for neural data.
</content>
</invoke>

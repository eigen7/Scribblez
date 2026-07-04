# Generational training

This is the training pipeline — implemented as
[scripts/post_move_value/train.py](../py/scripts/post_move_value/train.py) and
[scripts/max_move_per_lane/train.py](../py/scripts/max_move_per_lane/train.py) —
plus the C++ and coordination machinery it grows into. The core lifecycle
(rows-clock, generations, the sliding window, reuse-driven epochs, restart
reconciliation, and the live dashboard controls) is built; the game-pool
producer, the resource-contention manager, and distributed workers described
below are forward-looking. For the data pipeline it builds on, see
[docs/architecture.md](architecture.md).

## Motivation

Generational training replaced two earlier pipelines that sat at opposite
extremes:

- **Streaming**: C++ self-play fed sampled rows straight into the GPU loop
  through an in-process ring buffer. One position per game, no shuffle, no disk.
  Each position was used once and dropped. Ideal when generation is cheap and
  abundant (HastyBot self-play), but it wasted the expensive part of every game —
  the game was played to completion to yield a single training row. (The C++ ring
  buffer, `StreamingTrainSource`, is retained as the substrate the neural game
  pool will reuse; see [The game-pool producer](#the-game-pool-producer-c).)

- **Disk** ([generate_data.py](../py/scripts/generate_data.py) + a fixed-epoch
  trainer): generate all `.slog` data up front, then epoch over it. Reused every
  position across epochs and sampled many turns per game, but the lifecycle was
  rigid — generate everything, then train everything — with no stop-and-resume
  ergonomics.

The streaming trainer was heavily CPU-bound: the GPU starved waiting for game
generation. The obvious lever is to extract more gradient signal from each
generated position — sample more turns per game, and reuse each position across
several passes — which the disk pipeline could do but the streaming pipeline
could not. The motivation sharpens when self-play graduates from HastyBot to a
neural agent: generating a game becomes far more expensive, so squeezing maximum
training value out of each one stops being optional.

**Generational training** gets the disk pipeline's data reuse with the streaming
pipeline's stop-and-resume ergonomics, and its structure extends cleanly to
neural self-play and, eventually, remote game-generation workers.

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
tags/<tag>/data/
  test/                      # frozen ONCE at run start, never regenerated
  generations/
    gen_000042/
      manifest.json          # {target_games, seed, player_spec, status}
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

### Decorrelation and overfitting are already handled

Two mechanisms the pipeline needs already exist:

- **Decorrelation within a generation** is free: `SlogDataset.iter_batches`
  shuffles the whole loaded set each epoch via the C++ loader's `epoch_start`.
  As long as a generation holds many games, batches are decorrelated without a
  separate shuffle pass.

- **Multi-sample-per-game with anti-overfit** is `--turns-per-game K` with a
  per-epoch `epoch_index`: it draws a *fresh* window of K turns each epoch, so E
  epochs over one generation yield up to `E*K` *distinct* positions per game —
  not the same K rows hammered E times.

So the genuinely new work is the **lifecycle** (generate → train a few passes →
slide the window → repeat, restartable at any point) and the **producer and
resource-management** machinery below. The sampling and shuffling are done.

### The overfitting knob is reuse, not epochs

The quantity to bound is **gradient passes per unique position**, not epoch
count. A generation of G unique positions trained for E epochs at K turns/game
gives `E * min(K*games, G)` gradient samples over G uniques. Expose a target
**reuse factor** and derive the epochs-per-generation from the generation's
size, rather than exposing raw epochs — otherwise a small generation overfits at
the same E where a large one is fine.

## The game-pool producer (C++)

Today each self-play worker thread **owns an agent pair and plays one whole game
start to finish** ([self_play_engine.h](../engine/include/scribblez/self_play_engine.h),
[streaming_game_producer.h](../engine/include/scribblez/streaming_game_producer.h)),
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

## Producer process model: relaunch-per-generation vs. run-forever

The generational trainer currently **relaunches the `play_game` subprocess once
per generation** ([train.py](../py/scripts/post_move_value/train.py)
via [generate_data.run_games](../py/scripts/generate_data.py)): each generation
opens its directory, runs `play_game --games N --threads T` into it, and the
process exits at N games. The thread count is a live control (`CpuController`),
but it only takes effect at the next launch — **generation-boundary granularity**.

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
  via FFI** (as the `StreamingTrainSource` ring buffer already does), so these become direct calls
  rather than a wire protocol. A **subprocess + socket** is warranted only for
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

The orchestrator lives in the trainer script and drives its per-minibatch step
through a shared per-task `run_epoch`
([post_move_value/train_loop.py](../py/scribblez/post_move_value/train_loop.py),
[max_move_per_lane/train_loop.py](../py/scribblez/max_move_per_lane/train_loop.py)),
so the gradient step is isolated from the lifecycle. The learning rate is set per
step from a pure function of the rows-clock (see below).

```python
# train.py (skeleton)
state = resume_generational(paths, model, optimizer)   # rows_trained, gen_idx, epoch_in_gen
lr_fn = make_lr_fn(args.lr, args.warmup_rows)
ensure_frozen_test_split(paths, args)                  # generate ONCE if absent
refiller = BackgroundRefiller(paths, args) if args.window > 1 else None

while args.max_rows == 0 or state.rows_trained < args.max_rows:
    gen_dir = lifecycle.ensure_generation(paths, state.gen_idx, args, resources)  # (re)fill if partial
    window  = lifecycle.window_dirs(paths, state.gen_idx, args.window)
    ds      = SlogDataset(window, num_workers=resources.dataloader_workers())

    if refiller:                                       # fill gen_idx+1 in the background
        refiller.start(state.gen_idx + 1, resources)

    for e in range(state.epoch_in_gen, args.epochs_per_gen):
        metrics, state.rows_trained = run_epoch(model, optimizer, ds, device, args,
                                                epoch_index=global_epoch(state, e),
                                                rows_trained=state.rows_trained, lr_fn=lr_fn, ...)
        run_probes_and_calibration(...)                # keyed on rows_trained
        save_generational_checkpoint(paths, ..., epoch_in_gen=e + 1)
        resources.rebalance(loader_stats=ds.loader_stats(), gpu_busy_frac=meter.gpu_busy_frac())

    state.gen_idx += 1
    state.epoch_in_gen = 0
    if refiller: refiller.promote()
    else:        lifecycle.evict_beyond_window(paths, state.gen_idx, args.window)
```

Knobs: `--games-per-generation`, `--reuse-factor` (derives epochs-per-generation),
`--window` (W generations kept), `--turns-per-game`, `--warmup-rows`, `--max-rows`.

### Overlap: background refill

A strictly serial *generate a generation, then train on it* idles the GPU during
generation — the exact resource waste we are trying to escape. `BackgroundRefiller`
runs generation of `gen_{N+1}` (an out-of-process game-pool producer, or a
worker; see [Distributed](#distributed-game-generation)) while the trainer reuses
the current window. When the trainer finishes its passes, the next generation is
already on disk. This recovers the streaming trainer's CPU/GPU overlap *and* keeps
the reuse benefit. How many cores the refiller gets versus the DataLoader is
exactly what the resource manager arbitrates.

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

On startup the orchestrator reconciles disk against this state:

- Latest generation `status: generating`, or game count < `target_games` → a
  **partial** generation: finish or regenerate it before training. Because
  `play_game` writes whole `.slog` files atomically per `--games-per-file`,
  counting complete files against the target is reliable — there are no
  half-written rows.
- Checkpoint's `epoch_in_generation < epochs_per_generation` → resume passes over
  the current window.
- Otherwise → open the next generation.

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
  `producer_blocked_ns` / `consumer_blocked_ns` from the streaming source
  ([ffi.py](../py/scribblez/ffi.py) `StreamingTrainSource.stats`), an analogous
  `loader_stats()` to be surfaced from the C++ DataLoader, and the GPU-busy
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
- **CPU core budget** `C`, split across generation / DataLoader / torch. An
  "auto-balance" toggle hands the split to the feedback controller, which the tab
  then merely visualizes (stacked cores over time, overlaid on GPU-busy fraction
  and the producer/consumer block-ratio); off, sliders pin it manually.
- Later, **per-domain GPU priorities** for the neural regime's contention lock.

This rides the same throughput/metrics tables already written during streaming
training.

## Distributed game generation

The neural regime makes remote workers attractive: generation that needs a GPU
can be farmed to other machines so it never contends with the training GPU.
AlphaZeroArcade's controller/worker split is the reference design.

**The key simplification: on a single host, the on-disk generation directory *is*
the transport.** AlphaZeroArcade's data path has a localhost special case — when
a worker runs on the controller's host, the C++ binary writes its data file
directly to disk and the controller simply moves it; the length-prefixed
JSON-over-TCP machinery only engages for genuinely remote workers. That localhost
path is exactly the `.slog`-files-in-`gen_N/` interface above. Therefore:

- The in-process producer, out-of-process local workers, and neural self-play all
  need **zero networking** — they write `.slog` into the generation directory.
- Remote workers are a purely additive final step: a control channel plus a
  file-blob socket bolted onto the *same* generation-directory interface. Nothing
  upstream changes.

The controller/worker shape, mapped onto this system:

| AlphaZeroArcade | Here |
|-----------------|------|
| LoopController (central: trainer, model, generation clock, contention manager) | the lifecycle orchestrator |
| SelfPlayServer (thin per-machine launcher) | wrapper that runs a game-pool producer in-process or as a worker |
| self-play worker (C++ binary, own TCP connection) | game-pool producer writing `.slog` |
| Per-generation binary relaunch (gen-0 no model → later gens neural) | per-generation regeneration (gen-0 HastyBot → later neural) |
| Controller pushes new model per generation | controller publishes `gen_N/model.onnx`; workers pull before producing `gen_{N+1}` |

Properties worth adopting when that step comes: **workers dial into the
controller** (a passive listener), so adding a machine is "point another process
at host:port" with no controller-side reconfiguration; the **Python supervisor
holds a durable control connection while the C++ worker holds the data/weights
connection**, so the binary can be relaunched per generation (config changes like
no-model gen-0) without losing registration; and **model distribution piggybacks
on the pause/resume cycle** rather than a bespoke push flow — on a single box that
cycle is the same GPU-lock handoff the contention manager already performs.

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

| Step | Build | Networking | GPU contention |
|------|-------|-----------|----------------|
| 1 | Game-pool producer (G ≫ T, live `target_threads_`); discrete-generation lifecycle on the rows-clock; commit-tracking DB; shared `run_epoch`; rows-clock LR | none | none (HastyBot) |
| 2 | CPU budget → `ContentionManager` (resource/domain/priority); sliding row-window; discrete-vs-continuous as a parameter; dashboard control widget | none | abstraction only |
| 3 | Neural self-play in-process: batched eval on the game-pool; GPU lock (TRAINING > SELF_PLAY + hijack); model reload on resume | none | yes (priority lock) |
| 4 | Remote workers: control channel + file-blob socket onto the existing generation-directory interface | yes | remote = own GPU, no contention |

Steps 1 and 2 are load-bearing and painful to retrofit: the game-pool underpins
all later generation, and keeping "who fills a generation" behind a producer
interface (plus the resource/domain abstraction) is what makes steps 3 and 4
additive rather than rewrites. The distributed layer is genuinely deferrable —
build it only when a second machine exists.

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

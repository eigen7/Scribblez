# Generational training

The training pipeline — implemented as the train roles of the position_eval
and max_move_per_lane workloads
([scribblez/position_eval/trainer.py](../py/scribblez/position_eval/trainer.py),
[scribblez/max_move_per_lane/trainer.py](../py/scribblez/max_move_per_lane/trainer.py))
— plus the machinery it grows into. The core lifecycle (rows-clock,
generations, sliding window, one epoch per generation, restart
reconciliation, live controls, distributed generation via the master
dashboard — see
[position_eval_workload.md](position_eval_workload.md)) is built; the
game-pool producer and the resource-contention manager are forward-looking.
For the data pipeline it builds on, see [architecture.md](architecture.md).

## Motivation

Two simpler shapes sit at opposite extremes: **streaming** (C++ self-play
feeds rows straight into the GPU loop through an in-process ring buffer; one
position per game, used once — wastes most of each generated game and starves
the GPU on generation) and **one-shot disk** (generate everything, then epoch
over it — full reuse, but a rigid lifecycle with no stop-and-resume). The
lever that matters is gradient signal per generated position — sample more
turns per game, reuse each position across passes — which matters even more
once self-play graduates from HastyBot to a far more expensive neural agent.
Generational training combines the disk shape's reuse with stop-and-resume
ergonomics and extends to neural self-play and remote generation workers.

## Core concepts

- **The rows-clock.** Every quantity that must survive a restart or index the
  dashboard is keyed on cumulative rows trained — the dashboard x-axis and
  the restart cursor (in the rolling checkpoint) — never on wall-clock.
- **Generations and the sliding window.** A generation is a batch of
  self-play games in its own directory
  (`data/generations/gen_NNNNNN/{manifest.json, *.slog}`); the trainer trains
  over a sliding window of the most recent `W` generations
  (`SlogDataset` takes a list of directories).
- **Decorrelation and multi-sampling come from the data pipeline.**
  `SlogDataset.iter_batches` shuffles the whole loaded set each pass, and
  `turns_per_game K` with the generation index as `epoch_index` draws a
  *fresh* window of K turns each pass — a game's `W` passes over its window
  residency yield up to `W·K` distinct positions, not the same K rows
  hammered `W` times.
- **Epoch and generation are the same clock.** Each generation is trained
  exactly once (one epoch over the window it completes), so a game's lifetime
  reuse is fixed at `window · turns_per_game` passes by construction. This
  bound is load-bearing: at ~40 passes per game (the old reuse-derived
  epochs), every position of a game sharing one WLD target let the model
  memorize game outcomes — train accuracy kept climbing while held-out
  quality and play strength decayed from ~1M rows on. At 4 passes the same
  budget of rows keeps improving.

## The lifecycle

The trainer is a pure consumer: generator workers stage whole-file chunks,
the generation scheduler assigns them to generation directories, and the
trainer waits for the cursor's generation to complete, trains one epoch over
the window, checkpoints/exports/publishes under the generation's index,
evicts beyond the window, and advances
(protocol details in [position_eval_workload.md](position_eval_workload.md)).
Generation overlaps training via the scheduler's ahead-limit: the fleet runs
continuously up to `open_ahead` generations in front of the trainer's
published cursor, then gates.

**Restart is "run the script again."** Authority for what has been done is
the per-generation manifests plus the rolling checkpoint (`rows_trained`,
`generation_index`); on startup the trainer reconciles disk against that
state — wait for a filling generation, or advance. Chunks are whole files
assigned by a single writer, so
counting committed games against targets is reliable.

**Learning rate is a persisted manual control, not a schedule.** An annealing
schedule assumes a known horizon, which an open-ended stop-and-resume run
lacks; instead the base LR is a live dashboard control (following
KataGo/LC0's operator-stepped fixed rate), every change logged as a
rows-clock event that annotates the loss plots. Caveat: the manual step-down
wisdom comes from SGD-with-momentum systems; AdamW absorbs much of what a
drop provides, so expect smaller effects.

**Live controls** (base LR, DataLoader workers, torch threads) share one
mechanism: a per-tag control table the trainer polls at its natural cadence —
no IPC into the hot loop — with values persisted and restored on restart.

## Why a window, not a wipe

Wipe-and-regenerate is the `W = 1` degenerate case. For neural self-play a
sliding window is the correct default: data from an older, weaker model ages
out automatically without a hard distribution reset. For stationary HastyBot
data either works, so a modest window is the safe universal default. The
related dial — discrete generations with reuse-factor passes (current) vs a
continuously advancing row window (fresher, right for neural data) — sits on
the same rows-clock and window abstraction, so it is a parameter, not a
rewrite.

## Forward-looking: the game-pool producer (C++)

Today each self-play worker thread owns an agent pair and plays one whole
game start to finish
([game_engine.h](../engine/include/arena/game_engine.h)), with
the thread count fixed at construction. The proposed pool decouples in-flight
games from worker threads — G active game slots with G ≫ T workers, each
advancing one game by one unit of work — buying:

1. **Live thread tuning**: workers park/wake between work-units, no game
   abandoned, no pool rebuild — the actuator a resource controller needs.
2. **Batched GPU inference — the neural-self-play substrate**: a unit of work
   becomes "advance until the game needs a network eval, then yield," so many
   in-flight games gather into one batched GPU forward. The
   one-game-per-thread structure cannot express this.

## Forward-looking: relaunch-per-chunk vs run-forever

Generators currently relaunch the `play_game` subprocess once per chunk;
thread-count changes apply at chunk boundaries. A run-forever producer (long-
lived, retuned live, weights refreshed in place, games straddling model
versions) is the AlphaZeroArcade shape, and its motivations are specifically
neural — long games, expensive model loads, GPU tails. For HastyBot none of
this applies (games are ~milliseconds, no model), so relaunch costs nothing.

Beyond the game-pool, a forever producer needs: flow control (produce ~a
generation ahead, then park); **drain-and-flush at generation boundaries** —
a `.slog` file must never straddle generation directories, since the loader
trusts file headers and the lifecycle counts committed games per directory
(relaunch gets this free: process exit flushes); crash supervision; and a
control path (simplest local form: run the producer in-process via FFI, so
control is direct calls, with subprocess+socket only for isolation or remote
workers).

The neural-specific interfaces (eval-batching hook, weight-refresh API) are
defined by the TensorRT neural agent and cannot be designed correctly before
it exists — so this design is recorded here and implemented at roadmap step
3, when the agent makes the interfaces concrete.

## Forward-looking: resource contention

Three consumers compete for CPU — game generation, the C++ DataLoader, and
PyTorch's own threads — and, in the neural regime, self-play competes with
training for the GPU. The abstraction is a contention manager over resources,
domains, and priorities (modeled on AlphaZeroArcade's `GpuContentionTable`):
divisible resources (CPU cores, split by a feedback controller reading
producer/consumer blocked-time counters with GPU-busy as the north star) and
exclusive locks (the GPU, awarded TRAINING > SELF_PLAY, yielded
cooperatively). The orchestrator never special-cases neural vs HastyBot —
HastyBot generation simply never requests the GPU lock.

## Roadmap

| Step | Build | Status | GPU contention |
|------|-------|--------|----------------|
| 1 | Discrete-generation lifecycle on the rows-clock; per-directory manifests; shared `run_epoch`; rows-clock LR | built | none (HastyBot) |
| 2 | Distributed generation: generator worker fleet + generation scheduler (staging ingest, pacing gate) on the master dashboard | built | none (HastyBot) |
| 3 | Game-pool producer (G ≫ T, live thread target); contention manager; continuous sliding row-window as a parameter | future | abstraction only |
| 4 | Neural self-play: batched eval on the game-pool; GPU priority lock; model-stamped chunks + model distribution to generators | future | yes |

Keeping "who fills a generation" behind the staging/manifest interface is
what makes steps 3 and 4 additive rather than rewrites.

## Deliberately out of scope

AlphaZeroArcade machinery not needed at this scale and not to be copied
preemptively: the multi-database schema, the ratings/self-eval domains,
fork-run retrain windows, the two-filesystem cloud syncer. The load-bearing
ideas adopted are narrow: the rows-clock, the sliding window, manifest-based
commit tracking, and the game-pool.

## Open questions

- **Generation size** — large enough that in-generation shuffling
  decorrelates batches, small enough that refill keeps pace; measure.
- **Where the bottleneck lands** — if reuse makes the GPU the constraint, the
  CPU controller is moot; the blocked-time instrumentation is worth adding
  early because it says which world we are in.
- **Reuse** — per-game passes are `window · turns_per_game`; whether the
  right setting differs for expensive neural data (where each game costs far
  more to generate) is unknown.

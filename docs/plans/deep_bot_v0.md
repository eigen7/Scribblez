# DeepBot v0: the leave inference model and the policy-set model

**Status: proposed 2026-10-14; not built.** The implementation plan for the
first two models of [deep_bot.md](deep_bot.md), each trained from
HastyBot-vs-HastyBot games in standard Scrabble (hidden leaves):

1. **The leave inference model (LIM),** the network form of DeepBot's
   [beliefs](deep_bot.md#beliefs). It predicts the opponent's rack from the
   observer's view of the game.
2. **The policy-set model (PSM),** in both of the
   [forms](deep_bot.md#the-policy-set-model) the design presents. Training
   both on the same data lets the measurements choose between them.

The LIM comes first. Its labels are free (true racks from replay, no sims),
so it is the quick way to prove the workload machinery. Its comparison with
`RackInferrer` also says whether the belief source matters for the PSM's
labels. The PSM workload takes its belief source as a parameter, so it does
not wait on a trained LIM.

Both models train on positions from the whole game except the endgame, with
no weighting by bag size.

## One workload per model

Each model is one dashboard workload that does everything: it generates data,
trains, slides its data window (deleting what falls out), and publishes eval
metrics after every generation. Both follow the generational shape of
[position_eval](../position_eval_workload.md) and reuse its machinery:

| Piece | Reused from |
|---|---|
| Spec and registration | `WorkloadSpec` / `RoleSpec` ([workloads/base.py](../../py/scribblez/workloads/base.py)), an entry in `WORKLOADS` |
| Self-play generation | `selfplay_gen.generate` with `face_up_leaves=False` |
| Staging, ingest into generations, gating | the generational scheduler and data home ([generational/scheduler.py](../../py/scribblez/generational/scheduler.py), [data_home.py](../../py/scribblez/generational/data_home.py)) |
| The window and eviction | `lifecycle.window_dirs`, `lifecycle.evict_beyond_window` ([generational/lifecycle.py](../../py/scribblez/generational/lifecycle.py)) |
| Records and metrics | `TrainRecorder` → `train_ingest` → `metrics` in `dashboard.db` ([generational/records.py](../../py/scribblez/generational/records.py)) |
| Dashboard | a `WORKLOAD_TABS` entry ([web/src/workloads.tsx](../../web/src/workloads.tsx)), plus one `FIGURES` builder per workload for its eval figure |

**Roles.**

- **`generate`:** parallel, CPU, local or ssh. Any number of workers.
- **`train`:** singleton, GPU. It hosts the data home, so the scheduler runs on
  its machine.

**The trainer's loop**, per generation:
1. wait for the generation;
2. train one epoch over the window;
3. evaluate on the held-out set;
4. stage the record;
5. evict generations beyond the window.

**Held-out data.** Every `holdout_every`-th generator chunk goes to a
held-out directory instead of a generation, as `move_set_eval` holds out its
sweep pairs. The trainer evaluates on the most recent `holdout_size`
positions from it, so the eval set never overlaps training data.

## Workload 1: `lim`

### Data and rows

The generator is unchanged self-play: HastyBot against HastyBot, with
`face_up_leaves=False`. A `.slog` stores the initial racks and every move with
the tiles drawn after it, so replay reconstructs every player's true rack at
every turn.

A row is one turn, before the endgame (the `.slog`'s eligible region, which
ends where the bag empties):

- **The observer** is the player to move.
- **The target** is the opponent's rack at that moment, as a sorted sequence
  of tiles. Its length is public: 7, or fewer near the end.

`turns_per_game` sets how many turns of each game a pass draws, as in
position_eval.

### Engine work

**A target.** `OppRack` in the `AllTargets` registry
([training/training_targets.h](../../engine/include/training/training_targets.h)):
the opponent's rack from replay, as up to 7 sorted tile indices (27 letters
including the blank) plus a length. It is a label only; nothing feeds it to
an input.

**An input block for history.** The position encoder's hidden-leaves arm
already encodes the board, the observer's rack and the unseen pool. The reads
the LIM exists for span several moves: a short play after a scoring play, or
a play the opponent passed up. A new block, `OppHistory`, encodes the
opponent's last `k` moves (`history_moves`, default 3):

- one plane per move, marking the squares that move placed tiles on;
- scalars per move: tiles played, whether it was an exchange (and how many
  tiles) or a pass, its score, and the bag count when it was made.

The block is registered in `InputEncodingSpec`
([encoding/input_encoder.h](../../engine/include/encoding/input_encoder.h)),
so the DataLoader and FFI carry it without new plumbing.

### Model

- **The trunk:** the position evaluation trunk, conv or transformer (a
  `profile`), on the hidden-leaves input plus `OppHistory`.
- **The rack head:** autoregressive over the sorted rack. At step `t` it
  emits a 27-way softmax given the trunk's pooled vector and the letters
  already emitted. Two masks keep it legal:
  - letters the observer's unseen pool has run out of, net of letters already
    emitted;
  - letters that sort before the previous one.

  A small transformer decoder over the steps is the default.
- **The loss:** the negative log-likelihood of the true sorted rack.

Sampling for DeepBot (inverse CDF from supplied variates, so racks pair
across candidates) and the ONNX export come with the PSM's `belief=lim` mode,
not in this workload.

### Eval metrics

Per generation, on the held-out positions:

| Metric | Meaning |
|---|---|
| `eval_nll` | mean NLL of the true rack, nats per rack |
| `eval_nll_vs_uniform` | `eval_nll` minus the NLL of a uniform draw from the unseen pool (exact, hypergeometric) |
| `eval_nll_vs_inferrer` | `eval_nll` minus `RackInferrer`'s NLL (its leave posterior from the last move, plus a uniform refill) |
| `eval_nll_bag_*` | `eval_nll` by bag band |
| `eval_letter_calibration` | mean absolute gap between predicted and actual expected count, per letter |

The `RackInferrer` baseline is computed once per held-out set by an engine
entry point (FFI or tool) and cached, since it does not change with the
model. Its comparison is what decides the PSM's default belief source.

### Parameters

`history_moves`, `turns_per_game`, `games_per_generation`, `window`,
`holdout_every`, `holdout_size`, `max_rows`, plus the trunk's and the
optimizer's usual parameters through profiles.

## Workload 2: `psm`

### Data: labeled pools

A generator worker's cycle:

1. **Play** one HastyBot-vs-HastyBot game, `face_up_leaves=False`.
2. **Pick positions.** `positions_per_game` turns (default 2), uniform over
   the eligible region. Surveying every turn is what makes blind_spots cost
   about ten minutes per game on 28 threads; two turns spread the same
   compute over more games.
3. **Survey each position** with `sim_candidate_survey_tool` in its
   exhaustive recipe, screen stage only (no confirm stage). The candidates
   are every legal play without a blank, the equity top cut (which may
   include blank plays), every distinct exchange, and the pass. Racing stays
   on.
4. **Write a pool record** per position (below).

**Opponent racks: the standard-Scrabble mode.** The survey tool's
hidden-leaves path today draws the opponent's whole rack uniformly from the
unseen pool. A new flag, `--opp-racks`, sets the belief:

- `uniform`: today's behavior;
- `inferrer`: `RackInferrer` on the opponent's last move, plus a uniform
  refill, through the same `OppLeaveSampler` the Monte Carlo ground-truth
  path uses ([sim/monte_carlo_sim.cpp](../../engine/src/sim/monte_carlo_sim.cpp));
- `file`: racks supplied per rollout from a file, which is how `belief=lim`
  will feed LIM samples drawn in Python.

Rollout `i` uses the same rack sample for every candidate, so the per-move
values stay paired.

**The pool record** (a new compact sidecar beside the `.slog`, not the
survey's JSON):

- **per position:** the `.slog` game and turn;
- **per candidate:** the `Move` (16 bytes); mean win equity; rollouts
  completed (racing stops some early); the paired standard error against the
  leader; the static-equity rank; and `q(a)`.

`q(a)` is the soft best-move label: the probability that `a` is best,
computed at generation time from the paired differences against the leader
under a normal approximation. It treats the non-leader moves as independent
of each other, an approximation acceptable for v0.

**Generations** are `positions_per_generation` pools. The window and
eviction work as in `lim`. Pools are expensive, so the window default is
large.

**Parameters:** `belief` (`uniform`, `inferrer` or `lim`),
`positions_per_game`, `rollouts` (screen budget, default 300), `race`,
`solve_max_unseen`, `positions_per_generation`, `window`, `holdout_every`,
`holdout_size`, `forms`, `max_rows`.

### Inputs at train time

Inputs are rebuilt by replay, as `move_set_eval`'s trainer does: the pool's
`.slog` reference and its candidate moves go to `ffi.move_set_positions`,
which returns the board rows and per-move cross-check cells and letters.
The process runs in the hidden-leaves arm (`ffi.set_opp_leave_input`,
adopted from a header flag in the pool sidecar). No legal-move enumeration is
needed, because the pool lists the candidates.

### The two forms

Both use the move set model's encoders (the board trunk, `MoveEncoder`,
`CrossCheckEncoder`), trained from scratch. Today's move set checkpoints are
face-up and differ in their input blocks.

**Form 1, `greedy`:** a set-conditioning stage plus a gain head.
- The stage is the evidence-fusion stage with chosen-move tokens and no sim
  channels ([evidence_fusion.py](../../py/scribblez/evidence_fusion.py)).
- Rows come by subset assembly. Per position, draw subsets `S` of sizes 0 to
  `N_max − 1`, biased toward high-value moves as deployment sets are. The
  target for a held-out move `a` is `max(0, v(a) − max_S v)`. A batch of
  subsets per position shares one trunk pass, which is the training-side form
  of the cached steps.
- The loss is Huber on the gain.

**Form 2, `slot`:** a slot decoder with `N_max` slots (default 16).
- The slots cross-attend to the move embeddings and self-attend to each
  other, and each points to a move by softmax.
- The loss is the coverage loss over slot prefixes, with the pool's `q(a)` as
  the label distribution, summed over `N` in {1, 2, 4, 8, 16}.

### Training both forms

The `forms` parameter is `greedy`, `slot` or `both`.

With `both`, one trainer holds two models and two optimizers and alternates
their steps on the same batches. Records carry the form as a key prefix
(`greedy_loss`, `slot_eval_recall_8`, ...), so the dashboard plots them side
by side. Both models are move-set-sized, so one GPU holds them.

A form per machine needs two trainer roles reading one data home. Eviction
would then wait for both trainers' pulls, an extension of the `pulled`
acknowledgment in `lifecycle`. That comes only if one GPU's throughput binds,
which it should not while label generation is the bottleneck.

### Eval metrics

Per generation and form, on held-out pools, for `N` in {1, 2, 4, 8, 16}:

| Metric | Meaning |
|---|---|
| `eval_recall_N` | how often the pool's sim-best move is in `S` |
| `eval_cover_N` | `Σ_{a∈S} q(a)`, the soft form of recall, P(best ∈ S) |
| `eval_regret_N` | mean `v(a*) − max_{a∈S} v(a)`, in win equity |
| `eval_tail_recall_N` | recall on pools whose sim-best move is outside the static-equity top 10 |
| `eval_set_ms` | time to build one batch of 256 sets at `N = 8` |

Each metric is reported beside the same metric for the static-equity top `N`,
the baseline every DeepBot set has to beat. The figure plots recall and
regret against `N` per form, over generations.

### Cost

From the blind_spots measurement, the exhaustive screen at 1,000 rollouts
costs about 30 seconds of a 28-thread machine per position. At 300 rollouts
with racing, it is roughly 10 seconds: about 360 pools per machine-hour, or
10,000 pools in about 28 machine-hours. Generation sets the pace, and it is
embarrassingly parallel, so the `generate` role is meant for a fleet
(owned ssh machines and rented CPU machines, as blind_spots runs).
Mid-game positions have more legal moves than small-bag ones, so the real
rate is measured on the first generation and shown on the Stats tab.

## Build order

Each step is a PR with its own tests.

**The LIM:**

1. **Engine:** the `OppRack` target, the `OppHistory` input block, and their
   FFI exposure, with encoder tests.
2. **Python:** the LIM model, its generational trainer and eval (the uniform
   baseline), the `lim` workload spec with held-out chunks, and the dashboard
   tab and figure.
3. **The `RackInferrer` baseline** on held-out sets, and `eval_nll_vs_inferrer`.

*Measure:* the LIM's NLL against uniform draws and against `RackInferrer`,
overall and by bag band. This decides the PSM's default belief.

**The PSM:**

4. **Engine:** `--opp-racks` in the survey tool (`uniform`, `inferrer`,
   `file`), exchanges and the pass in the exhaustive candidate set, and the
   pool sidecar with `q(a)`.
5. **Python:** the `psm` workload's generator, held-out pools, generations,
   and the pool reader on `ffi.move_set_positions`.
6. **The two forms,** their trainers under `forms`, the eval metrics and the
   figure.

*Measure:* recall, cover, regret and tail recall against `N` for each form,
and against static equity; time per batch of sets. This decides the form.

**Later:**

7. **`belief=lim`:** the LIM's ONNX export, seeded sampling, and the `file`
   rack source fed from it.
8. **A trainer per form,** if one GPU binds.

## Open questions

- **History length** for the LIM, and whether our own recent moves belong in
  `OppHistory` too: what the opponent drew depends on what we played.
- **Rack or leave as the LIM's target.** The full rack is simpler and is what
  DeepBot samples. Predicting the leave and adding the uniform refill is
  exact for the refill part and may train faster.
- **Blank plays in the PSM's candidate set.** v0 includes only those in the
  equity top cut. Every blank designation would multiply the pool.
- **`rollouts` against pool count.** Fewer rollouts per pool means noisier
  `q(a)` but more positions. The soft labels absorb some of the noise.

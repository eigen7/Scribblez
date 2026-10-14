# DeepBot v0: the leave inference model and the policy-set model

**Status: proposed 2026-10-14; not built.** The implementation plan for the
first two models of [deep_bot.md](deep_bot.md), each trained from
HastyBot-vs-HastyBot games in standard Scrabble (hidden leaves):

1. **The leave inference model (LIM),** the network form of DeepBot's
   [beliefs](deep_bot.md#beliefs). It predicts the opponent's leave (the
   tiles they kept from their last move) from the observer's view of the
   game. The opponent's rack is that leave plus a refill drawn uniformly from
   the rest of the observer's unseen pool.
2. **The policy-set model (PSM),** in both of the
   [forms](deep_bot.md#the-policy-set-model) the design presents. Training
   both on the same data lets the measurements choose between them.

The LIM comes first. Its labels are free (true leaves from replay, no sims),
so it is the quick way to prove the workload machinery, and the PSM's label
generation then draws opponent leaves from it.

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

- **`generate`:** parallel, local or ssh. Any number of workers.
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
the tiles drawn after it, so replay reconstructs every player's true rack, and
so every leave, at every turn.

A row is one turn, before the endgame (the `.slog`'s eligible region, which
ends where the bag empties):

- **The observer** is the player to move.
- **The target** is the opponent's leave from their last move, as a sorted
  sequence of tiles. Its length is public: the opponent's rack size minus the
  tiles that move played or exchanged (a pass keeps the whole rack).

`turns_per_game` sets how many turns of each game a pass draws, as in
position_eval.

### Engine work

The LIM is a standalone model, so its inputs come from their own encoder, not
from blocks added to the position encoder's `InputEncodingSpec` (which would
change every position model's inputs).

**The input encoder** (`LimInputEncoder`), from the observer's view:

- **The current position:** the board, the current cross-checks, the
  observer's rack, the unseen pool, scores and the bag count. These come from
  the same helpers the position encoder uses, so conventions match.
- **The last 5 moves:** the opponent's last 3 and the observer's last 2,
  interleaved in game order. For each move:
  - a plane marking the squares it placed tiles on;
  - the cross-checks as they stood *before* the move was played, so the
    model sees which hooks and lanes the player had and passed up;
  - scalars: tiles played, whether it was an exchange (and how many tiles) or
    a pass, its score, and the bag count when it was made.

  Missing moves early in the game are zero, with a presence flag.

**The target.** `OppLeave`: the opponent's leave from replay, as up to 7
sorted tile indices (27 letters including the blank) plus its length. It is a
label only.

**FFI.** A DataLoader mode that emits LIM input rows and `OppLeave` targets
from `.slog` windows, beside position_eval's mode.

### Model

A standalone network in `py/scribblez/lim/`. It has its own architecture and
does not share another model's trunk or weights.

**The encoder:** a small spatial tower over the board planes (the current
ones and each history move's planes, stacked as channels), plus an MLP over
the scalars, fused into a context vector.

**The output: 29 floats.** The context vector feeds a linear layer that emits:

- **`θ_l`, one per letter (27):** a log-weight for "the leave contains `l`".
  Positive means likely has it, negative means likely doesn't, so one number
  covers both reads.
- **`a` and `b ≥ 0`:** a tilt on the leave's absolute quality `v(ℓ)`, its
  static leave value in equity points (HastyBot's leave values,
  `HastyEquity::default_leaves_path()`).

Those numbers define the leave distribution as the pool prior, tilted:

`log p(ℓ) = log P_prior(ℓ | pool) + Σ_{l ∈ ℓ} θ_l + a·v(ℓ) − b·v(ℓ)² − log Z`

over leaves `ℓ` of the public length `L` that the observer's unseen pool can
supply. `P_prior` is the hypergeometric draw from that pool:
`Π_l C(n_l, c_l)` up to a constant, where `n_l` is the pool's count of `l` and
`c_l` the leave's.

- **`a = b = 0` is exactly "no information":** the leave is a uniform draw
  from the pool. `a > 0` shifts belief toward better leaves, `a < 0` toward
  worse ones, and `b > 0` concentrates it around a value.
- **The quality term carries the interactions.** Leave value already folds in
  balance, synergy, duplication and stems, so one tilt moves a whole class of
  leaves at once: raising `a` favors well-balanced, bingo-prone leaves and
  leaves unbalanced ones down. The letter terms stay per-letter, which is all
  the has-it and doesn't-have-it reads need.
- **Quality is absolute.** The tilt acts on leave value in points, against
  whatever the current pool can produce. With both blanks unseen, a positive
  `a` reaches up to blank leaves; with both played, it favors the best leaves
  that remain.

For v0 data the quality term fits especially well: HastyBot chooses its leave
by score plus exactly this leave value.

### The leave tables

Static, built once offline, one per length `L` from 1 to 6. Each row is a
leave: its 27 counts, a 27-bit presence mask, and `v(ℓ)`. The 6-tile table has
737,311 rows (about 30 MB); the 4-tile table has 25,254. A 7-tile leave arises
only from a pass, and v0 samples it from the prior alone. If leave values
later become board-aware, only the `v(ℓ)` column changes, per position.

### Likelihood and training

One pass over the table for the leave's length gives the exact
log-likelihood:

1. **Per-letter prior terms:** `log C(n_l, c)` for each letter and count, at
   most 27 × 8 values.
2. **Each row's log-weight:** skip the row if some `c_l > n_l`; otherwise
   `Σ_l logC[l][c_l] + Σ_{l ∈ mask} θ_l + a·v − b·v²`, about 28 additions.
3. **`log Z`** is the log-sum-exp of the log-weights, and the loss is
   `log Z` minus the true leave's log-weight.

In PyTorch, the pass is a gather and sum over the table's columns, batched over
rows and computed in chunks, and gradients flow to the 29 outputs. Training
directly on this likelihood matters. Per-letter has-it classifiers and a
separate quality regressor, trained independently, would give marginals that
are not the weights to multiply together. Trained jointly, the outputs *are*
the weights, and the readable per-letter probabilities ("likely has an S")
come out of the same pass as marginals, along with the implied mean and spread
of the opponent's leave value.

### Sampling

**Algorithm 1: enumerate, then inverse CDF.** The default.

1. Compute every row's log-weight, as above.
2. Exponentiate against the maximum and take a running sum in the table's
   fixed order, which gives the CDF.
3. For each sample's variate `u ∈ [0, 1)`, binary-search the CDF for
   `u × total`. That row is the leave.
4. Add the refill: `7 − L` tiles drawn uniformly from the pool minus the leave.

A 6-tile query is about 20M additions, 10–20 ms on one CPU thread, and each
sample after that is a binary search. Shorter leaves are far cheaper, and rows
the pool cannot supply drop out. Because the order is fixed and sampling
inverts a CDF, one variate gives the same leave whenever two queries have the
same distribution, so samples pair across candidates as everywhere else in
the design. The same pass computes `log Z`, so training, evaluation and
sampling share one implementation.

**Algorithm 2: per-letter proposal, then rejection.** An optimization for
many unbatched CPU queries, built only if profiling calls for it.

1. **Proposal:** drop the quality factor. What remains,
   `Π_l C(n_l, c_l) · exp(θ_l [c_l > 0])` at fixed length, factors over
   letters. A small table of partial sums over (letter, tiles still to place),
   about 27 × 7 × 7 operations, gives exact sampling by walking the letters.
2. **Accept** with probability `exp(a·v − b·v² − M)`, where `v` comes from a
   static index keyed by the leave's packed counts and `M` bounds the quality
   term over that length's range of values. If rejected, draw again.

Accepted samples are exact. The acceptance rate falls when the quality tilt
is strong, so the sampler estimates it from a few trial draws and falls back
to Algorithm 1 below about 5%. Pairing is looser, since a varying number of
variates is consumed.

**Where each runs:**

| Use | Volume | Algorithm |
|---|---|---|
| PSM label generation | one query per position, ~300 samples | Algorithm 1 on the CPU; 20 ms against tens of seconds of rollouts |
| The agent, Bob's beliefs | ~2,000 queries per turn, ~20 samples each | Algorithm 1 batched on the GPU, one pass over the table for all queries |
| Many unbatched CPU queries | — | Algorithm 2, falling back to 1 |

### The comparison head

A second profile, `head = autoregressive`, replaces the 29-float output with
an autoregressive decoder over the sorted leave: a 27-way softmax per step,
masked by the pool and by sort order, with the same encoder. It is flexible
where the tilted prior is constrained, so held-out NLL between the two
measures what the compact output gives up, overall and for 6-tile leaves.

**Export.** The workload exports each checkpoint to ONNX. The engine serves
the encoder through its TensorRT runtime as a new `NeuralNet` spec, which
emits the 29 floats; the sampling pass is engine code, not part of the graph.

### Eval metrics

Per generation, on the held-out positions:

| Metric | Meaning |
|---|---|
| `eval_nll` | mean NLL of the true leave, nats per leave |
| `eval_nll_vs_uniform` | `eval_nll` minus the NLL under the hypergeometric prior over leaves from the unseen pool ([belief/leave_prior.h](../../engine/include/belief/leave_prior.h)) |
| `eval_nll_vs_inferrer` | `eval_nll` minus `RackInferrer`'s NLL, its leave posterior from the last move |
| `eval_nll_bag_*` | `eval_nll` by bag band |
| `eval_letter_calibration` | mean absolute gap between predicted and actual presence probability, per letter |
| `eval_value_calibration` | the implied mean leave value against the true leave's value, and the coverage of the implied spread |

The two baselines are computed once per held-out set by an engine entry point
and cached, since they do not change with the model. `RackInferrer` is the
reference the LIM should beat: it reads one move under a HastyBot-equity
model, and in these games the opponent *is* HastyBot.

### Parameters

`turns_per_game`, `games_per_generation`, `window`, `holdout_every`,
`holdout_size`, `max_rows`, plus the network's and the optimizer's usual
parameters through profiles.

## The rollout leaf: a versioned win table

PSM labels come from many rollouts per candidate. Played to the end of the
game, each rollout is slow and its outcome is noisy. Instead, following
Macondo's simmer, a rollout stops after `K` plies (`horizon_plies`), and a
fast function turns the position there into a win probability:

`P(win) = table(spread + leave value, tiles unseen)`

- **The spread** is from the side to move, plus the leave value of that
  side's remaining rack, as Macondo adds its "leftover".
- **Tiles unseen** are the bag plus the opponent's rack, from the side to
  move.
- **If the game ends** inside the horizon, the outcome is exact. If the bag
  empties with few enough tiles unseen (`solve_max_unseen`), the endgame
  solver values the position.

The engine already reads Macondo's table format
([sim/win_pct_table.h](../../engine/include/sim/win_pct_table.h)). What is new
is that the table becomes a **versioned artifact** of our own:

- **Fitting.** `fit_win_table.py` estimates `P(win | spread + leave value,
  unseen)` from `.slog` outcomes: bucketed, smoothed, and made monotone in
  spread. It writes the same CSV format.
- **Versions.** Each version is a file under the mount (`win_tables/v0.csv`,
  `v1.csv`, ...), with the corpus and policy it was fitted from. v0 is fitted
  on HastyBot-vs-HastyBot games, which the `lim` workload produces anyway.
  Later versions are fitted on games of the improved agent, since the win
  probability at a given spread shifts as play gets stronger. Macondo's
  bundled table is the sanity check for v0.
- **Selection.** The survey tool takes the table's path (`--win-table`) and
  the horizon (`--horizon-plies`), and the pool sidecar records which version
  labeled it.

## Workload 2: `psm`

### Data: labeled pools

A generator worker's cycle:

1. **Play** one HastyBot-vs-HastyBot game, `face_up_leaves=False`.
2. **Pick positions.** `positions_per_game` turns (default 2), uniform over
   the eligible region.
3. **Survey each position** with `sim_candidate_survey_tool` in its
   exhaustive recipe, screen stage only (no confirm stage). Rollouts are
   truncated at the win table. The candidates are below; racing stays on.
4. **Write a pool record** per position (below).

**Candidates.** Every legal play and every distinct exchange, plus the pass,
**including blank plays**. A blank multiplies the moves through its
designations, so one prune applies: a blank move (a play that places the
blank, or an exchange that returns it) is dropped when its score is 40 or
more points below the greedy move's, the highest-scoring play. An exchange
scores 0 under this rule.

**Opponent racks.** A new flag, `--lim <path to LIM ONNX>`, sets the
opponent's leave belief:

- **With the flag,** the tool runs the LIM once per position, from the
  chooser's view, and samples one leave per rollout by [Algorithm
  1](#sampling) from a seeded variate, with a uniform refill from the rest of
  the unseen pool.
- **Without it,** the opponent's whole rack is drawn uniformly from the
  unseen pool, which is today's hidden-leaves behavior.

Rollout `i` uses the same rack sample for every candidate, so the per-move
values stay paired. With `--lim`, the generator role needs a GPU for the
LIM's TensorRT engine; one evaluation per position makes the need small.

**The pool record** (a new compact sidecar beside the `.slog`, not the
survey's JSON):

- **per pool:** the win table version, the LIM version (or uniform), and
  `horizon_plies`;
- **per position:** the `.slog` game and turn;
- **per candidate:** the `Move` (16 bytes); mean win probability; rollouts
  completed (racing stops some early); the paired standard error against the
  leader; the static-equity rank; and `q(a)`.

`q(a)` is the soft best-move label: the probability that `a` is best,
computed at generation time from the paired differences against the leader
under a normal approximation. It treats the non-leader moves as independent
of each other, an approximation acceptable for v0.

**Generations** are `positions_per_generation` pools. The window and
eviction work as in `lim`. Pools are expensive, so the window default is
large.

**Parameters:** `lim_onnx` (empty for uniform), `win_table`,
`horizon_plies`, `positions_per_game`, `rollouts` (screen budget, default
300), `race`, `solve_max_unseen`, `positions_per_generation`, `window`,
`holdout_every`, `holdout_size`, `forms`, `max_rows`.

### Inputs at train time

Inputs are rebuilt by replay, as `move_set_eval`'s trainer does. The pool's
`.slog` reference and its candidate moves go to `ffi.move_set_positions`,
which returns the board rows and per-move cross-check cells and letters. The
process runs in the hidden-leaves arm (`ffi.set_opp_leave_input`, adopted
from a header flag in the pool sidecar). No legal-move enumeration is needed,
because the pool lists the candidates.

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
| `eval_regret_N` | mean `v(a*) − max_{a∈S} v(a)`, in win probability |
| `eval_tail_recall_N` | recall on pools whose sim-best move is outside the static-equity top 10 |
| `eval_set_ms` | time to build one batch of 256 sets at `N = 8` |

Each metric is reported beside the same metric for the static-equity top `N`,
the baseline every DeepBot set has to beat. The figure plots recall and
regret against `N` per form, over generations.

### Cost

From the blind_spots measurement, the exhaustive screen at 1,000 rollouts
played to the end of the game costs about 30 seconds of a 28-thread machine
per position. Two changes cut that: 300 rollouts instead of 1,000, and
rollouts truncated at `K` plies instead of played out. Blank plays push the
other way, by enlarging the candidate set. A guess is several hundred pools
per machine-hour. The real rate is measured on the first generation and shown
on the Stats tab. Generation sets the pace and is embarrassingly parallel, so
the `generate` role is meant for a fleet (owned ssh machines and rented
machines, as blind_spots runs). With `--lim`, those machines need a GPU.

## Build order

Each step is a PR with its own tests.

**The LIM:**

1. **Engine:** `LimInputEncoder` (current position, 5-move history with
   pre-move cross-checks), the `OppLeave` target, and the DataLoader/FFI mode,
   with encoder tests.
2. **Python:** the leave tables, the standalone LIM network with the
   tilted-prior head and the table pass, its generational trainer and eval
   (the uniform baseline), the `lim` workload spec with held-out chunks, and
   the dashboard tab and figure.
3. **The `RackInferrer` baseline** on held-out sets, and
   `eval_nll_vs_inferrer`.
4. **Serving:** the LIM's ONNX export, its `NeuralNet` spec on TensorRT, and
   Algorithm 1 in the engine (CPU, and batched on the GPU), with parity tests
   against the PyTorch model and its table pass.
5. **The comparison head:** `head = autoregressive`.

*Measure:* the LIM's NLL against the uniform prior and against
`RackInferrer`, overall and by bag band, for both heads.

**The win table:**

6. **`fit_win_table.py`** and the v0 table, fitted on the `lim` workload's
   games, compared with Macondo's bundled table.

**The PSM:**

7. **Engine:** in the survey tool, `--lim`, `--win-table` and
   `--horizon-plies`; blank plays with the 40-point prune, exchanges and the
   pass in the candidate set; and the pool sidecar with `q(a)`.
8. **Python:** the `psm` workload's generator, held-out pools, generations,
   and the pool reader on `ffi.move_set_positions`.
9. **The two forms,** their trainers under `forms`, the eval metrics and the
   figure.

*Measure:* recall, cover, regret and tail recall against `N` for each form,
and against static equity; time per batch of sets. This decides the form.

**Later:** a trainer per form, if one GPU binds.

## Open questions

- **`horizon_plies`.** Fewer plies are faster and less noisy, but lean harder
  on the win table. Macondo uses 5 plies with more than 14 tiles unseen.
- **`rollouts` against pool count.** Fewer rollouts per pool means noisier
  `q(a)` but more positions. The soft labels absorb some of the noise.
- **A CPU path for the LIM** in the generator, if renting GPU machines for
  the fleet proves costly. One evaluation per position is cheap enough for a
  CPU runtime, but the engine serves networks only through TensorRT today.

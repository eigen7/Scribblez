# Scribblez Project Roadmap

## Overview

Scribblez aims to build a superhuman Scrabble AI by training neural networks
that overcome two fundamental weaknesses of existing engines like Macondo:
context-blind leave evaluation and naive opponent rack inference. This document
lays out the planned progression from the current position evaluation model through
a full move set evaluation model, the data-generation strategies that support each stage,
and the evaluation machinery needed to validate progress at every step.

---

## Phase 1: Position Evaluation Model — Current

### What it is

The position evaluation model evaluates a board state *after* the active player has placed tiles but
*before* drawing replacements from the bag. Given the board, the player's
residual rack (post-play leave), the score differential, and recent move
context, the position evaluation model predicts:

- **Win/Draw/Loss probabilities** (WLD head) — the primary value signal.
- **Score differential** (ScoreDiff head) — a Gaussian (mean and standard
  deviation) over the clipped final score differential [-400, +400], trained by
  Gaussian negative log-likelihood against the observed differential.
- **Placement masks** (four auxiliary 15×15 heads) — where each player's next
  move will place tiles (`opp_next_placement`, `self_next_placement`) and the
  per-square win conjunctions Pr[places there AND wins] (`opp_win_placement`,
  `self_win_placement`) that feed the sim-evidence loop
  ([sim_residual_feedback.md](sim_residual_feedback.md)).

### Current data pipeline

1. **Self-play**: HastyBot vs. HastyBot games are played in parallel via
   `GameRunner`. Each completed game is serialized to `.slog` format — a
   compact binary log storing initial racks, every move, and every tile draw
   (~24 bytes per turn). Each game records its training-eligible turn region;
   every eligible turn becomes a training position.

2. **Training**: The `DataLoader` streams `.slog` files through an epoch-based
   API, expanding each game into one row per eligible turn. A `BlockDecoder`
   replays the game forward to each turn, materializing the post-move (or
   pre-move) input features on the fly. Diagonal symmetry augmentation (the
   board is invariant under `(r,c) → (c,r)`) is applied stochastically per-row.

3. **Model**: A ResNet trunk (88 spatial input planes × 15×15, plus 992 scalar
   features) branches into the three heads above. Training uses AdamW with a
   rows-clock base learning rate (startup warmup plus manual step-down).

The trainer that consumes this data is the **generational** generate→train loop:
it reuses each generated game across epochs, keeps the run restartable at any
point, and extends toward a game-pool producer, resource-contention management,
and a path to remote workers. It is laid out in
[docs/generational_training.md](generational_training.md), and becomes central in
Phase 2, where more expensive data generation makes reusing each game matter.

### Why start here

The position evaluation model is the natural first target because it directly provides the signal
needed for move selection: given a position and a candidate move, apply the move
and ask the position evaluation model "how good is this resulting state?" The player with the highest
post-move value wins. This is exactly the Q-value definition from the design
doc: $y_i = V(s^{\text{post-move}}_{a_i})$.

---

## Phase 2: Self-Play Diversification

### The problem with HastyBot self-play

HastyBot plays a single deterministic strategy (highest static equity). Training
exclusively on HastyBot games produces a model that learns the value of
*HastyBot-reachable* positions, which is a narrow slice of the full game tree.
The model has no reason to learn accurate values for positions arising from
suboptimal play, unusual openings, or creative tactical sequences — precisely
the positions where a superhuman agent would need the most reliable evaluation.

### Mechanisms

Two are built into the self-play machinery:

- **Move sampling** (`--hasty-temperature`, `--hasty-top-k`,
  `--hasty-temp-min-bag`): instead of playing the top-1 move, sample from the
  top-K moves via a softmax over their static equities. Temperature controls
  exploration breadth, optionally confined to the higher-bag (earlier) part of
  the game.
- **Random openings** (`--random-opening-mean`; see
  [architecture.md](architecture.md), "Random openings"): each game's first
  K ~ round(Exp(mean)) plies are played uniformly at random, driving self-play
  into off-policy states — especially unusual rack leaves — that agent play
  never visits. Positions preceding the last random ply are excluded from the
  training-eligible region so random play never pollutes an outcome target.

One remains future work:

- **Backtracking**: after playing a randomized move to completion, rewind to
  the same decision point, sample a *different* top-K move, and play that
  branch out too. This yields direct comparative signal ("from this exact
  position, move A won and move B lost") and amortizes the prefix's
  move-generation cost across several training samples. It needs a `.slog`
  extension for branch points and a branching mode in `GameRunner`.

---

## Phase 3: Model Validation and Evaluation Machinery

Validating neural network quality in a game-playing context requires multiple
complementary approaches. No single metric tells the whole story:

| Eval | What it tests | Failure mode it catches |
|---|---|---|
| Monotonicity probes | Structural coherence | Nonsensical evaluations (e.g., winning when behind) |
| Calibration testing | Probabilistic accuracy | Model is structurally sound but systematically biased |
| Monte-Carlo comparison | Absolute value accuracy | Model diverges from deep-search ground truth |
| Agent eval | Downstream utility | Model is calibrated but not *useful* for move selection |

This machinery is a prerequisite for Phase 4 (the move set evaluation model): it is how we know when
the position evaluation model is good enough to serve as the oracle for the move set evaluation model's training targets.

### 3.1 Structural analysis: monotonicity probes — built

For a fixed board + leave, batch-evaluate the position evaluation model across a sweep of score
differentials. The win-probability curve should be monotonically increasing in
the active player's score advantage, sigmoid-shaped, and smooth; a model can
achieve decent training loss while violating all three. Implemented in
[py/scribblez/position_eval/eval/](../py/scribblez/position_eval/eval/)
(probe bank sampled from the frozen test split, per-curve structural scores)
and rendered by the dashboard.

### 3.2 Calibration on held-out games — built

Measure whether predicted probabilities are *accurate* — when the model says
70% win probability, does that player win ~70% of the time? Metrics: Brier
score, log-loss, decile calibration curves, and score-diff MAE/sharpness,
computed over the frozen held-out split
([eval/calibration.py](../py/scribblez/position_eval/eval/calibration.py))
and tracked on the dashboard. Calibration matters beyond ordering: comparing
72% vs 68% between two moves is only sound if the numbers are accurate.

### 3.3 Monte-Carlo ground-truth comparison — built

A committed GCG position dataset with offline Monte-Carlo ground truth
(~10k rollouts per position via
[monte_carlo_sim_tool](../engine/apps/monte_carlo_sim_tool.cpp)); every
checkpoint's predictions are compared against it, per-position on the
dashboard's Positions tab and in aggregate on the Loss tab (see
[react_dashboard.md](react_dashboard.md)).

### 3.4 Agent evaluation: HastyBot + position-evaluation top-K

The most direct measure: an agent that ranks HastyBot's top-K candidates by
the position evaluation model's value ([NeuralAgent](../engine/include/scribblez/neural_agent.h),
`--player "--type=neural --model=... --top-k=K"`) is pitted against vanilla
HastyBot via `play_game`. Expected trajectory: an untrained model plays like a
*diluted* HastyBot and loses; as the position evaluation model approaches static-equity ordering the
win rate approaches 50%; a model that captures context static equity ignores
exceeds it. Remaining work: periodic automated match runs during training with
win-rate curves on the dashboard (today matches are run by hand).

---

## Phase 4: Move Set Evaluation Model

### What it is

The position evaluation model can be used to evaluate a candidate move.

This suggests an agent algorithm: enumerate all possible moves, evaluate each
with the position evaluation model, and select the move with the best evaluation.

A natural approach here is to construct an evaluation batch, consisting of
all possible moves, and to do a batch-inference on the GPU. A downside of this
approach is that all moves share the same board state, meaning that each
evaluation within the batch is doing redundant work — a major inefficiency.

This motivates a different model, the move set evaluation model, which accepts the board state and
a list of $N$ candidate moves, and outputs $N$ evaluations, which are
predictions of what the position evaluation model would output if passed the $N$ post-move states
corresponding to those $N$ candidate moves. Such a model would not need to
evaluate the common board $N$ separate times.

### Architecture

The move set evaluation model computes $Q(s, a_i)$ — the predicted value of playing move $a_i$ in
state $s$ — for each of $N$ candidate moves in a single forward pass. The
fundamental architectural challenge is that $N$ varies wildly per position,
from 1 (forced pass) to 10,000+ (with blanks on the rack). The network must
accept a fixed-size board state plus a variable-size set of $N$ moves and
produce $N$ scalar values.

**Board encoder** (fixed-size, computed once). The board state — tile
placements, premium squares, rack, scores, bag composition — is processed by
a CNN trunk into a spatial feature map
$\mathbf{H} \in \mathbb{R}^{225 \times d}$ (one $d$-dimensional vector per
board square) plus a global summary vector $\mathbf{g} \in \mathbb{R}^{d}$
(pooled from the spatial map plus scalar features). This is the same trunk
used by the position evaluation model, with shared weights. The entire point of the move set evaluation model is that this
encoding is computed once per position, not once per candidate move.

**Move encoder** (variable-size). Each legal move $a_i$ is encoded into a
vector $\mathbf{e}_i \in \mathbb{R}^{d}$ capturing the tiles placed (letter
and board position), the resulting leave, and the immediate score:

$$\mathbf{e}_i = W_{\text{move}} \left[ \text{Pool}\bigl(\{\text{Embed}(t_j, r_j, c_j)\}_{j=1}^{k_i}\bigr) \;\|\; \text{Embed}_{\text{leave}}(\ell_i) \;\|\; f_{\text{score}}(\text{score}_i) \right]$$

where $k_i$ is the number of tiles placed, $\ell_i$ is the leave, and $\|$ is
concatenation. The tile embedding encodes both the letter and its board
position, so the network can learn spatial patterns like "this tile lands on a
triple-letter square."

**Cross-attention scoring**. Each move embedding queries into the board's
spatial representation:

$$\mathbf{e}'_i = \text{CrossAttn}(Q{=}\mathbf{e}_i,\; K{=}\mathbf{H},\; V{=}\mathbf{H})$$

This lets each move attend to the relevant board regions — a move near a
triple-word square picks up that spatial context. The attended move embedding
is fused with the global context and projected to a scalar:

$$Q(s, a_i) = \mathbf{w}^\top \text{MLP}([\mathbf{e}'_i \;\|\; \mathbf{g}])$$

All $N$ moves are scored in a single batched cross-attention against the same
key/value set $\mathbf{H}$. One forward pass, regardless of $N$.

**V-head** (side output). The pre-move position value
$V(s) = \text{MLP}(\mathbf{g})$ is a second readout from the same board
encoder — no move input needed. The move set evaluation model thus provides both move-level
evaluations ($Q$) and a position-level summary ($V$) from a single board
encoding.

### The variable-size move set

The number of legal moves varies wildly per position. With no blanks on the
rack, a typical position has a few hundred; with one blank it can reach
2,000–3,000; with two blanks it can exceed 10,000. The bulk of that growth is
blank designations of the same word placement: holding `ABCDE??`, one 7-letter
word on one set of squares may have dozens of legal blank designations that use
the same rack tiles, occupy the same squares, and leave the same tiles behind,
differing only in immediate score (crosswords formed) and board texture (which
hooks they leave for future play).

A natural instinct is to collapse those near-duplicates with an upfront grouping
or filtering stage before scoring. This is **unnecessary**: the move set evaluation model scores the
whole candidate set in a single linear pass (see the inference pipeline below),
so a large `N` is not a compute problem at inference. Two considerations remain,
but neither warrants an upfront filter:

- **The near-duplicates are often not duplicates.** Differing blank letters
  produce different crosswords and different hooks, so the variants are
  frequently genuinely distinct moves — and the move set evaluation model, especially with the post-move
  cross-check features (see
  [lexical_features_for_value.md](lexical_features_for_value.md)), scores them
  differently. Truly strategically-equivalent instantiations are less common
  than the raw count suggests.
- **Diversity of the simulation set.** In principle the top-$K$ chosen for
  Monte Carlo could be crowded by variants of one footprint. A cheap
  footprint-level dedup applied to the *ranked* set at selection time is an
  optional safeguard — grouping applied after scoring to a handful of moves, not
  an upfront reduction of the full set.

An upfront filter that discards candidates before scoring would risk dropping a
strategically critical move (e.g. a modest-scoring play that blocks an
opponent's triple-word access), and it is unclear how to do such filtering
safely without a learned component that reintroduces exactly that risk. Such
filtering is therefore deferred unless profiling shows the move set evaluation model's scoring to be a
bottleneck (see the rollout note below).

### Exchange and pass moves

The $2^7 = 128$ exchange and pass moves share a uniform structure: the board
is unchanged afterward, and the only variation is which tiles the player
keeps. Rather than running them through the move set evaluation model's move encoder and
cross-attention, a **dedicated exchange head** reads the board encoder's
global vector $\mathbf{g}$ and outputs:

- A 7-bit mask identifying the best subset of tiles to keep.
- A value prediction for that exchange.

The exchange head reduces to "which tiles do I want to keep, given the board
context?" Its predicted value competes directly with the move set evaluation model's best word-play
value to determine whether exchanging is the right choice.

### Inference pipeline

The move set evaluation model scores all candidates in a single pass, so the pipeline is short:

1. **GADDAG generates all legal moves** — every legal word placement, exchange,
   and pass.

2. **Encode the board once.** The CNN trunk produces the spatial map
   $\mathbf{H}$ and global vector $\mathbf{g}$ for the position. This is the
   expensive step, and it is paid once regardless of the number of candidates.

3. **Score every word-play candidate in one cross-attention pass.** Each move
   embedding cross-attends into the shared $\mathbf{H}$; moves do **not** attend
   to one another, so the cost is $O(N)$ — linear in the candidate count, with
   the board encode amortized across all of them. Even $N$ in the thousands is a
   single linear pass, which is why no candidate-set reduction is needed before
   scoring. Exchange and pass moves are routed to the exchange head (above)
   rather than the move encoder.

4. **Take the top-$K$ for simulation** (e.g. $K = 5$–10), optionally after a
   cheap footprint-level dedup to keep the set diverse. These candidates enter
   Monte Carlo rollouts or final selection.

**Rollout note.** The $O(N)$ argument assumes the move set evaluation model is evaluated over the full
candidate set roughly once per decision — at the root. If a search or rollout
policy instead invokes the move set evaluation model over all $N$ at every ply, the constant factor of a
large $N$ re-enters and a pre-scoring filter becomes worth revisiting. Keeping
the inner rollout policy cheap, or restricting full evaluation over all $N$ to the
root, avoids this.

### Training the move set evaluation model

The move set evaluation model is trained to approximate the position evaluation model. At a given position, generate legal
moves, apply each one, and evaluate the resulting post-move state with the position evaluation model.
These position-evaluation values become the move set evaluation model's training targets:

$$y_i = V_{\text{PositionEval}}(s^{\text{post-move}}_{a_i})$$

This bootstrapping from the position evaluation model avoids the chicken-and-egg problem of needing
a policy to define the move set evaluation model's targets. Move-set-evaluation training should begin only after
the position evaluation model has reached reasonable quality (see Phase 3 for how to measure this) —
a poorly calibrated position evaluation model would produce noisy targets, compounding errors.

The expensive operation is **position-evaluation-model evaluation** — a full board re-encode per
candidate move — not the move set evaluation model's forward pass. So target generation, not move-set-evaluation
inference, is where compute is spent. Strategies to manage it:

- **Label a subset, mask the loss.** A target need not be computed for every
  legal move. Label an unbiased subset of moves per position with the position evaluation model (e.g.
  a uniform sample) and compute the move set evaluation model's loss only over the labeled moves,
  masking the rest. This decouples target-generation cost from $N$ entirely —
  the number of targets is a free parameter — while keeping the training signal
  unbiased. Over many positions, every frequently occurring move still receives
  signal.
- **Offline target generation.** Pre-compute the position-evaluation targets at every sampled
  position and store them alongside the `.slog` data, amortizing inference cost
  across training epochs.
- **Bootstrapping.** Once the move set evaluation model is partially trained, use it to generate its own
  targets via self-play (the standard RL loop), reducing dependence on position-evaluation-model
  inference.

---

## Phase Summary

| Phase | Goal | Status |
|---|---|---|
| 1 | Train the position evaluation model on HastyBot self-play | Done — model + generational training pipeline |
| 2 | Diversify training data | Move sampling + random openings done; backtracking open |
| 3 | Validate position-evaluation-model quality | Probes, calibration, Monte-Carlo comparison done; automated match eval open |
| 4 | Move set evaluation model | Design ready (board/move encoder, single-pass cross-attention scoring, exchange head); gated on the Phase 3 quality bar |

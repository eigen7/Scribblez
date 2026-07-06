# Scribblez Project Roadmap

## Overview

Scribblez aims to build a superhuman Scrabble AI by training neural networks
that overcome two fundamental weaknesses of existing engines like Macondo:
context-blind leave evaluation and naive opponent rack inference. This document
lays out the planned progression from the current post-move value model through
a full pre-move model, the data-generation strategies that support each stage,
and the evaluation machinery needed to validate progress at every step.

---

## Phase 1: Post-Move Value Model (M_post) — Current

### What it is

M_post evaluates a board state *after* the active player has placed tiles but
*before* drawing replacements from the bag. Given the board, the player's
residual rack (post-play leave), the score differential, and recent move
context, M_post predicts:

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
   (~24 bytes per turn). Each game records how many of its turns are
   training-eligible (the bag-non-empty prefix, or all turns under
   `--sample-endgames`); every eligible turn becomes a training position.

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

M_post is the natural first target because it directly provides the signal
needed for move selection: given a position and a candidate move, apply the move
and ask M_post "how good is this resulting state?" The player with the highest
post-move value wins. This is exactly the Q-value definition from the design
doc: $y_i = V(s^{\text{post-move}}_{a_i})$.

---

## Phase 2: Within-Game Move Randomization and Backtracking

### The problem with HastyBot self-play

HastyBot plays a single deterministic strategy (highest static equity). Training
exclusively on HastyBot games produces a model that learns the value of
*HastyBot-reachable* positions, which is a narrow slice of the full game tree.
The model has no reason to learn accurate values for positions arising from
suboptimal play, unusual openings, or creative tactical sequences — precisely
the positions where a superhuman agent would need the most reliable evaluation.

### The solution: randomized self-play with backtracking

Instead of always playing the top-equity move, we inject controlled randomness
into the self-play games:

1. **Move sampling**: At each turn, instead of playing the top-1 move, sample
   from the top-K moves (e.g., K=5–10) according to a softmax distribution
   over their static equities. The temperature parameter controls exploration
   breadth — high temperature early in training for broad coverage, annealed
   down as the model matures.

2. **Backtracking**: After playing a randomized move and continuing the game to
   completion (to obtain the outcome label), *rewind* the game state to the
   same decision point and sample a *different* move from the top-K. Play that
   branch out to completion as well. This multiplies the number of training
   positions extracted per "stem" game without re-doing the expensive
   move-generation for the prefix.

3. **Multiple samples per game**: With backtracking, a single game prefix can
   generate multiple (position, outcome) pairs from the same decision point but
   with different continuations. This is especially valuable for learning the
   relative values of moves at critical junctures (e.g., whether to open a
   triple-word lane or play safe).

### Why this matters

- **State-space coverage**: Randomization ensures the model sees positions from
  suboptimal and unusual play lines, not just the narrow HastyBot corridor.
  This is critical for generalization — a model that only knows HastyBot
  positions can't reliably evaluate a novel opening or an opponent's mistake.

- **Comparative signal**: Backtracking from the same game state with different
  moves provides direct comparative data — "from this exact position, move A
  led to a win and move B led to a loss." This is a much stronger learning
  signal than independently sampled games that happen to pass through similar
  (but not identical) positions.

- **Efficient data generation**: Backtracking amortizes the cost of
  move-generation and game-prefix replay across multiple training samples.
  The expensive part of self-play is move generation (GADDAG traversal +
  equity evaluation); backtracking reuses all of that work for the prefix.

### Implementation considerations

- The `.slog` format will need extension to support multiple sampled turns per
  game (currently one) and/or multiple continuations from a branch point.
- `GameRunner` will need a branching mode where it checkpoints game state at
  the sampled turn, plays out one continuation, rewinds, and plays out
  alternatives.
- Temperature scheduling (how randomization breadth evolves over training)
  will be a tunable hyperparameter.

---

## Phase 3: Model Validation and Evaluation Machinery

Validating neural network quality in a game-playing context requires multiple
complementary approaches. No single metric tells the whole story — structural
properties, downstream utility, and statistical calibration each reveal
different failure modes. Building this machinery is a prerequisite for Phase 4
(M_pre), because we need to know when M_post is good enough to serve as the
oracle for M_pre's training targets.

### 3.1 Structural Analysis: Monotonicity Probes

**Idea**: For a fixed board + leave, batch-evaluate M_post across a sweep of
score differentials. The resulting win-probability curve should be:

- **Monotonically increasing** in the active player's score advantage.
- **Sigmoid-shaped**: approaching 0 when far behind, approaching 1 when far
  ahead, with a transition region around the break-even point.
- **Smooth**: no abrupt jumps or non-monotonicities.

**Implementation plan**:

1. Curate a bank of representative (board, leave) positions — varying board
   openness, tile distribution, game phase (early/mid/late).
2. For each position, construct a batch of inputs with score differentials
   swept across [-200, +200] in steps of 1.
3. Run inference and extract the WLD head's win probability for each step.
4. Score each curve on:
   - Monotonicity violations (count of sign changes in the finite difference).
   - Sigmoid fit quality (R² of a logistic regression to the curve).
   - Smoothness (total variation of the curve).
5. Aggregate into a single **structural plausibility score** and track it
   across training epochs.

**Why this matters**: A model can achieve decent loss on the training set while
still producing structurally incoherent evaluations (e.g., predicting higher
win probability when behind). Monotonicity probes catch these failures early.

### 3.2 Agent Evaluation: HastyBot + M_post Top-K

**Idea**: Build a simple agent that uses HastyBot's move generator to produce
the top-K candidate moves, then selects the move with the highest M_post
value. Pit this agent against vanilla HastyBot in a series of matches and
track win percentage over training.

**Expected trajectory**:

- **Early training** (random network): The agent is effectively a *diluted*
  HastyBot that picks randomly among the top-K instead of always picking top-1.
  It should lose to HastyBot, and the margin of loss quantifies how much
  signal the model lacks.
- **Mid training**: The agent should approach HastyBot's win rate (50%) as
  M_post learns to approximate static equity ordering.
- **Late training**: The agent should *exceed* HastyBot's win rate, because
  M_post captures contextual information (board state, score differential,
  bag composition) that static equity ignores.

**Implementation plan**:

1. Create a `NeuralTopKAgent` that:
   - Uses `HastyEquity` to generate + rank the top-K moves.
   - Applies each move to the current game state.
   - Runs M_post inference on the K resulting positions.
   - Selects the move with the highest predicted win probability.
2. Integrate into `GameRunner` so it can be pitted against `HastyBotAgent`.
3. Run periodic evaluation matches (e.g., 1000 games every N training epochs).
4. Track win rate, average score differential, and per-game-phase statistics.

**Why this matters**: This is the most direct measure of whether M_post
translates into better play. Structural probes test mathematical properties;
agent eval tests competitive performance.

### 3.3 Calibration: Direct Prediction Accuracy on Held-Out Games

**Idea**: From self-play or from a database of expert games, we know the actual
game outcomes. We can measure whether M_post's predicted probabilities are
*accurate* — when it says 70% win probability, does that player actually win
roughly 70% of the time?

**Metrics**:

- **Brier score**: Mean squared error between predicted win probability and
  actual outcome (0 or 1). Decomposes into calibration + resolution +
  uncertainty components.
- **Log-loss**: Cross-entropy between predicted distribution and actual
  outcome. More heavily penalizes confident wrong predictions.
- **Calibration curve**: Bin predictions into deciles (0–10%, 10–20%, ...,
  90–100%) and plot predicted vs. actual win rate. A perfectly calibrated
  model produces a diagonal line.
- **Score-diff calibration**: For the ScoreDiff head, compare the predicted
  distribution's mean against the actual final score differential. Track the
  mean absolute error and the predicted distribution's sharpness (entropy).

**Implementation plan**:

1. Reserve a held-out set of `.slog` files (never seen during training).
2. At evaluation time, decode each held-out game's sampled position with the
   `BlockDecoder`, run inference, and collect (prediction, actual outcome)
   pairs.
3. Compute Brier score, log-loss, and calibration curves.
4. Track all metrics across training epochs; plot learning curves.

**Why this matters**: A model can have good structural properties
(monotonicity) and still be poorly calibrated (systematically overconfident or
underconfident). A well-calibrated model directly translates to better
decision-making when comparing moves — if M_post says move A gives 72% win
probability and move B gives 68%, you need those numbers to be *accurate*, not
just correctly ordered, to make sound decisions in close situations.

**Relationship to other evals**: The three evaluation approaches are
orthogonal:

| Eval | What it tests | Failure mode it catches |
|---|---|---|
| Monotonicity probes | Structural coherence | Nonsensical evaluations (e.g., winning when behind) |
| Agent eval | Downstream utility | Model is calibrated but not *useful* for move selection |
| Calibration testing | Probabilistic accuracy | Model is structurally sound but systematically biased |

All three should be run continuously and reported on a dashboard.

### 3.4 Evaluation Infrastructure

To support all three eval approaches, we need:

- **Eval harness binary/script**: A single entry point that loads a model
  checkpoint, runs all three eval suites, and produces a structured report
  (JSON + human-readable summary). Integrated into the training loop to run
  automatically every N epochs.
- **Position bank**: A curated set of (board, leave, score) tuples for
  monotonicity probes. Should cover diverse game phases and board
  configurations. Stored as a small `.slog` or custom format.
- **Held-out data split**: A fixed set of `.slog` files excluded from training,
  used exclusively for calibration measurement. Must be established before
  training begins and never contaminated.
- **Match server**: Infrastructure for running HastyBot-vs-NeuralTopK matches
  efficiently (parallelized, with deterministic seeding for reproducibility).
- **Metrics database + dashboard**: Time-series storage of all eval metrics
  keyed by (training epoch, eval type). Enables plotting learning curves
  and detecting regressions.

---

## Phase 4: Pre-Move Model (M_pre)

### What it is

M_post can be used to evaluate a candidate move.

This suggests an agent algorithm: enumerate all possible moves, evaluate each
with M_post, and select the move with the best evaluation.

A natural approach here is to construct an evaluation batch, consisting of
all possible moves, and to do a batch-inference on the GPU. A downside of this
approach is that all moves share the same board state, meaning that each
evaluation within the batch is doing redundant work — a major inefficiency.

This motivates a different model, M_pre, which accepts the board state and
a list of $N$ candidate moves, and outputs $N$ evaluations, which are
predictions of what M_post would output if passed the $N$ post-move states
corresponding to those $N$ candidate moves. Such a model would not need to
evaluate the common board $N$ separate times.

### Architecture

M_pre computes $Q(s, a_i)$ — the predicted value of playing move $a_i$ in
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
used by M_post, with shared weights. The entire point of M_pre is that this
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
encoder — no move input needed. M_pre thus provides both move-level
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
or filtering stage before scoring. This is **unnecessary**: M_pre scores the
whole candidate set in a single linear pass (see the inference pipeline below),
so a large `N` is not a compute problem at inference. Two considerations remain,
but neither warrants an upfront filter:

- **The near-duplicates are often not duplicates.** Differing blank letters
  produce different crosswords and different hooks, so the variants are
  frequently genuinely distinct moves — and M_pre, especially with the post-move
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
filtering is therefore deferred unless profiling shows M_pre scoring to be a
bottleneck (see the rollout note below).

### Exchange and pass moves

The $2^7 = 128$ exchange and pass moves share a uniform structure: the board
is unchanged afterward, and the only variation is which tiles the player
keeps. Rather than running them through M_pre's move encoder and
cross-attention, a **dedicated exchange head** reads the board encoder's
global vector $\mathbf{g}$ and outputs:

- A 7-bit mask identifying the best subset of tiles to keep.
- A value prediction for that exchange.

The exchange head reduces to "which tiles do I want to keep, given the board
context?" Its predicted value competes directly with M_pre's best word-play
value to determine whether exchanging is the right choice.

### Inference pipeline

M_pre scores all candidates in a single pass, so the pipeline is short:

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

**Rollout note.** The $O(N)$ argument assumes M_pre is evaluated over the full
candidate set roughly once per decision — at the root. If a search or rollout
policy instead invokes M_pre over all $N$ at every ply, the constant factor of a
large $N$ re-enters and a pre-scoring filter becomes worth revisiting. Keeping
the inner rollout policy cheap, or restricting full M_pre-over-all-$N$ to the
root, avoids this.

### Training M_pre

M_pre is trained to approximate M_post. At a given position, generate legal
moves, apply each one, and evaluate the resulting post-move state with M_post.
These M_post values become M_pre's training targets:

$$y_i = V_{\text{M\_post}}(s^{\text{post-move}}_{a_i})$$

This bootstrapping from M_post avoids the chicken-and-egg problem of needing
a policy to define M_pre's targets. M_pre training should begin only after
M_post has reached reasonable quality (see Phase 3 for how to measure this) —
a poorly calibrated M_post would produce noisy targets, compounding errors.

The expensive operation is **M_post evaluation** — a full board re-encode per
candidate move — not M_pre's forward pass. So target generation, not M_pre
inference, is where compute is spent. Strategies to manage it:

- **Label a subset, mask the loss.** A target need not be computed for every
  legal move. Label an unbiased subset of moves per position with M_post (e.g.
  a uniform sample) and compute M_pre's loss only over the labeled moves,
  masking the rest. This decouples target-generation cost from $N$ entirely —
  the number of targets is a free parameter — while keeping the training signal
  unbiased. Over many positions, every frequently occurring move still receives
  signal.
- **Offline target generation.** Pre-compute the M_post targets at every sampled
  position and store them alongside the `.slog` data, amortizing inference cost
  across training epochs.
- **Bootstrapping.** Once M_pre is partially trained, use it to generate its own
  targets via self-play (the standard RL loop), reducing dependence on M_post
  inference.

---

## Phase Summary

| Phase | Goal | Key deliverable | Depends on |
|---|---|---|---|
| 1 | Train M_post on HastyBot self-play | Working model + training pipeline | *(done)* |
| 2 | Diversify training data | Randomized self-play with backtracking | Phase 1 |
| 3 | Validate M_post quality | Eval harness (monotonicity + agent + calibration) | Phase 1 |
| 4 | Pre-move model (M_pre) | Board/move encoder, single-pass cross-attention scoring, exchange head | Phase 3 quality bar |

Phases 2 and 3 should be developed in parallel — evaluation machinery is
needed to know when M_post is good enough to support Phase 4.

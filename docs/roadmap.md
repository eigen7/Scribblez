# Scribblez Project Roadmap

Scribblez aims to build a superhuman Scrabble AI by training neural networks
that overcome two fundamental weaknesses of existing engines like Macondo:
context-blind leave evaluation and naive opponent rack inference. This
document lays out the progression from the position evaluation model through
a full move set evaluation model, the data-generation strategies supporting
each stage, and the evaluation machinery that validates progress.

## Phase 1: Position Evaluation Model — built

The position evaluation model evaluates a board state *after* the active
player has placed tiles but *before* drawing replacements. Given the board,
the residual rack (leave), the score differential, and recent move context,
it predicts:

- **Win/Draw/Loss probabilities** (WLD head) — the primary value signal.
- **Score differential** (ScoreDiff head) — a Gaussian over the clipped final
  differential, trained by Gaussian NLL.
- **Placement masks** (four auxiliary 15×15 heads) — where each player's next
  move will place tiles, and the per-square win conjunctions that feed the
  sim-evidence loop ([sim_residual_feedback.md](sim_residual_feedback.md)).

Data pipeline: HastyBot self-play games serialized to `.slog` (initial racks,
moves, draws; each game records its training-eligible turn region); the
`DataLoader` expands each game into one row per eligible turn by replaying
forward, with stochastic diagonal-symmetry augmentation. The model is a
ResNet trunk over spatial planes plus scalar features, trained with AdamW
under the generational generate→train lifecycle
([generational_training.md](generational_training.md)).

Why start here: the position evaluation model directly provides the signal
move selection needs — apply a candidate move and ask "how good is the
resulting state?" This is the Q-value definition from the design doc.

## Phase 2: Self-Play Diversification

HastyBot plays one deterministic strategy, so training exclusively on its
games teaches values only for HastyBot-reachable positions. Mechanisms:

- **Move sampling** (built): sample from the top-K moves via a softmax over
  static equities, with temperature optionally confined to the earlier
  (higher-bag) part of the game.
- **Random openings** (built; see [architecture.md](architecture.md)): each
  game's first `K ~ round(Exp(mean))` plies are played uniformly at random,
  driving self-play into off-policy states. Positions preceding the last
  random ply are excluded from the training-eligible region so random play
  never pollutes an outcome target.
- **Backtracking** (future): rewind to a decision point and play out a
  *different* top-K move, yielding direct comparative signal from the same
  position and amortizing the prefix cost. Needs a `.slog` extension for
  branch points and a branching mode in `GameRunner`.

## Phase 3: Model Validation and Evaluation Machinery

No single metric suffices; the four complementary evals, and the failure mode
each catches:

| Eval | What it tests | Failure mode it catches |
|---|---|---|
| Monotonicity probes | Structural coherence | Nonsensical evaluations |
| Calibration testing | Probabilistic accuracy | Structurally sound but biased |
| Monte-Carlo comparison | Absolute value accuracy | Divergence from deep-search ground truth |
| Agent eval | Downstream utility | Calibrated but not *useful* for move selection |

This machinery is how we know when the position evaluation model is good
enough to serve as the oracle for the move set evaluation model's targets.

- **3.1 Monotonicity probes — built.** For a fixed board + leave, sweep the
  score differential; the win-probability curve should be monotone, sigmoid,
  and smooth. [py/scribblez/position_eval/eval/](../py/scribblez/position_eval/eval/),
  rendered by the dashboard.
- **3.2 Calibration on held-out games — built.** Brier, log-loss, decile
  calibration, score-diff MAE/sharpness over the frozen test split.
  Calibration matters beyond ordering: comparing 72% vs 68% between moves is
  only sound if the numbers are accurate.
- **3.3 Monte-Carlo ground-truth comparison — built.** A committed GCG
  position dataset with offline Monte-Carlo ground truth
  ([monte_carlo_sim_tool](../engine/apps/monte_carlo_sim_tool.cpp)); every
  checkpoint is compared against it on the dashboard.
- **3.4 Agent evaluation — partially built.** An agent that ranks HastyBot's
  top-K candidates by position-evaluation value
  ([NeuralAgent](../engine/include/agent/neural_agent.h)) plays vanilla
  HastyBot via `play_game`. Win rate approaching then exceeding 50% tracks
  the model capturing context that static equity ignores. Remaining:
  periodic automated match runs during training (today matches are by hand).

## Phase 4: Move Set Evaluation Model

Evaluating every legal move with the position evaluation model means a full
board re-encode per candidate — redundant, since all candidates share the
board. The move set evaluation model accepts the board plus all `N` candidate
moves and predicts, in one pass, what the position evaluation model would say
about each post-move state.

### Architecture

- **Board encoder** (computed once per position): the position evaluation
  model's trunk, shared weights, producing a spatial map `H` (one vector per
  square) and a pooled global vector `g`.
- **Move encoder**: each move becomes a vector from its placed tiles (letter
  + board position, so spatial patterns like "lands on a triple-letter" are
  learnable), its leave, and its score.
- **Cross-attention scoring**: each move embedding queries into `H` (moves do
  not attend to each other), is fused with `g`, and projects to a scalar
  `Q(s, aᵢ)`. All `N` moves score in one batched pass — `O(N)`, with the
  board encode amortized.
- **V-head**: a pre-move position value read from `g` alone.
- **Exchange head**: the 128 exchange/pass moves leave the board unchanged,
  so a dedicated head reads `g` and outputs the best keep-mask and its value,
  competing directly with the best word-play value.

### No upfront candidate filtering

`N` ranges from 1 to 10,000+ (blanks). Collapsing near-duplicate blank
designations before scoring is unnecessary and risky: the single linear pass
makes large `N` a non-problem at inference; differing blank letters produce
different crosswords and hooks, so variants are often genuinely distinct; and
an upfront filter risks dropping a strategically critical move (the modest
play that blocks a triple-word lane). The one legitimate concern — top-K
diversity for simulation — is handled *after* scoring, by a cheap
footprint-level dedup of the ranked handful. Caveat: the `O(N)` argument
assumes full-set evaluation happens roughly once per decision, at the root.
A neural *rollout* policy instead scores only the top-`k` moves by hasty
equity (small fixed `k`, ~16). Nothing is cached across plies, so the
per-ply cost is one trunk encode plus `k` move scorings: the pruning caps
the move-encoding cost that a 20,000-move two-blank position would
otherwise impose, and the fixed `k` gives static tensor shapes for batching
plies across concurrent rollouts. Context-blind pruning is second-order
inside rollouts — both simulated players share the policy, so residual bias
largely cancels in candidate comparisons — whereas at the root it would
drop exactly the moves the model exists to find.

### Training

Distillation from the position evaluation model: targets are its evaluations
of candidates' post-move states, begun only once Phase 3 says the oracle is
good enough. The expensive step is target generation (a full re-encode per
labeled candidate), not move-set-evaluation inference, so:

- **Label a subset, mask the loss** — an unbiased sample of moves per
  position gets targets; the loss is masked to them. Target count becomes a
  free parameter, decoupled from `N`, and the signal stays unbiased.
- **Offline target generation** — precompute and store targets alongside the
  `.slog` data, amortizing across epochs.
- **Bootstrapping** — a partially trained move set evaluation model can later
  generate its own targets via self-play.

## Phase Summary

| Phase | Goal | Status |
|---|---|---|
| 1 | Train the position evaluation model on HastyBot self-play | Done — model + generational training pipeline |
| 2 | Diversify training data | Move sampling + random openings done; backtracking open |
| 3 | Validate position-evaluation-model quality | Probes, calibration, Monte-Carlo comparison done; automated match eval open |
| 4 | Move set evaluation model | Design ready; gated on the Phase 3 quality bar |

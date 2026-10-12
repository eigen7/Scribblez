# DeepBot: information-set search with regression transfer

**Status: proposed 2026-10-12; plan-reviewed the same day ([review
record](#review-record)); not built, and not yet placed in the
[roadmap](../roadmap.md).** It is an engineered alternative to
[supreme_bot.md](supreme_bot.md), the current direction. The two pursue the
same flows and differ in what is learned ([Where this sits](#where-this-sits)).
How DeepBot relates to SupremeBot is an open human call.

**Goal.** A search that does what a strong player does at
[Richards–Johnson](../analysis/richards-johnson-exchange/README.md):

- It reasons about a reply *kind* across many possible opponent racks at once,
  such as "Mike sets up a second S lane with YE\_\_".
- It carries what it learned about one candidate over to the others.
- It revisits a candidate when a later finding changes how its earlier search
  should be read.

**Decision.** Search a tree whose decision nodes are **information sets**: a
node is keyed by what its mover can observe. Hidden tiles are resolved at
explicit **chance nodes**, sampled with seeds shared by every candidate.

- **Backup:** Gumbel MuZero's completed values and improved policy, with a
  per-action posterior in place of visit counts. Values are recomputed from
  current estimates, never accumulated.
- **Transfer:** a **per-turn Bayesian linear regression** that predicts how
  far a search result departs from the network's prior, from features the
  network computes for every move. It sets the estimate and the uncertainty
  of every action not yet searched. Anything a search has reached overrides
  it.
- **Training:** the features are meta-learned offline (a closed-form ridge
  over logged search residuals), so that this regression works.
- **Build order:** start shallow. Ply one is searched as a table of
  candidates × opponent racks × replies, and recursion is added only if
  measurements call for it.

## Purpose

[simulation_information_flow.md](../simulation_information_flow.md) walks an
idealized agent through Richards–Johnson and names what it does:

- **Sideways valuation:** one ETS rack's result prices a quarter of the rack
  space.
- **Sideways steering:** ALLEE's probes change which reply Mike tries after
  other candidates.
- **Revision:** later probes re-price earlier ones.

It also shows why plain node statistics cannot carry those flows in Scrabble:
below the first draw, most nodes are visited once.

DeepBot gets all three flows from two pieces:

1. **A sound backup.** Each node's value is recomputed from its children's
   current estimates. Nothing is a running sum of past probes.
2. **One learned similarity, fitted in closed form.** It decides what a search
   result says about actions no probe reached.

The learned part is small: a feature map trained offline, plus a regression
with a few dozen to a few hundred weights fitted per decision. Aggregation,
weighting by evidence and shrinkage are classical statistics, not something a
network has to learn.

## Where this sits

- **[supreme_bot.md](supreme_bot.md)** learns every flow in one network that
  reads a context of every probe step. DeepBot fixes the backup rule and the
  form of the transfer (linear in learned features).
  - *What DeepBot gains:* with transfer off it is an ordinary convergent
    search, which addresses SupremeBot's "no free convergence" risk. Its
    learning problem is far smaller than the reader's, which matters because
    M1a's reader found no transfer
    ([supreme_bot_m1a.md](supreme_bot_m1a.md)).
  - *What it gives up:* any transfer that is not linear in the features, and
    learned draws.
- **[sim_residual_feedback.md](sim_residual_feedback.md)** supplies two
  principles DeepBot keeps:
  - residuals are measured against the prior's prediction for the *same*
    state;
  - candidates are allocated by expected gain over the best so far.

  It also supplies the root shortlist. Its two shortcomings motivate this
  plan:
  - hasty rollouts never play the reply that matters;
  - a key move may not be illuminated by the sims of its siblings.

  DeepBot searches the opponent's replies instead of rolling them out, and
  gives every candidate value from its own probes.
- **[rack_conditional_evidence.md](rack_conditional_evidence.md)** wanted the
  same per-rack transfer through learned evidence tokens. DeepBot's chance
  samples play the part of its rack indices, and its regression stands in for
  the fusion stage.
- **Information-set search** is the standard answer to strategy fusion in
  imperfect-information games, as in ISMCTS (Cowling, Powley & Whitehouse,
  2012). DeepBot keeps the structure and replaces the visit-mean backup.

In the flow vocabulary, DeepBot's profile is sideways in every column except
chance steering, plus revision:

| root steering | steering below root | chance steering | action valuation | chance valuation | revision |
|---|---|---|---|---|---|
| sideways | sideways | fixed (seeded samples) | sideways | sideways | yes |

### Reuse

| Component | Role in DeepBot |
|---|---|
| `MoveProposalService` / `MoveSetEvalService` ([agent/](../../engine/include/agent/), [nn/](../../engine/include/nn/)) | root shortlist; per-node priors `v̂(a)`; the hidden state `h(a)` after a new aux output ([Features](#features-and-their-training)) |
| `CrossCheckEncoder` inputs ([training/cross_check_delta.h](../../engine/include/training/cross_check_delta.h)) | the mechanical features |
| `RackInferrer` ([belief/rack_inference.h](../../engine/include/belief/rack_inference.h)) | the root chance node's posterior over the opponent's rack (standard Scrabble) |
| SimRunner's CRN seeding ([sim/sim_runner.h](../../engine/include/sim/sim_runner.h)) | the pattern for draw seeds shared across candidates; a world-seeded draw source replaces `Bag`'s RNG draw |
| `EndgameSolver` ([endgame/endgame_solver.h](../../engine/include/endgame/endgame_solver.h)) | leaves once the bag is empty |
| M1a near-endgame corpora (`m1a-endgame-*` tags) | the first fit of the prior's error variance `τ²` |
| BestBot, `mset-sim` | match baselines, not reused code |

`PreEndgameSolver` is **not** used. It aggregates over every bag draw itself
and ranks plays pessimistically. DeepBot's chance nodes take over that job, with
sampled draws.

## The tree

**Node kinds.**

- **Decision nodes** are keyed by the mover's information set: the public
  history plus the mover's own rack. Under face-up leaves, the opponent's
  public leave is part of the public history.
- **Chance nodes** resolve tiles hidden from the next mover:
  - at the root, the opponent's rack, drawn from the posterior (under face-up
    leaves, only their unseen draws);
  - after each move, the mover's refill.

Two different hidden states that lead to the same observations for the next
mover reach the **same** decision node. After ALLEE, every sampled Mike rack
that can play YEET reaches one node for Nigel, "ALLEE, YEET, Nigel's rack R".
That merges chance outcomes wherever the observations are identical. Transfer
then covers outcomes that are similar but not identical.

**Deferred draws.** A refill is hidden from the opponent, so the opponent's
decision must sit *above* the chance node that resolves it. Following
[supreme_bot.md](supreme_bot.md#face-up-leaves-deferred-draws), a player's
draw is sampled when that player next decides. In standard Scrabble this holds
for refills, because no one else observes them before the drawer's next turn.
It needs a pending-draw state that the engine does not have yet, with
SupremeBot's two guards:

- **Exchanges:** legality uses the bag count net of pending draws, and an
  exchange resolves pending draws before its tiles return to the bag.
- **The bag emptying:** pending draws resolve before the bag would empty and
  before the solver takes over.

A **paired-world test** checks it. Change a hidden draw while holding the
deciding player's observations fixed, and the decision must not change.

**Chance samples.** A chance node's outcomes are drawn from a per-turn seed
keyed by `(sample index, draw ordinal)`, in the manner of SimRunner. Every
candidate therefore meets the same draws wherever its draws line up, and luck
cancels in comparisons. Each chance node holds the outcomes sampled so far,
with probability weights `ω`. These are importance weights where a proposal
differs from the true distribution; the first version samples the true
distribution, so `ω = 1`. A probe walks a `Game` copy with a seeded draw
source in place of `Bag`'s RNG draw.

**Expansion.** Expanding a decision node costs:

- one move generation;
- per-move input encoding, including the cross-check deltas;
- one move set evaluation pass over every legal move `a`.

The pass returns a value prior `v̂(a)` (win equity, from the mover's
information set) and a feature vector `φ(a)`. Expansion rolls nothing out.

**Leaves.** Once both racks are known and the bag is empty, a node's value
comes from the endgame solver at its configured budget (default 400 nodes).
A verdict maps to `(Q, V)`:

- **Proven:** the class gives 1, 0.5 or 0, with near-zero variance.
- **Unproven:** the spread maps to win equity through a logistic fitted
  against high-budget solves. The variance comes from that fit's residuals.

At Richards–Johnson after a five-tile play, two tiles are left in the bag.
Mike's two-tile and seven-tile replies empty it. A one-tile reply leaves one
tile, a chance node, and then endgames.

## Values

Each action `a` at a decision node has an estimate `Q(a)` and a variance
`V(a)`, from the mover's point of view.

| action is | `Q(a)` | `V(a)` |
|---|---|---|
| searched | the value of the chance node below it, flipped to this mover if the next decider is the other player | propagated |
| a solver leaf | the mapped verdict | from the verdict mapping |
| unsearched | `v̂(a) + b(a) + μᵀφ(a)`: prior, plus intercept, plus transferred correction | `φ(a)ᵀΣφ(a) + τ²(a)` |

`b(a)` is a per-band intercept, and `μ, Σ` are the regression posterior
([Transfer](#transfer)). `τ²(a)` is the prior's own error variance
([Uncertainty](#uncertainty)).

**Chance nodes.** A chance node's value is the `ω`-weighted mean of its
sampled children. Its variance is the sampling variance of that mean (spread
over the samples divided by the effective sample size), plus the weighted
children's own variances. Under-sampled chance nodes are uncertain, and
their uncertainty propagates.

A decision node below a chance node can be reached from many hidden states,
for example Nigel's node after YEET. Each upstream path reaches it with
weight `ω × π′(YEET)` from its own Mike node, and those weights set the
posterior over hidden states inside that node's chance children. That is
Bayesian updating on the opponent's move under the modeled policy.

**The improved policy.** At each decision node, `π′(a)` is the probability
that `a` is the mover's best action, under independent Gaussians
`N(Q(a), V(a))`. It is computed with a fixed per-node seed, so a recompute
changes `π′` only when the estimates change. It sharpens toward the best
action as variances shrink, and stays spread over every action that could
still be best while they are wide.

**Decision node value:** `Σ_a π′(a) Q(a)`, with variance
`Σ_a π′(a)² V(a) + Σ_a π′(a)(Q(a) − value)²`. This takes the place of both
the visit mean and a hard max:

- A **mean over visits** dilutes the value with exploratory children.
- A **max over estimates** selects the largest error (the optimizer's curse),
  and the estimates here carry model bias that does not average away.

This is Gumbel MuZero's completed-Q construction (Danihelka et al., 2022),
with one change: Gumbel's `σ` scales with the node's maximum visit count, so
visits spent on one action lend confidence to the guessed values of the
others. Per-action variances do not. Gumbel's `σ` form stays as an ablation
arm.

**Recompute, never accumulate.** Every value is a function of the current
estimates below it. When the regression changes, every unsearched estimate
changes, and the expanded tree is recomputed bottom-up. Searches made before a
finding keep their values, which are still right about what they reached.
`π′` decides their weight anew.

## Transfer

**The model.** For each searched action, the **residual** is
`δ(a) = Q_raw(a) − v̂(a)`. It measures how far search moved the value from
the prior, both as expectations over the same information set. Model it as
`δ(a) = b(a) + wᵀφ(a) + ε`:

- `b(a)` is an intercept per band of (side to move, depth, bag count), fitted
  without shrinkage. It absorbs systematic gaps, such as the difference
  between a one-ply lookahead and the network's own reading of the parent.
  The transfer weights then carry only what varies between moves.
- `w ~ N(0, λ⁻¹ I)` is shared by every node and candidate in the turn. `λ` is
  set by marginal likelihood.
- `ε ~ N(0, V_raw(a) + τ²(a))`.

Refitting costs `O(N d²)` for `N` residuals and `d` features.

**Raw and corrected tracks.** With transfer on, each node keeps two values:

- **Raw:** unsearched actions sit at `v̂ + b`. The fit uses this track only.
- **Corrected:** includes the transfer. `π′`, allocation and the pick use this
  one.

Fitting on corrected values would let the regression confirm its own output.
With transfer off, the two tracks coincide.

**Why the features never see a hidden rack.** `v̂` and `φ` at Nigel's nodes
cannot see Mike's rack, because the move set model's input lumps the bag and
the opponent's rack into one unseen pool
([game_state_encoder.h](../../engine/include/encoding/game_state_encoder.h)).
Under information-set nodes that is correct, not a defect. Nigel's decision
should not depend on Mike's rack, and his node's searched value is already an
average over Mike's racks. Adding the hidden rack to `φ` would let the
regression reintroduce clairvoyance.

**The Richards case.** Under ALLEE, Mike's node for one ETS rack searches
YEET and finds it a disaster for Nigel, a large residual. The posterior mean
moves along `φ(YEET)`. Mike's nodes for other racks hold YETI or YEAN with
nearby feature vectors, so their estimates rise, their variances fall, and
`π′` shifts toward them. YENS has the same footprint but no S hook, lies
elsewhere in feature space, and barely moves. Under VELATE, which also plays
five tiles, Mike's replies share those directions and inherit the correction.
Under the exchange, the bag-count feature differs, so they inherit less.

**Transfer yields to search.** Transfer only sets the estimates of actions
nothing has reached. In the limit where everything is searched, the
regression has no effect. A wrong transfer costs efficiency, not
correctness.

## Search control

A **probe** picks a root candidate, walks down choosing actions at decision
nodes and seeded samples at chance nodes, expands one node, and backs up
along its path. Refits and full recomputes run every `B` probes.

- **Concurrency.** Workers descend with virtual loss: in-flight probes count
  toward `N`. The recompute is double-buffered, so readers see a consistent
  snapshot.
- **At decision nodes below the root,** choose the action that maximizes
  `π′(a) − N(a) / (1 + Σ_b N(b))`. This is Gumbel MuZero's interior rule,
  which pulls visit shares toward `π′`. At the opponent's nodes, add a
  per-node Gumbel perturbation of `π′`, so that different racks try different
  low-prior replies and the regression spreads what one finds. The paper's
  regret guarantee covers only the root, so the perturbation is an E2 arm, on
  and off.
- **At chance nodes,** take the next unsampled seed while the node has fewer
  samples than its share of the budget. Otherwise revisit the child with the
  largest weighted variance.
- **At the root,** allocate by expected gain over the incumbent,
  `E[max(0, q(m) − q(best))]`, using paired variances over the shared draws.
- **The candidate set** is the move proposal shortlist plus a **timing
  floor**: the best candidate for each number of tiles played, 0 through 7,
  plus the top few exchanges. Transfer between candidates is weakest along
  the bag-count axis, so recall along it cannot rely on transfer.
- **The pick** is the highest corrected `q(m)` among candidates that have
  reached a minimum share of the budget.

## Features and their training

**The features.** `φ(a)` concatenates:

- `g(h(a))`, a projection of the move set model's per-move hidden state
  `h(a)`: the post-ReLU activation inside its value head, width `C` (192 in
  current models). No serving graph emits it today. Emitting it needs:
  - an aux output on `MoveSetEvaluationSpec`
    ([nn/model_specs.h](../../engine/include/nn/model_specs.h));
  - a re-export;
  - parity coverage.

  The output is read back in BF16. E0 reads it from PyTorch directly.
- **Mechanical consequence features:**
  - the cross-check change summary: hooks created and destroyed, S hooks in
    particular;
  - tiles played, the bag count after the move, the score;
  - the side to move, and the depth.

The mechanical features hedge against a blind spot in `h` itself. If `h`
does not encode "S hook at O8, bag empty, mover holds an S", no `w` can
express the correction.

**Width.** With an identity `g`, `d ≈ 200`. E0 compares that against a fixed
PCA to 64 dimensions, fitted on a corpus. The search uses whichever
generalizes better at equal cost.

**Training `g`** is meta-learning with a closed-form base learner (R2D2,
Bertinetto et al., 2019):

1. **The corpus is the search log** ([below](#the-search-log)).
2. **An episode is one logged turn.** Its residuals split into support and
   query, stratified by root chance sample, so the query tests transfer
   across racks.
3. **The base learner** solves the ridge, with intercepts, on the support set
   in closed form. It is differentiable in `g`.
4. **The loss** is the Gaussian NLL of the query predictions, so `Σ` is
   calibrated as well as `μ`.
5. **The backbone** stays frozen, so `v̂` cannot drift while `g` learns.

**Selection bias.** What gets searched is chosen by `π′`, so a log holds the
prior's favorites, while the regression's predictions matter most for the
actions the prior ranks low. Logging therefore forces exploration: a fixed
fraction of probes at the opponent's nodes take uniformly random or
reverse-prior actions. The query error is reported stratified by prior rank.
The headline is the error on the bottom ranks.

**Controls**, with M1a's discipline:

- *shuffled:* residuals permuted among a turn's actions;
- *mechanical only:* no `g`;
- *frozen `h`:* identity `g`.

Transfer is the query error with real residuals minus the error with
shuffled ones.

### The search log

A new sidecar, specified in its own PR:

- **Snapshot at the end of the turn.** `Q_raw` changes while the subtree
  grows, so only end-of-turn values are logged, with each action's search
  effort.
- **Moves and seeds, not features.** Following the replay-reconstruction
  invariant ([architecture.md](../architecture.md)), the trainer replays a
  turn from its moves and seeds and recomputes `h` and the mechanical
  features. A replay test checks the reproduction.
- **A backbone fingerprint per file,** so training rejects logs whose `v̂`
  came from a different model.

## Uncertainty

Three error models set the variances. Before any search exists, their first
fits come from existing data:

- **`τ²`, the prior's error.** A per-band constant (bag count, side to move).
  The first fit uses the M1a near-endgame corpora, whose teacher priors sit
  next to solved-endgame labels. Later fits use search logs. A head that
  predicts `|δ|` may come later.
- **The solver's error** on unproven verdicts, by unseen count at the
  chosen budget, measured against high-budget solves.
- **The leaf model's error,** if a depth cap evaluates nodes with the teacher.

These variances are the control surface. An overconfident `τ²` collapses `π′`
onto the prior's favorite, which is the very blind spot this plan targets.
Calibration (empirical coverage on held-out residuals) is checked at every
milestone.

## Improving the priors

Transfer corrects what is specific to a position. Patterns that recur belong
in the weights. The priors lack setup patterns because their training data is
HastyBot-style play ([sim_labeled_candidates.md](sim_labeled_candidates.md)).

- **Student values** distill from raw search values `Q_raw` of well-searched
  actions, never from corrected ones, which would contain the transfer the
  target is meant to replace.
- **The teacher** retrains on DeepBot self-play outcomes once DeepBot plays
  games, under the [generational lifecycle](../generational_training.md).
- **`g`** retrains on each generation's logs. As the priors improve,
  residuals shrink, and transfer covers a narrower remainder.

## Cost

Expansion dominates. Each expansion runs the trunk once for its position,
plus per-move encoding over hundreds to thousands of moves. The move set
service caps a row group at 4,096 moves. Batching across nodes follows the
`BatchingPositionEvalService` pattern. Solver leaves add one solve each, and
near the endgame almost every expansion at ply two is a solve.

| Item | Size |
|---|---|
| Ply-one expansions | `C·K` for `C` candidates and `K` root samples: 20 × 200 = 4,000 if every pair is expanded |
| Recompute | about `E · actions · d` multiply-adds: 4,000 × 500 × 200 = 400M (130M at `d` = 64) |
| `φ` storage | 4,000 × 500 × 200 × 2 bytes ≈ 0.8 GB in BF16; at `d` = 64, about 0.26 GB |

Lazy expansion and allocation keep `E` far below the full layer. The
numbers that decide whether the design is affordable are measured in E0.5:
expansions per second (mid-game and near-endgame) and solves per second.

## Build order

Each step names its PRs and its measurement. The shallow search (E1) comes
before recursion, and transfer is measured on/off on the same substrate.

**Prerequisite.** A named move set checkpoint that meets a maturity bar on
`eval_win_mae`, from the revised model now training after #346. E0 records
which checkpoint it used.

1. **E0, the offline transfer test.** Python, with no search code.
   - *Positions:*
     - Richards–Johnson after ALLEE and after VELATE;
     - pos-09 and egotize;
     - the ACETA setup;
     - a sample of the [blind_spots](../blind_spots.md) tag.
   - *Hidden information:* Mike's 7-tile rack is sampled from the unseen
     pool, about 200 samples, uniform, with `RackInferrer` as a second arm.
     This is not face-up for Mike. Under face-up, WOE would make six of his
     tiles public and leave almost no rack variation to test. The
     face-up-trained checkpoint evaluates Mike's nodes, with Nigel's leave as
     the public opponent leave. That inconsistency is acceptable for a test
     of the regression, not of a decision.
   - *Truth:*
     - For replies that empty the bag: endgame solves at a high budget,
       using proven verdicts only. The fraction proven is reported.
     - For one-tile replies: enumerate the last tile and solve each case.
     - For mid-game positions: a fixed two-ply expansion with teacher
       leaves, reported as a separate regime.
   - *Fit:* the ridge with intercepts, `λ` by marginal likelihood, identity
     `g` against PCA-64, with and without the mechanical features. It is fit
     on half the root samples and predicts the other half, and also fit
     after YE\_\_ replies appear in only one or two samples.
   - *Pre-registered gates* (numbers to be confirmed, see [Open
     questions](#open-questions)):
     - *Pass, per regime:* held-out R² of the residual at least 0.15 above
       the shuffled control.
     - *Pass, Richards specifically:* after two samples, YE\_\_ S-hook
       replies in held-out samples are predicted within half of their mean
       absolute residual, and YENS-like replies stay within `2τ` of zero.
     - *Kill:* R² within 0.05 of shuffled in every regime. Transfer of this
       form is then dead. A pass in the near-endgame regime alone gates only
       near-endgame work.
2. **E0b, the residual bias table.** It reuses E0's expansion code. In random
   mid-game positions, expand one and two plies over sampled hidden states,
   and tabulate the mean `δ` by side, depth and bag. This sets the intercept
   bands, and shows how large the systematic part is before any search
   relies on it.
3. **E0.5, the error models and throughput.**
   - Fit `τ²` from the M1a corpora.
   - Fit the solver verdict mapping and its error at the chosen budget.
   - Microbenchmark a full expansion (movegen, encoding, the move set pass,
     reading back `h`) and solver calls per second.
   - *Gate for E1:* calibration of both error models, and an expansion rate
     that fits a match-time budget.
4. **E1, the shallow search.** Separate PRs:
   1. Export `h` as an aux output (spec, re-export, parity test).
   2. The seeded draw source, the pending-draw state with its guards, and
      the paired-world test.
   3. Information-set nodes, chance nodes, expansion and solver leaves,
      tested on toy trees with known values.
   4. The backup (`π′`, chance aggregation, recompute) and the search
      control, depth-capped at the opponent's reply, with teacher or solver
      values below. Includes the timing floor and the root allocation.
   5. Transfer (the regression, intercepts, both tracks) and the search log
      with forced exploration.

   *Measure:* transfer on against off at equal time. That is discovery cost
   (probes until YE\_\_ replies carry their solved share of `π′`), the
   Richards pick, and the known positions. Matches against BestBot and
   `mset-sim` come second; a weak substrate must not mask a useful transfer.
5. **E2, depth.** Recursion below the opponent's reply. Taken only if E1's
   failures trace to missing depth. Includes the perturbation arm and the
   Gumbel-`σ` ablation.
6. **E3, the learned projection.** Train `g` on E1/E2 logs with the controls
   and the rank stratification. Repeat E1's measurements.
7. **E4, the generational loop.** Student distillation from raw search
   values, then DeepBot self-play for the teacher.
8. **E5, standard Scrabble.** A standard-Scrabble prior, `RackInferrer`
   extended to more than the last move, and match play.

## Risks

- **Linear transfer may be too weak.** Unrelated errors in one turn must lie
  in separable feature directions. If they interfere, the fallback is a
  Gaussian process with a learned kernel, at `O(N³)` per fit.
- **The variances are the control surface.** This is covered under
  [Uncertainty](#uncertainty), and calibration is gated at E0.5.
- **Expansion throughput.** Full move generation plus a move set pass per
  node costs far more than a hasty rollout ply. If E0.5's rate is too low,
  the budget goes to ply one, where discovery matters most.
- **Chance coverage.** A reply that appears in no sampled rack is never
  searched. At best it is priced by transfer. `K` and posterior-weighted
  sampling set that coverage.
- **The modeled opponent.** `π′` models the opponent as choosing their
  believed-best action. The posterior over hidden states at our later nodes
  inherits that model.

## Open questions

- **Relation to SupremeBot (human call).** DeepBot as a parallel track
  against M1a's reader, a replacement, or the evaluation baseline for M1a's
  "beats hand-built transfer" bar. The roadmap is unchanged until this is
  decided.
- **The E0 gate numbers (human call).** 0.15 R² over shuffled, half the mean
  absolute residual, and 0.05 for the kill are proposals. They should be
  settled before E0 runs.
- **A policy prior inside `π′`,** to model a realistic rather than an optimal
  opponent.
- **Residual space.** Win equity or logit, where errors near certain wins and
  losses are comparable.
- **One `w` per turn or per candidate group.** Sharing across candidates is
  what moves ALLEE's lesson to VELATE, but it may blur corrections that differ
  between kinds of candidate.

## Review record

Plan review, 2026-10-12: four independent panelists. Hidden complexity, scope
and integration were Claude subagents; the rival designer was Codex, with
repository access. Every critique, with its resolution:

| Critique | Resolution |
|---|---|
| **Blocking.** E0 never fixes face-up or hidden. Under face-up, WOE makes six of Mike's tiles public, so "200 worlds" collapse to a few racks. | Revised. E0 samples Mike's whole rack (uniform, plus a `RackInferrer` arm), states the checkpoint's input arm, and says why the inconsistency is acceptable for E0. |
| "The bag is empty after any reply" is false: one-tile replies leave a tile. E0's truth depends on solver settings. | Revised. One-tile replies enumerate the last tile. Truth uses proven verdicts at a high budget, with the fraction proven reported. |
| Separate trees per world are clairvoyant: a player optimizes against draws they cannot see (Codex's two-draw counterexample). Residuals would learn and transfer that bias. Raised by two panelists. | Revised: decision nodes are information sets, hidden tiles resolve at chance nodes below the decisions that must not see them (deferred draws), and seeded samples keep CRN pairing. The design changed most here. |
| Test a shallow reply table before building a recursive tree; transfer may be rejected because a new substrate is weak. | Revised. E1 is depth-capped at the opponent's reply, recursion (E2) is conditional, and E1's headline is transfer on against off on the same substrate. |
| The move set model cannot see the world's opponent rack, so world-specific effects all land in `δ`. | Rejected as a defect, with reasoning: under information-set nodes the prior *should* not see it. A hidden rack in `φ` would reintroduce clairvoyance. Documented under [Transfer](#transfer). |
| `h(a)` is not exported by any serving graph, and identity `g` makes `d ≈ 200`, not 64. Raised by two panelists. | Revised. An export PR in E1, BF16 readback, PCA-64 as an E0 arm, cost redone at `d` = 200. |
| `EndgameVerdict` has no equity or variance; `PreEndgameSolver` aggregates over draws, so it is not a per-world leaf. Its cost is missing. Raised by two panelists. | Revised. Verdict-to-`(Q, V)` mapping fitted in E0.5; `PreEndgameSolver` dropped (chance nodes do its job); solver cost in Cost and E0.5. |
| Shallow residuals carry systematic bias (soft-max lookahead against the network's own reading); E0 cannot see it. | Revised. Unshrunk per-band intercepts in the regression; E0b tabulates the bias first. |
| E1 needs `τ²` and the solver error model, which only E1's corpus would fit. | Revised. E0.5 fits both from existing data (M1a corpora, high-budget solves), with a calibration gate. |
| Face-up deferred draws have no engine support and change the core data structure. | Revised. A pending-draw state with its guards and the paired-world test is its own E1 PR. Under the information-set design, deferral is needed in standard Scrabble too. |
| The meta-learning corpus covers only searched actions; deployment predicts unsearched, low-prior ones. | Revised. Forced exploration in logging, and query error stratified by prior rank, with the bottom ranks as the headline. |
| E0's pass and kill criteria are qualitative, and the hyperparameters are unset. | Revised. Pre-registered metrics and `λ` by marginal likelihood. The numbers are an **open human call**. |
| E0 depends on an mset checkpoint that is still training. | Revised. Named prerequisite with a maturity bar. |
| Expansion cost omits per-move encoding; trunk cost per position dominates; the 4,096-row cap. Raised by two panelists. | Revised. Cost rewritten; E0.5 microbenchmarks a full expansion. |
| Parallel probes and the global recompute have no concurrency rule; `π′` sampling noise flips argmaxes. | Revised. Virtual loss, a double-buffered recompute, `π′` draws with a fixed per-node seed. |
| The search log is a new sidecar, not one line. | Revised. Its own section and PR: end-of-turn snapshot, seed replay test, backbone fingerprint. |
| Worlds need an ordered bag; `Bag` is an RNG container. | Revised. A seeded draw source keyed by (sample, draw ordinal), after SimRunner. |
| Overlap with existing agents and plans is acknowledged only at the level of documents. | Revised: a [reuse](#reuse) table. Placement against SupremeBot is an **open human call**. |
| E1 is too large for one PR. | Revised. E1 is five PRs, each with its own test. |
| Defer the meta-learning spec and the generational loop until E2 shows transfer. | Rejected. The request was a plan covering training as well as decisions. Both stay, gated in the build order (E3 and E4 follow measured transfer). |
| Drop the information-set machinery and importance weights from v1. | Rejected. The information-set revision makes them core: `ω` carries the opponent-policy weighting at our later nodes. |
| The perturbation, the Gumbel-`σ` ablation and the GP fallback are speculative extras. | Partly revised. The perturbation and the ablation move to E2 as arms; the GP stays only as a risk line. |
| The dual tracks and the global recompute are heavier than transfer-off E1 needs. | Revised. The tracks coincide with transfer off; the second track arrives with the transfer PR. |
| E0 tests mostly the bag-empty regime. | Revised. A mid-game regime with its own gate, and the kill criterion scoped by regime. |
| Land the timing floor alone in the existing pipeline first. | Rejected. BestBot already sims the top 80, which includes EELLT, and still ranks it 12th to 40th, so recall alone does not fix Richards. The floor matters only with better valuation. |

# DeepBot: shared-world search with regression transfer

**Status: proposed 2026-10-12; not built, not plan-reviewed, and not yet
placed in the [roadmap](../roadmap.md).** It is an engineered alternative to
[supreme_bot.md](supreme_bot.md), the current direction. The two pursue the
same flows. Where they split is noted under [Where this
sits](#where-this-sits).

**Goal.** A search that does what a strong player does at
[Richards–Johnson](../analysis/richards-johnson-exchange/README.md):

- It reasons about a reply *kind* across many possible opponent racks at once,
  such as "Mike sets up a second S lane with YE\_\_".
- It carries what it learned about one candidate over to the others.
- It revisits a candidate when a later finding changes how its earlier search
  should be read.

**Decision.** Search a tree whose chance outcomes are a fixed set of sampled
**worlds** shared by every candidate. Back values up the way Gumbel MuZero
does, with completed values and an improved policy, but with a per-action
posterior in place of visit counts. Transfer between branches with a
**per-decision Bayesian linear regression**. It predicts how far a search
result departs from the network's prior, from features the network computes
for every move. The regression is refit as the search runs. It sets the
estimate and the uncertainty of every action that has not been searched, and
anything a search *has* reached overrides it. Train the features offline so
that this regression works: a meta-learned closed-form ridge over logged
search residuals.

## Purpose

[simulation_information_flow.md](../simulation_information_flow.md) walks an
idealized agent through Richards–Johnson and names what it does:

- **sideways valuation:** one ETS rack's result prices a quarter of the rack
  space;
- **sideways steering:** ALLEE's probes change which reply Mike tries after
  other candidates;
- **revision:** later probes re-price earlier ones.

It also shows why plain node statistics cannot carry those flows in
Scrabble. Most nodes below the first draw are visited once, so there is
nothing to average.

DeepBot gets all three flows from two pieces:

1. **A sound backup.** Each node's value is recomputed from its children's
   current estimates. Nothing is a running sum of past probes.
2. **One learned similarity, fitted in closed form.** It decides what a search
   result says about actions no probe reached.

The learned part is small: a feature map trained offline, plus a regression
with a few dozen weights fitted per decision. Aggregation, weighting by
amount of evidence, and shrinkage are classical statistics, not something
the network has to learn.

## Where this sits

- **[supreme_bot.md](supreme_bot.md)** learns every flow in one network that
  reads a context of every probe step. DeepBot fixes the backup rule and the
  form of the transfer (linear in learned features), and keeps per-node
  statistics. The trade-off:
  - **What DeepBot gains:** with transfer off it is an ordinary convergent
    search, which addresses SupremeBot's "no free convergence" risk. Its
    learning problem is much smaller than the reader's, which matters because
    M1a's reader found no transfer
    ([supreme_bot_m1a.md](supreme_bot_m1a.md)).
  - **What it gives up:** any transfer that is not linear in the features,
    and SupremeBot's learned draws.
- **[sim_residual_feedback.md](sim_residual_feedback.md)** supplies two
  principles DeepBot keeps:
  - a residual is measured against the prior's prediction for the *same*
    state;
  - candidates are allocated by expected gain over the best so far.

  It also supplies the root machinery: the move proposal shortlist, and the
  common-random-number sims of [sim_runner.h](../../engine/include/sim/sim_runner.h).
  Its two shortcomings motivate this plan:
  - hasty rollouts never play the reply that matters;
  - a key move may never be illuminated by the sims of its siblings.

  DeepBot searches the opponent's replies instead of rolling them out, and
  gives every candidate value from its own probes.
- **[rack_conditional_evidence.md](rack_conditional_evidence.md)** wanted the
  same per-rack transfer through learned evidence tokens. DeepBot's worlds
  play the part of its rack indices, and its regression stands in for the
  fusion stage.

In the flow vocabulary, DeepBot's profile is sideways in every column except
chance steering, plus revision:

| root steering | steering below root | chance steering | action valuation | chance valuation | revision |
|---|---|---|---|---|---|
| sideways | sideways | fixed (the worlds) | sideways | sideways | yes |

## The tree

**Worlds.** At the start of a turn, sample `K` worlds. A world fixes
everything hidden from the mover:

- the opponent's rack, drawn from the posterior
  ([belief/rack_inference.h](../../engine/include/belief/rack_inference.h));
  under face-up leaves, only the opponent's unseen draws;
- the bag order, which fixes every later draw.

Every root candidate is searched in the same `K` worlds, so the luck of the
draw cancels in comparisons between candidates. Worlds drawn from a proposal
other than the posterior carry importance weights `ω_k`. Drawing them from
the posterior itself (`ω_k = 1`) is the first version.

**Nodes.** The root is ours, and its edges are our candidates. Under
candidate `m`, world `k` gives the opponent's decision node `n(m, k)`. Below
that, nodes alternate between the two players, with draws taken from the
world's bag order. A node is **expanded** when its legal moves have been
generated and scored by the prior. An action whose child node has been
expanded is **searched**.

**Expanding a node.** Expansion costs one move generation plus one batched
pass of the move set evaluation model over every legal move `a`. The pass
returns:

- a **value prior** `v̂(a)`: win equity from the mover's point of view, with a
  score-difference distribution;
- a **feature vector** `φ(a)` ([Features](#features-and-their-training)).

Expansion does not roll anything out. A node's value comes from its children,
as below.

**Leaves.** A new node may instead get its value from a stronger estimator:

- once the bag is empty, the endgame solver at its configured budget
  ([endgame_solver.h](../../engine/include/endgame/endgame_solver.h));
- with one tile in the bag, the pre-endgame solver.

Neither is exact. An `EndgameVerdict` with `proven` set has near-zero
variance. An unproven one gets a variance from its error model
([Uncertainty](#uncertainty)).

## Values

Each action `a` at an expanded node carries an estimate `Q(a)` and a variance
`V(a)`, both from the node mover's point of view.

| action is | `Q(a)` | `V(a)` |
|---|---|---|
| searched | the child node's value, flipped to this mover | propagated from the child |
| a solver leaf | the verdict's win equity | 0 if proven, else the solver's error model |
| unsearched | `v̂(a) + μᵀφ(a)`, the prior plus the transferred correction | `φ(a)ᵀΣφ(a) + τ²(a)` |

`μ` and `Σ` are the regression posterior ([Transfer](#transfer)). `τ²(a)` is
the prior's own error variance, the part the features cannot explain
([Uncertainty](#uncertainty)).

**The improved policy.** At each node, let `π′(a)` be the probability that `a`
is the mover's best action, under independent Gaussians `N(Q(a), V(a))`. It is
estimated with a few dozen joint draws, which costs `O(draws · actions)` per
node. As variances shrink, `π′` sharpens toward the best action. While they
are wide, it stays spread across every action that could still be best.

**Node value.** `value(n) = Σ_a π′(a) Q(a)`, with variance
`Σ_a π′(a)² V(a) + Σ_a π′(a)(Q(a) − value(n))²`. This takes the place of both
the visit-weighted mean of MCTS and a hard max:

- A **mean over visits** dilutes the value with every exploratory child.
- A **max over estimates** picks whichever has the largest error, the
  optimizer's curse. The estimates here carry model bias that does not
  average away, and the max would favor the ones built on transfer, which are
  the noisiest.
- **Weighting by `π′`** stays near a mean while the evidence is thin, and
  approaches the max as it firms up.

This is Gumbel MuZero's completed-Q construction (Danihelka et al., 2022).
There, `σ` scales with the node's maximum visit count, which in effect makes
the node's mover rate every action with equal confidence. That is the wrong
answer here. Suppose 100 visits went to the wrong reply under a candidate,
and a later finding raises an unsearched reply. Gumbel's `σ` would then
trust that reply's guessed value with the confidence those 100 visits
bought. Per-action variances avoid this. The Gumbel `σ` form stays as an
ablation arm.

**Chance aggregation.** A candidate's value is the weighted mean over worlds,
`q(m) = Σ_k ω_k q(m, k) / Σ_k ω_k`. A world not yet expanded under `m` is an
unsearched edge like any other. Its estimate is the candidate's prior plus
the transferred correction, so worlds can be expanded lazily.

**Recompute, never accumulate.** Every value above is a function of the
current estimates below it. When the regression changes, every unsearched
estimate changes with it, and the tree is recomputed bottom-up. The cost is
linear in the size of the expanded tree. This is how the search revises
itself:

- searches made before a finding keep their values, which are still right
  about the actions they reached;
- their *weight* in each node's value is decided anew by `π′`.

## Transfer

**The model.** For each searched action, the **residual** is
`δ(a) = Q_raw(a) − v̂(a)`. That is how far search moved the value away from
the prior, for the same post-move state. Assume `δ(a) ≈ wᵀφ(a) + ε`, with a
prior `w ~ N(0, λ⁻¹ I)` and heteroscedastic noise `ε ~ N(0, V_raw(a) + τ²(a))`.

- There is one `w` per turn, shared by every node, world and candidate.
- The side to move and the depth are among the features, so one fit can hold
  different corrections for our replies and the opponent's.
- With `d` features, refitting is `O(N d²)` for `N` residuals. At `d ≈ 64` and
  a few thousand residuals, the fit is negligible beside move generation.

**Raw and corrected tracks.** Each node keeps two values:

- **Raw:** computed with transfer off. Unsearched actions sit at the plain
  prior `v̂`. Residuals and the regression fit use this track only.
- **Corrected:** includes the transfer. `π′`, allocation and the final pick
  use this one.

If corrected values fed the fit, a searched node's value would partly depend
on `w`, and the regression would be confirming its own output.

**The Richards case.** In world A, Mike's reply YEET is searched and turns
out to be a disaster for Nigel, a large residual. The posterior mean moves
along `φ(YEET)`. In world B, YETI has a nearby feature vector, so its
estimate rises and its variance falls, and `π′` shifts weight toward it.
YENS, the same footprint without the S hook, lies elsewhere in feature space
and barely moves. Under VELATE, which also plays five tiles, Mike's replies
share those feature directions and inherit the correction. Under the
exchange the bag-count feature differs, so they inherit less.

**Transfer yields to search.** Transfer only sets the estimates of actions
nothing has reached. Searching an action replaces its transferred estimate
with the subtree's. In the limit where everything is searched, the
regression has no effect. A wrong transfer therefore costs efficiency (search
goes to the wrong places) but cannot outlast the search that contradicts it.

## Search control

A **probe** picks a root candidate and a world, walks down choosing actions,
expands one node, refits, and recomputes the path. Workers run probes in
parallel and batch the refit and recompute every `B` probes.

**At interior nodes**, choose the action that maximizes
`π′(a) − N(a) / (1 + Σ_b N(b))`, where `N` counts probes through each
action. This is Gumbel MuZero's interior rule: it pulls visit shares toward
`π′`. One addition: a per-world Gumbel perturbation of `π′` at the
opponent's nodes. Each world then tries a different handful of low-prior
replies, and the regression spreads whatever one world finds to the others.
Discovery costs about one search per reply kind instead of one per world.
The paper's regret guarantee covers only the root, so the perturbation has
to be tested.

**At the root**, allocate probes by expected gain over the incumbent,
`E[max(0, q(m) − q(best))]`, using the candidates' paired variances over the
shared worlds. This is the proves-best acquisition of sim_residual_feedback.md,
computed from the tree instead of predicted by a head.

**The candidate set** is the move proposal model's shortlist, plus a
**timing floor**: the best candidate for each number of tiles played, 0
through 7, with the top few exchanges. Transfer between candidates is weakest
along the bag-count axis. Nothing in ALLEE's subtree says "keep the
bag full", so recall along that axis cannot depend on transfer. At
Richards–Johnson the floor is what brings EELLT in.

**The pick** is the candidate with the highest corrected `q(m)`, among
candidates that have reached a minimum share of the budget.

## Features and their training

**The features.** `φ(a)` concatenates:

- a learned projection `g(h(a))` of the move set model's per-move hidden
  state `h(a)`. That is the vector its value readout reads, computed for
  every legal move during expansion at no extra cost;
- mechanical consequence features:
  - the cross-check change summary
    (`CrossCheckEncoder`'s inputs: hooks created and destroyed, and S hooks
    in particular);
  - tiles played, the bag count after the move, the score;
  - the side to move, and the depth.

The mechanical features hedge against a blind spot in the representation
itself. If `h(a)` does not encode "S hook at O8, with the bag empty and the
mover holding an S", no `w` can express the correction. A cross-check delta
encodes the hook exactly.

**Training the projection** is meta-learning with a closed-form base learner
(R2D2, Bertinetto et al., 2019):

1. **The corpus:** logged searches. Each refined action records its world,
   its path, `v̂(a)`, `Q_raw(a)` and `V_raw(a)`. Following the
   replay-reconstruction invariant ([architecture.md](../architecture.md)),
   the log stores moves and world seeds, not features. The trainer replays
   them and recomputes `h(a)` and the mechanical features.
2. **An episode** is one logged turn. Split its residuals into a support set
   and a query set at random, stratified by world, so that held-out worlds
   test transfer across chance outcomes.
3. **The base learner** solves the ridge on the support set in closed form.
   It is differentiable in `g`.
4. **The loss** is the heteroscedastic squared error of the ridge's
   predictions on the query set, together with the predictive variances
   (Gaussian NLL), so `Σ` is calibrated as well as `μ`.
5. **The backbone** is frozen at first, so the priors `v̂` cannot drift while
   `g` learns. Fine-tuning it jointly, with the student's distillation loss
   as an anchor, is a later step.

**Controls**, matching M1a's discipline:

- *shuffled:* residuals permuted among a turn's actions;
- *mechanical only:* `g` removed;
- *frozen `h`:* identity `g`.

Transfer is the query error with real residuals minus the error with
shuffled ones.

**Bootstrapping.** The first corpus comes from DeepBot with frozen features
and identity `g`, or with transfer turned off. Neither needs a trained
projection. Searches logged with transfer off are the cleaner corpus, because
the allocation never depended on the features being learned.

## Uncertainty

Three error models feed the variances. All are fitted from the search corpus:

- **`τ²(a)`, the prior's error.** First as a per-band constant: bag count,
  plus side to move. The squared residuals on a held-out corpus give it.
  Later, possibly a head on the student that predicts `|δ|`.
- **The solver's error** on unproven verdicts, as a function of the unseen
  count and the budget, measured against high-budget solves.
- **The leaf model's error** where a node is evaluated by the teacher rather
  than expanded, if the depth cap requires that.

These variances drive both `π′` and allocation, so a miscalibrated variance
skews the search systematically. Checking calibration (empirical coverage
of the predicted intervals on held-out residuals) is part of every
milestone.

## Improving the priors

Transfer corrects what is specific to a position. Patterns that recur, such
as "a setup that empties the bag next to an S lane while its player holds an
S", belong in the network's weights. The priors lack them today because their
training data comes from HastyBot-style play, which never makes those setups
([sim_labeled_candidates.md](sim_labeled_candidates.md)).

The generational loop moves search results into the priors:

- **Student values** distill from the raw search values `Q_raw(a)` of actions
  searched to a minimum effort, alongside the teacher. Raw, so the target
  never contains the transfer it is meant to replace.
- **The teacher** retrains on DeepBot self-play outcomes once DeepBot plays
  whole games, under the [generational lifecycle](../generational_training.md).
- **The projection `g`** retrains on each generation's search logs. As the
  priors improve, residuals shrink, and transfer covers a narrower remainder.

## Information sets

Within a world, every player's decision sees that world's hidden tiles.

- **Our nodes below the root** see the opponent's rack.
- **The opponent's reply values** are computed through subtrees in which our
  draw is fixed by the world's bag order, so the opponent sees it too.

This is strategy fusion. At the opponent's first reply it favors the
opponent: our candidates are valued as if the opponent knew our draw. It is
likely small at ply one, and grows deeper in the tree.

- **Under face-up leaves**, deferred draws remove most of it
  ([supreme_bot.md](supreme_bot.md#face-up-leaves-deferred-draws)): sample a
  player's draw when that player next decides. For DeepBot, that means a
  world fixes the draws per *decision* rather than per bag position. The
  shared-world pairing then holds only up to the first deferred draw.
- **In standard Scrabble**, the opponent reasons about our hidden leave.
  SupremeBot's opponent contexts are one answer. For DeepBot this is an open
  question, and the first milestones run face-up.

## Cost

Per turn, with `C` candidates, `K` worlds and `E` expanded nodes:

- **Expansions:** `E` move generations and `E` batched move set passes. One
  full expansion layer at ply one alone is `C·K`, for example 20 × 200 =
  4,000. Lazy world expansion and allocation keep `E` well under the full
  layer, and the budget sets it.
- **Refits:** `O(N d²)` every `B` probes.
- **Recompute:** the corrected estimates of every unsearched action at every
  expanded node, `O(E · actions · d)`. At `E` = 4,000, 500 actions and
  `d` = 64, that is about 130M multiply-adds per recompute: one batched GPU
  call, or batching `B` probes per recompute.
- **`π′`:** `O(draws · actions)` per node on the recomputed path. A full
  recompute needs it at every node, so it batches with the recompute.

The memory footprint is the expanded tree plus each node's `φ` matrix,
4,000 × 500 × 64 floats ≈ 0.5 GB at FP32, so `φ` is stored in BF16 or `E` is
capped.

## Build order

Each step has a measurement that decides whether to take the next one.

1. **E0, the offline transfer test (no search code).**
   - *Setup:* Richards–Johnson after ALLEE and after VELATE. Sample about 200
     worlds. Generate Mike's replies, and run the solver on the prior's top
     replies plus every YE\_\_ reply. The bag is empty after any reply, so the
     values are solver verdicts.
   - *Measure:* fit the ridge on half the (world, reply) residuals, using
     frozen `h` plus the mechanical features, and predict the other half.
     Also fit after seeing YE\_\_ replies in only one or two worlds.
   - *Pass if:* the held-out YE\_\_ replies with an S hook are predicted near
     their solved values, YENS-like replies stay near zero, and ALLEE's fit
     transfers to VELATE's replies.
   - *Also run:* the same test on the other known positions (pos-09,
     egotize, the ACETA setup, a sample of the
     [blind_spots](../blind_spots.md) tag), and a residual rank check
     (how much variance a few components explain).
   - *Kill criterion:* if the residuals are high-rank noise everywhere, the
     regression transfer is dead before any search is built.
2. **E1, the search core with transfer off.** Worlds, expansion, solver
   leaves, the per-action `π′` backup, the interior rule, root expected gain,
   the timing floor. Compare against BestBot and `mset-sim` at equal time,
   and against the Gumbel-`σ` ablation. The backup alone must not lose to
   them. Log its searches: they are the first projection corpus.
3. **E2, transfer with frozen features.** Measure:
   - discovery cost: probes until YE\_\_ replies carry their solved share of
     `π′` across worlds, with and without the per-world perturbation;
   - the Richards pick;
   - known positions;
   - match play against E1.
4. **E3, the meta-learned projection.** Train `g` on E1/E2 logs with the
   shuffled, mechanical-only and frozen controls. Repeat E2's measurements.
5. **E4, the generational loop.** Student distillation from raw search
   values, then DeepBot self-play for the teacher.
6. **E5, standard Scrabble.** Posterior worlds from rack inference, and the
   information-set question above.

## Risks

- **Linear transfer may be too weak.** Several unrelated errors in one turn
  must lie in separable feature directions. If they interfere, the fallback
  is a Gaussian process with a learned kernel: the same construction with
  more capacity, at `O(N³)` per fit.
- **The variances are the control surface.** An overconfident `τ²` collapses
  `π′` onto the prior's favorite, which is the blind spot this plan exists to
  escape. An underconfident one spreads the budget thin. Calibration is
  measured at every step.
- **Expansion throughput.** Full move generation and a move set pass at every
  expanded node is far more per node than a hasty rollout ply. If `E` per
  second is too low for match time controls, the budget goes to ply one and
  two, where discovery matters most, and deeper values come from the teacher.
- **The worlds bound chance-node discovery.** Mike's racks are fixed in
  advance, so a reply that appears in no sampled world's rack is never
  searched; at best it is priced by transfer. `K` and
  posterior-weighted sampling set that coverage. Steering which worlds are
  sampled, with importance weights, is a later extension.

## Open questions

- **The value inside `π′`.** Whether `π′` should temper "probability of being
  best" with a learned policy prior, to model a realistic rather than optimal
  opponent.
- **Depth.** The depth cap below the opponent's reply when the bag is not
  close to empty, and how the budget splits between more worlds and deeper
  search.
- **Residual targets.** Whether residuals should be in win equity or in a
  transformed space (logit), where errors near certain wins and losses are
  comparable.
- **One `w` per turn or one per candidate group.** Sharing across candidates
  is what transfers ALLEE to VELATE, but it may blur corrections that differ
  between candidate kinds.

# Sim-residual feedback: re-evaluating moves with Monte Carlo evidence

## Purpose

Move selection (roadmap Phase 4) runs `M_pre` over all legal moves, sends the
top-`K` to Monte Carlo simulation, and picks the best simmed move. This
document proposes making that a **loop**: the simulation results for each
simmed candidate are fed back into `M_pre` as *evidence*, and the full move
set is re-evaluated conditioned on that evidence, yielding a new set of
candidates to sim before the final pick.

The informative signal in the evidence is the **residual** — the gap between
what the sims observed and what the network predicted. The intuition: each
position has nuances driven by lexical facts — which words are makeable where,
with which unseen tiles — that the network cannot fully derive, because doing
so would require searching the lexicon over tiles not on the current rack,
especially for future turns. This is the network's primary blind spot
(established by the lexical-NN track; see
[lexical_features_for_value.md](lexical_features_for_value.md)). Monte Carlo
rollouts *do* consider those possibilities, honestly, via the GADDAG. When the
sims reveal that a board region is hotter or more dangerous than the network
believed, that surprise is an indirect but grounded clue about lexical
structure. A network trained to condition on sim evidence can produce a
runtime-search-informed re-evaluation that no amount of static input
engineering can match, because the evidence comes from deep, position-specific
search.

In the design below the residual is computed *inside* the network — the raw
sim observations are fed in, paired with the network's own predictions — rather
than precomputed as an input. The reasons are architectural and are covered in
[Evidence-set conditioning](#evidence-set-conditioning-the-architecture).

## Where this sits

- **Roadmap Phase 4** ([roadmap.md](roadmap.md)): the one-round pipeline —
  GADDAG generates all moves, `M_pre` scores them in one cross-attention pass,
  top-`K` go to simulation. This proposal wraps that pipeline in an iteration.
- **Scribblez.pdf §8.1 (Search-Derived Knowledge Buffers)**: the design doc
  envisions a buffer where truths discovered during search are recorded for the
  network to read. The evidence set below is a concrete instantiation of that
  buffer, with a natural training story.
- **The belief system's iterative particle generation** (Scribblez.pdf §3.5)
  follows the same idiom: propose, gather evidence, condition on the evidence,
  re-propose. Here the proposer is `M_pre`, the evidence is sim results, and
  the re-proposal is the next candidate set. At its sequential extreme (see
  [the schedule spectrum](#the-schedule-spectrum)) the loop is a learned,
  amortized root search — candidate expansion informed by accumulated rollout
  evidence, MCTS-at-the-root in spirit.
- **Engineered lexical features**
  ([lexical_features_for_value.md](lexical_features_for_value.md)) are
  complementary, not redundant. The features attack *recall* — getting
  lexically promising moves into the first candidate set at all. The evidence
  loop attacks a different failure: the first-pass evaluation misjudged the
  board, the sims exposed it, and the re-evaluation corrects the ranking.
  Neither mechanism subsumes the other (see
  [Limitations](#limitations-and-caveats)).
- **`M_post`'s `OppNextPlacement` head**
  ([training_targets.h](../engine/include/scribblez/training_targets.h))
  already predicts a 15×15 mask of where the opponent's next move will place
  tiles. The heads below are that head conjoined with the game outcome — this
  proposal upgrades an auxiliary head into a load-bearing part of the decision
  loop.

## The per-square outcome heads

Two 15×15 heads on the value models, each a per-square Bernoulli probability
(implemented: `OppWinPlacementTarget` / `SelfWinPlacementTarget` in
[training_targets.h](../engine/include/scribblez/training_targets.h), served
with the marginal placement heads by the shared `mask_conv` stack in
[model.py](../py/scribblez/post_move_value/model.py)):

- **Opponent danger**: for square `S`,
  `Pr[opponent's next move occupies S  AND  opponent wins the game]`.
  A high value reads as "this is a dangerous spot — to the extent the opponent
  wins more than the baseline suggests, a play here is likely the cause."
- **Self opportunity**: `Pr[our next move occupies S  AND  we win the game]` —
  hot spots for our own follow-up.

**Training targets are free from self-play logs.** For any sampled position the
log contains the next move each player made and who won, so the target for each
head is the indicator plane (placed squares of that next move) multiplied by
the win indicator. Loss is per-square BCE. An exchange or pass places no tiles,
so its plane is all zeros — the heads are per-square probabilities, not a
distribution over squares, and need not sum to anything.

Note what the conjunction mixes: a square can score high because the player
*plays* there often, or because playing there *wins* often. Each conjunction
is therefore paired with a marginal occupancy head (`opp_next_placement` /
`self_next_placement`) so the network can disentangle the two.

## Sim evidence

The Monte Carlo sim of candidate move `M` runs `S` rollouts from `M`'s
post-move state (opponent racks sampled, play continued by the rollout
policy). Each rollout observes an actual opponent reply (its placed squares)
and an outcome, so the sim yields, per head, an **empirical map** — the
fraction of rollouts in which the opponent's reply occupied `S` and the
opponent went on to win — directly comparable to the network's prediction for
`M`. It also yields a **sim value estimate** (empirical WLD over the rollouts)
and the **rollout counts** (a confidence signal; empirical maps from a few
hundred rollouts are noisy, and the network needs to know how much to trust
them).

### Evidence must stay paired with its move

It is tempting to compress the per-candidate evidence into shared board-level
planes (say, an element-wise mean or max of the candidates' danger maps) so it
can be fed through the input encoder like any other feature. This loses the
part of the signal that matters most. Suppose candidate `A` leaves a hot spot
open and the opponent capitalizes on it in `A`'s rollouts, while candidate `B`
neutralizes it. An aggregate map shows danger at the hot spot but not *which
moves suffer it* — and the mapping from a move to "does the danger persist" is
itself lexical. `B` can kill the threat without occupying the hot squares at
all: by consuming the anchor the threat needed, by changing a cross-check
adjacent to the lane, by using up a hook. Asking the network to reconstruct
that conditionality from an aggregate re-introduces exactly the blind spot the
mechanism exists to fix. The paired evidence *demonstrates* the conditionality
instead — "`A` was simmed, danger at `H12`; `B` was simmed, danger gone" — and
the network only has to interpolate from demonstrated contrasts, not derive
blocking from first principles.

So the unit of evidence is the **pair**: (move encoding, sim observations for
that move). Aggregation across pairs is left to the network, where it can be
learned.

## Evidence-set conditioning: the architecture

The re-evaluation is a set-conditioned scoring:
`score(M′ | board, {(Mᵢ, sim-resultᵢ)})` for every legal move `M′`, given the
evidence pairs of whatever candidates have been simmed so far. (In ML terms
this is an attentive-neural-process shape: context pairs of input = move,
observation = sim outcome, queried at new inputs.) Concretely:

- **Evidence tokens.** Each simmed candidate becomes one token: its move
  encoding (`M_pre`'s move encoder, reused) fused with an encoding of its sim
  observations — the empirical maps, the sim value, the rollout counts — plus
  the network's own first-pass predictions for that move (already computed, so
  free to include; this hands the network the residual contrast directly
  rather than requiring it to recompute its earlier output).
- **Evidence self-attention.** The tokens attend to one another. Contrasts
  between pairs are the point — "these two moves differ *here*, and their
  danger maps differ correspondingly" is a pairwise computation.
- **Fusion into the board encoding.** The evidence tokens cross-attend with
  the board's spatial map `H`, producing an evidence-conditioned `H′`. The
  per-move scoring machinery is untouched: all `N` candidates cross-attend
  into `H′` exactly as they attend into `H` in the evidence-free pass. The
  network can write "danger at `H12` unless the lane is disturbed" into the
  spatial features in whatever learned form it finds useful — this is the
  learned replacement for any hand-crafted aggregation.

An **empty evidence set** must degrade gracefully to the plain one-pass
`M_pre` (the fusion stage becomes a no-op or near-no-op); training covers this
case explicitly (see [Training](#training)).

Because the fusion stage sits between the shared trunk and the heads, `M_post`
can take the same evidence input through the same stage — needed for the
distillation story below.

### Incremental inference across rounds

Between rounds at one decision point the raw inputs do not change — only the
evidence set grows — so nothing requires re-running the expensive stages.
The harness caches the network's intermediate activations:

- The **trunk output `H`** is computed once per decision (this is where almost
  all the FLOPs live), and the **`N` move encodings** are computed once — the
  legal-move set does not change.
- Per round, the incremental work is the evidence fusion stage (self-attention
  over the evidence tokens, cross-attention against the cached `H` to produce
  `H′`) plus one re-scoring pass (the cached move encodings cross-attending
  into `H′`). The re-scoring genuinely must re-run — the scores changing is
  the point — but it is the cheap linear pass that motivates `M_pre` in the
  first place.

This is the same mechanism as a KV-cache in a transformer decoder: the
"memory" is engineered and lossless — tensors held by the harness — with zero
learning burden on the network, and outputs bit-identical to a full recompute.
Every round therefore remains reproducible from stored inputs, in the same
spirit as the replay-reconstruction invariant
([architecture.md](architecture.md)).

Caching is also what makes **late fusion a load-bearing constraint** rather
than an incidental choice: evidence must not modulate the trunk's own layers,
or the trunk cannot be cached across rounds. The cost of late fusion is a
bound on how deeply evidence can reshape the spatial features; whether a few
post-trunk attention layers suffice is exactly what the kill-test measures.

**Why not a learned memory instead?** An alternative reading of "the network
holds the position in memory" is a fixed-size recurrent state, updated as
evidence arrives, from which re-evaluations (or a proposed next candidate) are
read out without re-presenting the board or the move set. This is rejected,
for three reasons in increasing order of severity:

- A learned state is lossy compression, and the deliverable is fine value
  *margins* between specific candidates. The candidate that matters most in
  this loop is precisely the low-salience one — the blocker sitting far down
  the ranking until evidence arrives — and low-salience content is what
  compression drops first. The cached-activation design compresses nothing.
- Emitting a next candidate without the move list present would require the
  network to *generate* a move — spell the word, place the tiles. The
  lexical-NN track showed that generation is the network's demonstrated blind
  spot (it recovers a play's score and anchor geometry but cannot fill the
  interior letters), and it inverts the system's division of labor: the GADDAG
  finds moves, the network values them. The workable alternative — *pointing*
  at a candidate via an argmax over the legal set — requires the move
  encodings to be present to score over, which is exactly the cached design.
- A network that is stateful across rounds makes training sequential
  (backpropagation through the round sequence, inherently ordered rows). The
  evidence-set-as-input formulation is stateless per call: a training row is
  (position, evidence set, targets), rows stay independent, and prefix sizes
  can be sampled freely.

The one place the architecture does adopt a learned recurrent state — the
belief compressor (Scribblez.pdf §3.5) — works because its object, a
distribution over racks, is genuinely soft and low-dimensional, and its
consumer is a decoder producing samples. Scores over `N` specific candidates,
where the decision rides on small margins between named alternatives, are the
opposite kind of object.

## The decision procedure

One generic loop, parameterized by a schedule (`B` candidates proposed per
round, `R` rounds):

1. GADDAG generates all `N` legal moves.
2. Evaluate all `N` with the current evidence set (initially empty) and
   propose the top `B` unsimmed candidates.
3. Sim the proposed candidates; append their (move, sim-result) pairs to the
   evidence set.
4. Repeat from 2 until `R` rounds have run (or the sim budget is spent).
5. Final pick: best move by simulation value over all simmed candidates.

**Why the payoff is promotion, not re-scoring.** For moves already simmed, the
sims themselves provide the ranking — a re-evaluation adds little. The value
of conditioning is that the next proposal round can **promote moves that no
earlier round selected**: e.g. a modest-scoring play that blocks the
newly-discovered hot spot at `H12`. A one-round pipeline can never select such
a move, no matter how many rollouts it spends.

A helpful robustness property: opponent hot spots are discovered by simming
*any* candidate that fails to block them — the opponent's rollout replies land
there regardless of which of our candidates was simmed. So coverage of
opponent danger is not sensitive to the exact composition of the first
proposal batch.

### The schedule spectrum

The architecture conditions on an evidence set of arbitrary size, so the
schedule is a tunable policy, not an architectural fork:

- **(B = K, R = 2)** — two batched rounds: sim the top-`K`, condition, sim the
  newly promoted moves, pick. Simplest data generation and training
  distribution; sims within a round parallelize; the top-`K` batch provides
  candidate diversity for free.
- **(B = 1, R = K)** — fully sequential: propose one, sim it, condition,
  propose the next. Every sim is maximally informed by all prior evidence, so
  the budget is better targeted — after simming `A` and seeing the `H12`
  danger, the network can propose the blocker `B` *as its next candidate*,
  which a batch selected up-front can never do.
- Intermediate (`B` small, a few rounds) interpolates.

Sequential trade-offs to weigh:

- **Latency, not throughput or compute.** `R` sequential sim-then-infer round
  trips serialize the decision. With
  [incremental inference](#incremental-inference-across-rounds) the
  network-side cost of `R` rounds is one trunk pass plus `R` cheap incremental
  passes, so the serialization cost is almost entirely sim latency. For
  self-play data generation even that matters less than it appears: the
  game-pool design ([generational_training.md](generational_training.md))
  keeps many games in flight and batches their GPU evaluations, so per-game
  serialization does not starve the hardware. Per-move *latency* does suffer,
  which matters for interactive or competitive play.
- **Exploration becomes an explicit problem at small `B`.** A batch top-`K`
  is diverse by construction. A greedy `B = 1` proposer may propose
  near-duplicates of the current best — exploitation, when what later sims
  should buy is *information* — and nothing in distillation-style training
  incentivizes information-seeking proposals. An acquisition mechanism on top
  of the learned score is therefore load-bearing at small `B`: a
  footprint/lane-overlap novelty penalty against already-simmed moves is the
  cheap version, and
  [covariance-guided selection](#covariance-guided-candidate-selection) is the
  principled one.
- **Training must cover every prefix.** The network sees evidence sets of size
  `0, 1, …` at inference, so training rows must span those sizes, and data
  generation must record the sequential trajectory that produced each set.

The recommended starting point is **(B = K, R = 2)**, moving toward smaller
`B` only if measurement shows targeted sims beat batch diversity.

## Covariance-guided candidate selection

Which candidate to sim next is an exploration problem the learned scores alone
cannot answer. If the top two moves `A` and `B` are near-duplicates — say, the
same tiles on the same squares, differing only in an inconsequential blank
instantiation — then after simming `A`, simming `B` is wasted budget no matter
what the evidence says, while a qualitatively different move `C` is worth
testing. The scores rank `B` second; nothing in them says `B` is *redundant*.

**Why not PUCT.** MCTS's exploration formula optimizes cumulative regret with
an independent prior per child. Here only the final pick matters (simple
regret — this is best-arm identification, not bandit play), and the arms are
heavily *correlated*. Correlation is the load-bearing structure: it is what
makes "don't sim `B` after `A`" fall out of the math rather than needing a
hand rule. The formal home is Bayesian optimization / best-arm identification
with correlated beliefs (the knowledge-gradient-for-correlated-normal-beliefs
setting).

### Two covariances, two jobs

The construction "run `S` sims for each of `M` moves, with sim `i` sampling
the unknown information identically across moves" produces an empirical
`M×M` covariance matrix that mixes two correlation sources, each useful for a
different purpose:

- **Epistemic value correlation.** Near-duplicate moves have nearly *equal
  true values* — uncertainty about `Q(A)` and `Q(B)` is shared. This is the
  covariance that should guide candidate selection: once `A` is observed, the
  posterior over `B` collapses with it and `B` carries no information, while a
  weakly correlated `C` stays uncertain and becomes the right probe.
- **Common-random-numbers (CRN) correlation of estimators.** Under shared
  seeds, `Var(Q̂_A − Q̂_B) = Var(Q̂_A) + Var(Q̂_B) − 2·Cov`, so paired sims make
  *comparisons* far sharper than independent sims — the dominant noise
  (rack and draw luck) cancels in the difference. This is variance reduction
  for the final pick and the stopping rule, not for exploration.

Macondo's simmer already uses CRN — its sim loop holds the sampled opponent
rack "constant on a per-iteration basis... the same for every play in plays"
(`montecarlo/montecarlo.go` in the Macondo repo) — and its
similar-plays stopping logic is best-arm identification with an
independent-arms model. The covariance model below is the missing correlation
structure. Two mechanical notes: candidates place different numbers of tiles,
so "the same randomness" for bag draws means a fixed shuffled bag order
consumed as needed; and CRN's benefit decays with rollout depth as the
trajectories diverge (the rack sample, the largest noise source, aligns
perfectly).

The empirical CRN covariance measures redundancy directly: high covariance
means the two moves respond identically to the same futures — decision-
redundant — and the seeds where their outcomes *differ* are exactly the
informative ones.

### A learned low-rank covariance head

Empirical covariance exists only between already-simmed moves; choosing an
unsimmed `C` requires *predicted* covariance — a model. Two facts make this
cheap:

- **Low-rank, not `N×N`.** `M_pre` emits, per move, a small embedding
  `φ(M) ∈ ℝʳ` alongside its score, with
  `Cov(M, M′) ≈ φ(M)·φ(M′)` plus a per-move diagonal noise term. One more
  per-move output head; the `O(N)` scoring pass is unchanged.
- **Training targets are a free byproduct.** Evidence generation already sims
  `K` candidates per labeled position; running those sims with shared seeds
  yields a `K×K` empirical covariance — `K(K−1)/2` pairwise targets per
  position to fit `φφᵀ` against. No new generation machinery.

### The root posterior and the acquisition rule

With predicted means (the `M_pre` scores), predicted covariance
(`φφᵀ + diagonal`), and sim results as observations, maintain a Gaussian
posterior over the value vector. The acquisition rule can then be as simple as
**Thompson sampling**: draw `Q̃ ~ N(μ, Σ)`, sim the argmax of `Q̃`, update.
Near-duplicates are automatically suppressed — their posteriors collapse
together with the simmed move's — while qualitatively different moves keep
winning draws as long as their uncertainty is unresolved. Posterior updates
are `O(N·r + r³)` at low rank. Knowledge-gradient or entropy-search
acquisitions are the principled upgrades within the same posterior if Thompson
proves too blunt.

The mechanism is schedule-agnostic: at `B = 1` it picks the next single
candidate; at larger `B` a diverse batch is one that maximizes posterior
information volume (a DPP-flavored selection over `φ`).

Three further consequences:

- **It generalizes footprint dedup.** The roadmap Phase 4 note about a cheap
  footprint-level dedup of the top-`K` is the degenerate rule
  "correlation ≈ 1 ⇒ drop one"; the posterior handles the continuum.
- **It sharpens the final pick and the stopping rule.** What matters between
  the top contenders is `P(Q_A > Q_B)`, governed by the variance of the
  *difference* — which is exactly what CRN shrinks and the covariance model
  estimates. This is the principled version of Macondo's similar-plays
  stopping condition.
- **It completes the division of labor.** The evidence-conditioned network is
  the learned *mean* update — how all `N` scores shift given what the sims
  revealed. The covariance layer supplies the *uncertainty* the network's
  point estimates fundamentally lack. Rather than trying to train
  information-seeking behavior into the proposer (which distillation targets
  give no incentive for), the exploration decision lives in an explicit,
  interpretable posterior bolted onto learned means and covariances.

### Caveats and sequencing

- **Gaussianity.** WLD outcomes are bounded and often near the extremes; model
  the posterior in logit space, or accept the approximation — at the level of
  sim aggregates (means over hundreds of rollouts) it is mild.
- **Low-rank misses idiosyncratic tails.** Two moves identical except that one
  leaves a hook open are mostly correlated but differ in the tail; the
  diagonal term absorbs some of this, and whether `r` must grow is empirical.
- **Noisy targets.** A `K×K` covariance from `S` sims is itself an estimate;
  fine at `K ≈ 10`, `S` in the hundreds, with the fit pooling across many
  positions.
- **Sequencing.** This is an additional layer (covariance head + root
  posterior), and cheap-before-rich applies: the footprint novelty penalty is
  the v0 acquisition, this is the v1, built only if the small-`B` schedule
  shows value. It lives inside roadmap step 6, off the kill-test's critical
  path — with one exception: the CRN requirement on the sim machinery
  (shared seeds across candidates at a position) must be in place from step 2,
  or the covariance targets never exist.

## Training

### Evidence semantics

The evidence input for a training row is **the set of (move, sim-result) pairs
gathered at the decision point** — uniform for every candidate being scored,
whether or not that candidate is itself in the set. This uniformity is what
keeps `M_pre` distillation targets well-defined across the whole move set (an
"own-sim" input would be undefined for the unsimmed majority).

### `M_post` with evidence

Data generation, per labeled position:

1. Run the proposal/sim schedule at the position with the current model,
   recording the evidence trajectory.
2. Store the **raw sim observations** — empirical maps, sim values, counts —
   alongside the `.slog` data. Raw observations are model-independent, so they
   never go stale as the model trains; the network's own predictions (the
   other half of each evidence token) are recomputed live at train time by the
   same forward machinery inference uses.
3. Training targets are unchanged: final game outcome (WLD, ScoreDiff) plus
   the per-square conjunction heads from the log.

Rows are trained at multiple evidence-prefix sizes, including **size zero** —
the zero-evidence rows are what keep the evidence-free first pass from
degrading, and they are free (every unlabeled position is one).

### `M_pre` with evidence

`M_pre`'s training story is unchanged in shape: it distills `M_post` (roadmap
Phase 4), now with the evidence set present on both sides through the shared
fusion stage. For a labeled position, the target for candidate `M` is
evidence-conditioned `M_post` evaluated on `M`'s post-move state, given the
same decision-point evidence. The "label a subset, mask the loss" strategy
from Phase 4 applies unchanged. (An alternative for later-round scoring —
training directly against sim values for simmed moves plus game outcome,
bypassing `M_post` — is coherent but departs further from the Phase 4
pipeline; it is noted as an open question.)

### The cost elephant

Evidence-conditioned training rows require running the sims at data-generation
time. At `K ≈ 10` and a few hundred rollouts per candidate, that is thousands
of rollout-games per labeled position — even at HastyBot speeds (~ms/game),
minutes per game of self-play, a ~10³–10⁴× slowdown over plain generation.
Mitigations, all compatible:

- **Label a sparse subset of positions.** Only rows carrying evidence need the
  sims; the rest train with empty evidence (needed anyway, per above). The
  labeled fraction is a free parameter, exactly as in Phase 4 target
  generation.
- **Cheap sims for labeling.** HastyBot rollouts with modest `S`; the rollout
  counts tell the network the noise level, so small `S` is a soft degradation,
  not a correctness problem.
- **The generational pipeline is built for this.**
  [generational_training.md](generational_training.md) exists precisely
  because expensive generation makes per-game reuse mandatory; evidence
  labeling is a (heavy) instance of the pattern, and the stored observations
  are per-generation artifacts like any other precomputed feature.

## Limitations and caveats

- **The sim is not unbiased ground truth.** Its "truth" is filtered through
  the opponent-rack sampling scheme (uniform over unseen tiles until the
  belief system lands) and the rollout policy (HastyBot initially). The
  evidence therefore encodes a mixture of lexical blindness, rack-inference
  mismatch, rollout-policy weakness, and Monte Carlo noise. Lexical blindness
  is expected to dominate — it is the one *systematic* gap between the network
  and the GADDAG-driven rollouts — but the conflation is real and worth
  remembering when interpreting results.
- **Self-created opportunities stay out of reach.** Sim evidence covers only
  the simmed post-move boards. A move whose value exists *only* because of
  structure it creates (an `S`-hook it opens), and which no proposal round
  selected, is never simmed — no evidence can rescue it. That recall gap
  belongs to the cross-check-delta feature
  ([lexical_features_for_value.md](lexical_features_for_value.md)); the two
  mechanisms genuinely need each other. (A sequential schedule softens this —
  a later round *can* propose an opportunity-creating move if earlier evidence
  hints at the region's value — but only the move's own sim reveals value that
  exists solely post-move.)
- **Correlation with the self-play rollout policy.** In HastyBot-generated
  training data, the actual opponent reply and the sim rollouts follow the
  same policy, so the empirical maps are highly predictive of the logged
  reply. The conjunction-head loss will improve for that shallow reason; the
  metric that matters is **WLD / calibration improvement**, not the
  spatial-head loss.
- **Per-square SNR.** Reply footprints are ~2–7 squares of 225, so most
  squares' empirical probabilities are small and variance-dominated at
  practical `S`. Genuine hot spots concentrate mass (they recur across
  rollouts), which is what makes them detectable; the count inputs exist so
  the network can discount the rest.
- **Evidence-map encoding is an open design point.** Each evidence token must
  encode a spatial observation (the 15×15 maps). Options range from a small
  conv encoder pooling each map to a vector, to keeping the maps spatial and
  letting the fusion cross-attention consume them per-square. The pooled form
  is cheaper; the spatial form preserves the *where*. Start pooled + let the
  fusion stage also see the maps as planes tagged per-token if the kill-test
  says location detail is being lost.

## De-risking: the kill-test

The load-bearing hypothesis is narrow: *conditioning on sim evidence improves
the value model's outcome prediction.* That is testable offline, cheaply,
before any agent or `M_pre` work:

1. Take a modest set of self-play positions; for each, generate evidence
   (top-`K` by the current `M_post` over candidate post-move states, HastyBot
   rollouts, empirical maps + values + counts).
2. Train evidence-conditioned `M_post` (evidence encoder + fusion stage) vs.
   the plain baseline on identical data.
3. Compare held-out WLD loss and calibration (the Phase 3 machinery).

If the evidence-conditioned model shows no WLD improvement, the whole loop is
moot — and that is learned without touching `M_pre`, the multi-round agent, or
any selection-time plumbing. The same experiment de-risks the fusion
architecture itself (evidence tokens, self-attention, board fusion), which is
the main new network component.
[monte_carlo_sim_tool](../engine/apps/monte_carlo_sim_tool.cpp) (the
Positions-tab ground-truth generator, see
[react_dashboard.md](react_dashboard.md)) already contains most of the rollout
machinery this experiment needs.

## Implementation roadmap

| Step | Build | Depends on | Status |
|------|-------|-----------|--------|
| 1 | Conjunction heads on `M_post` (targets from logs; per-square BCE). Independent value as probes even if the loop is never built. | — | **Done** — `opp_win_placement` / `self_win_placement`, plus the `self_next_placement` marginal so both conjunctions have an occupancy partner, through the full pipeline (target registry, decoder, FFI, model heads + BCE losses, ONNX export, TensorRT binding, dashboard loss series). |
| 2 | Sim machinery emits per-square empirical maps + value estimates + counts (extend the `monte_carlo_sim_tool` rollout core into a reusable `SimRunner`); **common random numbers across candidates at a position** (shared rack samples, fixed bag order) so pairwise covariance targets exist; storage format for sim observations alongside `.slog`. | 1 | — |
| 3 | **Kill-test** (above): evidence-conditioned `M_post` vs. baseline. **Go/no-go gate for everything below.** | 2, Phase 3 eval machinery | — |
| 4 | Evidence encoder + fusion stage in the shared trunk; multi-prefix-size training; evidence labeling integrated into generational data generation at a sparse position fraction. | 3 | — |
| 5 | `M_pre` inherits the heads and the fusion stage; distillation from evidence-conditioned `M_post`. | 4, roadmap Phase 4 | — |
| 6 | Multi-round agent (the decision procedure above); schedule tuning (`B`, `R`); acquisition — footprint novelty penalty first, then the covariance head + root posterior if the small-`B` schedule shows value; match-play eval vs. the one-round agent (Phase 3 agent-eval harness). | 5 | — |

Steps 1–3 are cheap relative to what they de-risk and are worth doing early;
steps 4–6 ride the Phase 4 timeline.

## Open questions

- **Schedule (`B`, `R`) and budget split** — including whether later rounds
  should use smaller `S`, and where on the acquisition ladder (novelty
  penalty → covariance-guided posterior) each schedule needs to sit.
- **Covariance parameterization** — the rank `r`, whether to fit in logit
  space, and whether `φφᵀ + diagonal` captures enough of the
  identical-except-one-hook tail structure.
- **Evidence-map encoding** — pooled-vector vs. spatial (see caveats); cheap
  to ablate inside the kill-test.
- **Whether scalar evidence alone captures most of the win.** If evidence
  tokens carrying only the sim value residual (no spatial maps) already
  deliver the WLD improvement, the spatial machinery can be deferred — the
  cheap-before-rich sequencing of
  [lexical_features_for_value.md](lexical_features_for_value.md) applies here
  too.
- **Later-round training targets** — evidence-conditioned `M_post`
  distillation (the default above) vs. training directly against sim values
  for simmed candidates.
- **Sim reuse across rounds** — candidates retained across rounds keep their
  rollouts; whether to top up counts as the evidence set grows.

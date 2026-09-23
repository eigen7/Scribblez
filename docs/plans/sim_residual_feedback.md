# Sim-residual feedback: re-evaluating moves with Monte Carlo evidence

**Status: built; waiting on training.** All six steps of the [implementation
roadmap](#implementation-roadmap) have code in the tree. The kill-test passed;
the fusion stage, the move proposal model and its trainer, the engine runtime
and the sequential agent (UltimateBot, `--type=ultimatebot`) exist. What
remains is compute: a proposal model trained on a deployment-quality corpus,
then the budget and stopping-threshold measurements. The gen-1 frozen-backbone
trial is recorded as the floor that training has to beat. The per-candidate
evidence token described here is also the base that
[rack_conditional_evidence.md](rack_conditional_evidence.md) proposes to
replace with per-rack evidence.

**Goal.** Turn move selection into a loop. Today's pipeline (roadmap track A)
runs the move set evaluation model over all legal moves, sends the top `K` to
Monte Carlo simulation, and picks the best simmed move. Here, each simmed
candidate's sim results are fed back into the model as *evidence*, and the
move set is re-evaluated conditioned on that evidence to choose the next
candidate to sim, before the final pick.

**Decision.** Condition on the **residual**, the gap between what the sims
observed and what the network predicted, and compute it inside the network by
feeding raw observations in beside the network's own first-pass predictions.
Choose the next candidate with a learned **expected-gain** ("proves-best")
head, one candidate at a time behind a mechanical anchor.

The residual is informative because the network's main blind spot is
lexical: it cannot search the lexicon over tiles not on the current rack (see
[lexical_features_for_value.md](lexical_features_for_value.md)), while
GADDAG-driven rollouts do consider those plays. A sim surprise is therefore a
grounded clue about lexical structure.

## Where this sits

- **Roadmap track A** ([roadmap.md](../roadmap.md)): wraps the one-round
  pipeline in an iteration.
- **[design.md](../design.md) §8.1, search-derived knowledge buffers:** the
  evidence set is a concrete instance, with a natural training story.
- **The belief system's iterative particle generation**
  ([design.md](../design.md) §3.5) uses the same propose / observe / condition
  idiom. At its sequential extreme the loop is a learned, amortized root
  search.
- **Engineered lexical features**
  ([lexical_features_for_value.md](lexical_features_for_value.md)) are
  complementary. They attack *recall* (getting lexically promising moves into
  the first candidate set); the evidence loop corrects rankings the sims show
  to be wrong. Neither subsumes the other.
- **The position evaluation model's placement heads** already predict where
  each player's next move lands; the conjunction heads below combine that with
  the game outcome.

## The placement heads

The position evaluation model has four placement heads, each a categorical
distribution over the 2927 move footprint classes
([training/footprint.h](../../engine/include/training/footprint.h)), trained
by masked softmax cross-entropy
([training_targets.h](../../engine/include/training/training_targets.h),
[position_eval/model.py](../../py/scribblez/position_eval/model.py)). The
per-cell form this plan was first written against was replaced by
footprints; see [footprint_native_placement.md](footprint_native_placement.md).

- **Occupancy** (`opp_next_placement`, `self_next_placement`): the footprint
  of each player's next move.
- **Opponent danger** (`opp_win_placement`): the opponent's next footprint,
  conjoined with the opponent going on to win.
- **Self opportunity** (`self_win_placement`): the mover's next footprint,
  conjoined with the mover winning.

Targets come free from self-play logs. A conjunction head's target is the
footprint when that seat went on to win, and a "not-win" class otherwise; a
pass or exchange is its own class. Each conjunction mixes "plays there often"
with "playing there wins often", which is why it is paired with an occupancy
head: together they let the network separate the two.

### Per-move placement planes

The heads read whatever position they are given, so the position evaluation
model yields a candidate's four distributions when run on that candidate's
**post-move state**. That is the only form the evidence loop can use (see
[sim evidence](#sim-evidence) for why the root's planes are the wrong
comparison). The move set evaluation model, which scores all `N` candidates
against one board encode, therefore has **per-move** versions of the same
four heads, distilled from the teacher's distributions at each candidate's
post-move state, just as its value heads are distilled from the teacher's
WLD.

Decoding a board-shaped distribution from a per-move vector needs a spatial
readout the scoring path does not otherwise have: per head and footprint slot,
the move's embedding is projected to a query that scores against each of the
225 board tokens. The heads are *read* only for the handful of simmed
candidates, but *trained* over every labeled candidate, which is where their
cost lands ([open questions](#open-questions)).

## Sim evidence

The sim of candidate `M` (`S` rollouts from `M`'s post-move state) yields, per
placement head, an **empirical distribution** (a histogram of the footprints
the rollouts played), plus a **sim value estimate** (empirical WLD and score
moments) and the **rollout count**. The count is a confidence signal: it tells
the network how much to trust a histogram built from few rollouts.

The histogram is comparable only to the network's prediction **for `M`'s
post-move state**, not to the root position's planes. The rollouts observe
plies played after `M`; the root planes predict the reply to whatever we end
up playing. Differencing the two would charge `M` with a change it caused,
and for a blocking candidate, causing that change *is* the merit being
measured, so the mismatch would erase exactly the signal the loop hunts for.
Supplying the right comparison is what the [per-move placement
planes](#per-move-placement-planes) are for.

Sims of different candidates at one position share their random draws (the
same sampled opponent racks, and a fixed shuffled bag order consumed as
needed): **common random numbers (CRN)**, implemented in
[sim_runner.h](../../engine/include/sim/sim_runner.h). Rack and draw luck then
cancel in *comparisons* between candidates, which is what the final pick and
the stopping rule rely on.

**Evidence stays paired with its move.** Aggregating the candidates'
histograms into shared board-level planes would lose which moves suffer a
danger, and the mapping from move to "does the danger persist" is itself
lexical (a move can kill a threat without occupying its squares). The unit of
evidence is the (move, sim result) pair; aggregation across pairs is left to
the network.

## Evidence-set conditioning: the architecture

Re-evaluation is set-conditioned scoring,
`score(M′ | board, {(Mᵢ, sim-resultᵢ)})` for every legal move `M′`, in the
shape of an attentive neural process
([evidence_fusion.py](../../py/scribblez/evidence_fusion.py)):

- **Evidence tokens.** One per simmed candidate: its move encoding (the move
  set evaluation model's move encoder, reused), fused with its sim
  observations (histograms, value, count) **and the network's own predicted
  placement distributions for that candidate**. Observed and predicted are
  stacked channel-wise, so the encoder sees observation and prediction for
  the same square side by side and forms the residual itself. Keeping both
  halves raw matters: a confident prediction contradicted by 40 rollouts and
  an unsure one contradicted by 2000 have the same difference and warrant
  different updates.
- **Evidence self-attention.** Contrasts between pairs of candidates are the
  point.
- **Fusion into the board encoding.** The board's spatial map `H`
  cross-attends into the evidence tokens, producing an evidence-conditioned
  `H′`. The per-move scoring machinery is unchanged: it attends into `H′`
  exactly as it attends into `H`.

**The predictions have to be inputs; the network cannot recover them.** An
evidence encoder that reads observations alone (the kill-test's
[sim_evidence/model.py](../../py/scribblez/sim_evidence/model.py), whose fusion
is a plain additive `x + ev_spatial`, with the encoder blind to `x`) can
express `posterior = prior + g(observation)` but not
`posterior = prior + k·(observation − prior)`. The second needs a term that
scales the prior down, and nothing downstream of an additive merge can
separate the two summands again. Such a model learns a correction that is
*marginal* over its own belief states: the same shift whether it had already
priced the danger in or was blind to it, which double-counts confirmations
and damps genuine surprises toward the average. Feeding the predictions in as
channels restores the contrast without moving the fusion stage.

An **empty evidence set** must reduce to the plain one-pass model; training
covers this case explicitly, and the fusion stage hard-gates itself to a
no-op on an empty set. The fusion stage sits between the shared trunk and the
heads, so the position evaluation model can take the same evidence through
the same stage (needed for the distillation variant below).

**Incremental inference.** At one decision point only the evidence set
changes between rounds, so the trunk output `H` and the move encodings are
computed once and cached; per round only the fusion stage and the cheap
re-scoring pass run, with outputs bit-identical to a full recompute. This
makes **late fusion a load-bearing constraint**: evidence must not modulate
the trunk's own layers, or the trunk cannot be cached across rounds. The
engine runs this as two ONNX graphs, a per-turn `cache` graph and a
per-iteration `step` graph ([evidence_staging.h](../../engine/include/agent/evidence_staging.h)).

The predictions carried by an evidence token are the **evidence-free
first-pass** predictions, so they cache with the move encodings and a token
never changes once created. Taking them from the conditioned pass instead
would make round `r`'s stored prediction a function of round `r−1`'s
evidence: a token that drifts as the set grows, defeating the caching and
turning the residual into a difference against an already-corrected belief.

**Rejected: a learned recurrent memory** (a fixed-size state updated as
evidence arrives). It is lossy compression, and the candidate that matters
most here is precisely the low-salience one that compression drops first.
Proposing a candidate without the move list present would require the network
to *generate* a move, its demonstrated blind spot. And a stateful network
makes training sequential, where the evidence-set formulation keeps training
rows independent.

## The decision procedure

The proposer's job is the same every round: given the sim results for the
candidates simmed so far (possibly none), pick the next candidate or
candidates to sim. One generic loop, parameterized by a schedule of `B`
candidates proposed per round over `R` rounds:

1. The GADDAG generates all legal moves.
2. Evaluate all of them under the current evidence set (initially empty) and
   propose the top `B` unsimmed candidates.
3. Sim the proposed candidates; append their (move, sim result) pairs to the
   evidence set.
4. Repeat from 2 until `R` rounds have run or the sim budget is spent.
5. Final pick: the best move by simulation value among all simmed
   candidates.

The first sim slot goes to a **mechanical anchor**, the highest-raw-score
move, regardless of the proposer's ranking. It is cheap insurance against the
model's blind spots, and its sim is valuable evidence: the residual on the
obvious move calibrates the rest of the evidence set.

The payoff is **promotion, not re-scoring**. Simmed moves are ranked by their
sims directly; conditioning matters because a later round can promote moves
no earlier round selected, such as the modest play that blocks a newly
discovered hot spot. Opponent hot spots are discovered by simming *any*
candidate that fails to block them, so danger coverage does not depend much
on the first batch's composition.

### The schedule spectrum

- **B = K, R = 2: two batched rounds.** The coarsest conditioning (two prefix
  sizes). Needs no acquisition mechanism beyond the value ranking plus a
  footprint-novelty penalty for diversity within a batch.
- **B = 1, R = K: fully sequential.** Every sim is informed by all prior
  evidence, and the loop admits early stopping (halt when no unsimmed move is
  likely to prove best). A greedy proposer tends to propose near-duplicates of
  the current best, so an acquisition mechanism ([candidate
  selection](#candidate-selection)) is load-bearing at small `B`.
- **Training covers every evidence-set size**, including zero. The evidence
  set is order-free (the fusion stage is permutation-invariant and the gain
  label is a max over the set), so the unit of training data is a *subset* of
  a position's simmed pool, not a recorded chain. One pool supplies
  combinatorially many evidence sets ([trajectory
  generation](#evidence-trajectory-generation)).

Wall-clock barely distinguishes the schedules at the intended budgets.
Rollouts parallelize *within* a candidate, and at hundreds to thousands of
rollouts per candidate one sim saturates the hardware, so total sim compute
is `K·S` either way; sequential rounds add only `R` cheap fusion-and-rescore
passes plus barrier waits. The design center is therefore **B = 1, R = K**:
sequential proposal behind the mechanical anchor, with batch mode kept as the
fallback if the sequential proposer fails to beat it.

## Candidate selection

With `N` candidates simmed, which should be the `(N+1)`th? This is an
exploration problem. We want to sim candidates likely to prove best, and a
candidate is likely to prove best if:

- **A.** it looks good on its own, and
- **B.** it looks different from previous candidates (a candidate that sims
  identically to an earlier one cannot come out strictly better).

Two approaches were considered:

1. **Model the covariance between candidates.** This helps with B. But a
   covariance target is hard to define, and it is unclear how to blend it
   with A in a principled way.
2. **Model "proves best" directly:** predict how much a candidate's sim would
   improve on the best so far.

**Decision: option 2**, as an **expected gain**,
`E[max(0, p(w) − best)]`, not a probability of gain.

Two structural facts about this target:

- **It is a thin transform of the conditioned value.** The acquisition score
  is approximately `E[max(0, conditioned value + sim noise − best-so-far)]`:
  a calibrated comparison of the evidence-conditioned value against a known
  scalar, at a noise level set by the rollout counts. The hard part is the
  conditioned value. The head cannot be the only training path, though: its
  labels exist only for simmed candidates, a handful per position chosen by
  the data-generation proposer, at thousands of rollouts each.
- **At an empty evidence set it reduces to value ranking.** With best-so-far
  at the floor, `E[max(0, p(w) − best)]` collapses to `E[p(w)]`, the value
  prediction itself. (The probability form instead degenerates to 1 for every
  move.) First-round proposal by value is the empty-evidence special case of
  the acquisition rule, not a separate mechanism.

The details:

- **Why expected gain and not probability.** The target is what a sim
  actually contributes, since the final pick is by sim value. Common random
  numbers make this decisive. Two candidates that differ only cosmetically (a
  blank designated differently, placed tiles reordered) place the same tiles,
  so they leave the same rack and refill from the same pool. Under a shared
  seed, rollout `i` then runs against a board that differs in a few inert
  letters and returns the *same* outcome. Their paired difference is near
  zero with almost no spread, so their expected gain is about 0, and the
  target suppresses redundant sims **by construction**, with no novelty
  penalty or footprint dedup. (The cancellation is exact only while rollouts
  run to the end of the game; see the truncation point below.)

  The probability form works only in the *exact*-tie case of requirement B:
  a candidate that sims identically never strictly exceeds, so its
  probability is 0. That case is rarer than it looks. A differently
  designated blank puts different letters on the board, changing hooks and
  cross-checks, so rollouts eventually diverge. Once the difference is small
  but non-zero, the probability form fails both ways: about 0.5 when the
  difference is noise-dominated, and about 1 when the duplicate is reliably a
  hair better (two more points off a premium square), spending a whole sim to
  discover two points of spread. Expected gain rates all three cases at about
  0, the correct answer in each.
- **The label must be CRN-paired.** The improvement is measured against the
  best-so-far *over the same seed set*, which is what makes a duplicate's
  target about 0 rather than a small random number. Labels drawn from one
  position's `.sobs` satisfy this automatically, since its candidates share a
  seed base. An improvement computed against a value from a different sim run
  does not, and silently reintroduces the noise the pairing cancels. The
  aggregate record suffices: with identical seed sets, the difference of
  means *is* the paired mean difference.
- **Value truncation weakens the cancellation, as a bias.** With truncated
  rollouts ([roadmap.md](../roadmap.md) item 2), the value model is read at the
  horizon, and two cosmetically different candidates no longer return
  identical outcomes: their leaves differ by those few letters, and the model
  evaluates them slightly differently. That difference is *deterministic
  given the boards*, so it is bias rather than variance: the paired mean
  difference converges to it instead of averaging to zero, and more rollouts
  do not remove it. A duplicate's expected gain under truncation is therefore
  bounded by the leaf model's local smoothness rather than structurally 0,
  and the effect is worst at shallow horizons, where genuine divergence has
  not yet built up and the difference between leaves is nearly pure model
  idiosyncrasy.

  The argument survives for two reasons. What matters is a duplicate's
  expected gain *compared with* a genuinely different candidate's, and the
  spurious term has to exceed a real improvement to misorder them. And the
  head *predicts* expected gain from the move and the evidence rather than
  computing it: per-instance leaf error does not generalize, so a regularized
  head regresses it toward zero. It survives only where the leaf model's bias
  is systematic enough to be learnable (consistently preferring one blank
  designation, say), which is a leaf-model calibration problem, not an
  acquisition-target one.
- **Winner's curse.** The best-so-far is a max of noisy sim estimates and is
  biased upward, and near-ties make the label a coin flip driven by rollout
  noise. The rollout-count inputs exist for exactly this; the head is
  calibrated to the sim configuration that produced its labels.
- **Policy dependence.** The distribution of (evidence set, best-so-far)
  reflects whichever proposer generated the trajectories, so the head trains
  increasingly off-policy as the proposer improves. This is handled
  generationally, like the scorer.
- **Batch diversity at `B` > 1.** The head scores candidates independently,
  so a top-`B` batch can be near-duplicates *of each other*; the footprint
  novelty penalty supplies diversity within a batch. At `B` = 1 the issue
  disappears.
- **Training rows.** From the simmed candidates a position already has
  (`.sobs`), any evidence subset plus a held-out simmed candidate is a labeled
  row (the set is order-free): combinatorially many correlated rows per
  position, with no new generation machinery.
- **Scope.** The head only picks the next candidate to sim. The stopping rule
  and the final pick between simmed contenders still rely on the paired (CRN)
  sim estimates.

## Training

### Evidence semantics

A training row's evidence input is the set of (move, sim result) pairs
gathered at the decision point, **the same for every candidate being
scored**, whether or not that candidate is in the set. This keeps
distillation targets well-defined across the whole move set.

### The position evaluation model with evidence (deferred)

Per labeled position: run the proposal/sim schedule, record the evidence
trajectory, and store the **raw sim observations** beside the `.slog` data.
Raw observations are model-independent and never go stale. The network's
own predictions (the other half of each token) are recomputed live at train
time, one extra evidence-free forward pass per simmed candidate at the
post-move state the replay already reconstructs, so there is no new head and
no new stored artifact. Targets are unchanged (WLD, score-diff, placement
heads). Rows train at several evidence-prefix sizes including **zero**; the
zero-evidence rows keep the evidence-free pass from degrading, and cost
nothing.

### The move proposal model

The deployed evidence consumer is a separate model: a **copy** of the move
set evaluation student (trunk, move encoder, heads, fusion stage) plus the
proves-best head. The student itself stays a pure distillation model (and,
under D2, the rollout policy); the copy is what trains on evidence. Two loss
components ([roadmap.md](../roadmap.md) item 5 is the spec):

- **The proves-best gain** (primary): Huber against the CRN-paired gain of a
  held-out simmed candidate over its evidence set's best. The best-so-far is a
  **known scalar at inference** (the max sim value over the evidence gathered
  so far), so it is fed to the head as an **input**
  (`evidence_fusion.best_so_far`) rather than reconstructed from the pooled
  evidence. The evidence reaches the head only through a mean pool, which
  cannot represent the max that the target depends on.
- **Conditioned value heads** (auxiliary): the conditioned WLD and score-diff
  against the held-out candidate's own sim outcome. The gain is a thin
  transform of the conditioned value, so these rows feed the head at no extra
  sim cost. The targets are sim outcomes, never the plain teacher, whose
  board-only readout would train the fusion stage to ignore evidence.

The backbone trains, so the copy is free to follow the sim signal, starting
from the student's ranking. The empty-evidence (prefix-0) rows keep the
evidence-free pass calibrated as a board-only prior on the simmed candidates.

There is deliberately **no self-distillation anchor**. An anchor would add
only one thing: extending that calibration to the *unsimmed* legal moves the
gain argmax ranges over at deployment. Doing that cleanly needs a live
frozen-student forward pass over **all** `N` candidates of each training
position, and the replay pipeline reconstructs encoded inputs, not a move
list; there is no move generator it can call, so all-`N` coverage would be a
new engine build for a speculative stabilizer. The gain head instead
generalizes from a diverse held-out set (anchor, on-policy, and low-value
off-policy draws) and the student starting point, with the backbone learning
rate as the drift knob. Whether the argmax over unsimmed moves holds up is
measured at the agent, and the anchor is a known fallback if it does not.

Distilling from an evidence-conditioned position evaluation model (the
previous section) remains the deferred, richer variant: dense conditioned
labels over the full move set, at the cost of first training that conditioned
teacher on real outcomes.

### Evidence-trajectory generation

The training pools come from running sims at ordinary self-play positions.
The concrete recipe (the greedy anchor, `A` on-policy picks, `B` uniform
off-policy draws, all at the deployment rollout configuration) is
[roadmap.md](../roadmap.md) item 4. Two structural facts shape it.

**Exploration is cheap in exactly the way AlphaZero's is not.** Which
candidates get simmed never changes the played move or the game outcome:
evidence labeling is a side computation on positions from ordinary
self-play, and a junk sim's label is its true CRN-paired gain, which is
correct data, not target pollution. The only cost of an off-policy sim is its
rollouts. So the off-policy floor can be generous, and it must exist. A move
class the model rates near zero is otherwise never simmed, never labeled and
never corrected, a blind spot that persists precisely when the teacher shares
it, which is the lexical case this loop exists for. Most floor sims confirm
"terrible, gain ≈ 0": a correct label in a region that otherwise has none,
and what stops the head from hallucinating gain there and spending deployment
sims on it.

Generation 0 keeps the floor assumption-free: a uniform draw over the legal
moves the anchor and on-policy picks did not take. The tempting objection is
that uniform draws almost never land on the interesting near-miss move. That
misreads the division of labor: near-miss candidates are what the
full-support on-policy softmax already covers, so the floor's job is the
*negative* one of confirming that the move classes the proposer ignores are
ignored for cause. Uniform does that well. And because the raw move list
enumerates every exchange subset and every blank designation, uniform draws
already sample exchanges and the tail at their (variant-inflated) natural
frequency, with no hand-specified stratum. Stratified or semantic draws (the
contention zone, high-leave plays, setups) remain a later refinement if the
uniform floor proves too coarse.

**The evidence set is order-free, so rows are assembled, not replayed.** Any
subset of the anchor and on-policy sims is a valid evidence set, and every
simmed candidate outside it is a labeled held-out row. One pool yields
combinatorially many rows, the right response to sims costing about 1,000
rollouts each. Three constraints keep the assembled distribution on
deployment's:

- every set contains the anchor, as every deployed set does;
- set sizes stop at the deployment sim budget, since larger sets are states
  the agent never visits;
- off-policy draws never enter a set, since deployed evidence holds only the
  anchor and proposer picks. They are labels only, coverage bought at zero
  cost to the input distribution.

**Why the on-policy side must be on-policy** (selected by the conditioned
loop itself rather than by a static ranking), for three reasons, none of them
AlphaZero's pollution argument. An off-policy selector wastes the budget
re-simming near-twins of the incumbent, whose gain labels are about 0:
correct, and worthless. Only proposals from the current model put labels on
the conditionally strong candidates the head exists to find. And the evidence
*sets* the model trains under should look like the sets the deployed loop
walks; sets built by a value softmax are bags of near-twins, correct labels
over an input distribution deployment never produces. Generation 0, with no
trained gain head, selects this side by a temperature softmax over the plain
student's values on the full candidate set; later generations run the
conditioned loop.

**The self-play games stay HastyBot's.** Playing the best simmed move instead
would couple exploration randomness and sim noise into every outcome-derived
target, and could only apply at the sparse labeled turns anyway.
Search-improved self-play is the generational pipeline's job: once the
sequential agent exists, a later generation regenerates whole games with it
as the playing policy, which also moves the corpus's *positions* onto the
distribution the agent actually faces.

### Cost

Evidence-carrying rows require running the sims at data-generation time: at
the deployment configuration, a pool of about 20 candidates × 1,000 rollouts
per labeled position, a 10³ to 10⁴ slowdown over plain generation.
Mitigations, all compatible:

- **Value truncation** (roadmap item 2): cheaper per rollout and cleaner
  evidence, per the kill-test's phase gradient.
- **Subset assembly**: combinatorially many rows per pool, so every rollout
  feeds many training rows.
- **Labeling a sparse subset of positions**: the rest train with empty
  evidence, which is needed anyway.
- **The generational pipeline**
  ([generational_training.md](../generational_training.md)), which exists for
  this reuse pattern.

**Not a mitigation: small `S`.** The count input lets a corpus with mixed `S`
degrade softly, but the corpus must include deployment-quality histograms: a
head trained only on noisy evidence has never seen the histograms it will be
asked to trust.

## Limitations and caveats

- The sim is not unbiased ground truth. The evidence mixes lexical blindness
  (expected to dominate, as the one systematic gap between network and
  rollouts) with rack-sampling mismatch, rollout-policy weakness, and Monte
  Carlo noise.
- Self-created opportunities stay out of reach: a move whose value exists
  only in structure it creates, and which no round proposes, is never simmed.
  That recall gap belongs to the lexical input features.
- In HastyBot self-play the logged reply and the rollouts share a policy, so
  the conjunction heads' loss improves for a shallow reason. The metric that
  matters is WLD and calibration, not the placement-head loss.
- A reply occupies 1 to 7 of 225 squares, so at practical `S` most of a
  histogram is noise. Genuine hot spots recur across rollouts, and the count
  input lets the network discount the rest.
- Evidence encoding was an open design point: a pooled vector (cheap) or
  spatial planes (keeps the *where*). The built fusion stage uses spatial
  planes, widened to footprint slot channels
  ([footprint_native_placement.md](footprint_native_placement.md)); whether
  they pay is the placement-plane ablation under [open
  questions](#open-questions).

## De-risking: the kill-test

The load-bearing hypothesis: *conditioning on sim evidence improves the value
model's outcome prediction.* Tested offline: the evidence-conditioned
position evaluation model against the plain baseline on identical data,
compared on held-out WLD loss and calibration.

**Result: passed.** Evidence gain of −0.0063 CE at 5.7 SE, with clean
controls. The magnitude is bounded by saturation of the root readout, and an
8× late-versus-early phase gradient supports the mechanism. Full numbers and
conclusions: [sim_obs_experiment_results.md](../sim_obs_experiment_results.md).

The pipeline is [sim_obs_tool](../../engine/apps/sim_obs_tool.cpp)
(candidates are the HastyBot-equity top K, so each position's evidence
contains the played move's own sim) feeding
[kill_test.py](../../py/scripts/kill_test.py). The evidence-conditioned model
is [sim_evidence/model.py](../../py/scribblez/sim_evidence/model.py), a
zero-initialized fusion stage on the regular post-move model, so the arms are
parameter-identical and differ only in their inputs.

```
# Generates self-play data + sim observations until stopped; resumable.
./py/scripts/generate_kill_test_data.py -t apple

# 4-armed test; can run (and rerun) while generation continues.
./py/scripts/kill_test.py -t apple
```

Data accumulates under `<mount>/tags/kill_test/<tag>/data/slogs` (`.slog`
batches plus `.sobs` sidecars, written atomically). Per-arm history is in
`<mount>/tags/kill_test/<tag>/cache/results/<arm>.json`, and the decision
metric is the best held-out `wld_ce`. Reading the arms:

- **`full` < `none`** by a margin that dwarfs seed noise: the hypothesis
  survives.
- **`shuffled` ≈ `none`** is the validity check (shuffled evidence has the
  same marginals but belongs to other positions). Measure `full`'s gain
  against `shuffled`.
- **`scalar` against `full`** locates how much of the gain needs the spatial
  planes.
- Evidence arms are **leave-one-out** by default: the played move's own sim is
  masked, because deployment only re-scores *unsimmed* moves, so leave-one-out
  gains are the transfer gains that matter. The optional `ownsim` arm
  (`--arms ownsim`) prices that shortcut.

### Face-up leaves

`--open-leaves` on both commands (under a dedicated tag) runs the same
experiment in **face-up-leaves Scrabble**: the tiles a player kept from their
last move are public, and replenishment draws stay hidden. This is the
variant the project develops in ([roadmap.md](../roadmap.md)), so it is the
main mode, not an instrument; the hidden-leaves arm remains runnable, and the
gap between the two is what belief inference would have to close. Compare arm
deltas only within a mode.

Mechanics: the model input gains the opponent-leave counts block
(`kOppLeaveCounts`, `input_encoder.h`); the leave is derived at replay time
from the `.slog` draws, with no game-runner changes; the `.sobs` header
records the condition, so mixing modes within a tag fails loudly.

```
./py/scripts/generate_kill_test_data.py -t apple-open --open-leaves
./py/scripts/kill_test.py -t apple-open --open-leaves
```

## Implementation roadmap

| Step | Build | Depends on | Status |
|------|-------|-----------|--------|
| 1 | Conjunction heads on the position evaluation model, targets from logs. Useful as probes even if the loop is never built. | — | **Done.** `opp_win_placement` / `self_win_placement`, plus the `self_next_placement` occupancy partner, through the full pipeline (target registry, decoder, FFI, model heads and losses, ONNX export, TensorRT binding, dashboard loss series). Since made footprint-categorical. |
| 2 | Sim machinery emitting per-head empirical distributions, value estimates and counts; **common random numbers across candidates at a position**; a storage format for sim observations beside `.slog`. | 1 | **Done.** [sim_runner.h](../../engine/include/sim/sim_runner.h) (CRN rollouts over play, exchange and pass candidates; W/D/L and delta moments; footprint histograms) and [sim_observation_log.h](../../engine/include/data/sim_observation_log.h) (the versioned `.sobs` sidecar). |
| 3 | **Kill-test**: evidence-conditioned position evaluation model against the baseline. **Go/no-go gate for everything below.** | 2 | **Done: passed** (above). |
| 4 | Evidence encoder and fusion stage reading the shared trunk, with tokens carrying the model's post-move predictions beside the observations; multi-size evidence-set training. | 3 | **Built on the student side**: the fusion stage and its exactness tests ([evidence_fusion.py](../../py/scribblez/evidence_fusion.py)) and the multi-prefix trainer (`py/scribblez/evidence/`). Evidence training of the position evaluation model stays deferred with the conditioned-teacher variant. |
| 5 | The **move proposal model** ([above](#the-move-proposal-model)): the student copy with the proves-best head (best-so-far as an input), trained gain-first with the sim-outcome auxiliaries and no self-distillation anchor, on subset-assembled rows from the hybrid pools. | 4; roadmap items 2 to 4 | **Built; training waits on a deployment-quality corpus.** The gen-1 frozen-backbone trial over the 200-rollout v1 corpus is the recorded floor: conditioned − plain soft-CE −0.0008, acquisition hit rate 0.57 against the plain value's 0.61. |
| 6 | The sequential agent (the decision procedure at `B = 1, R = K`) driven by the proves-best head; budget tuning and the early-stopping threshold. Batched multi-round scheduling is the fallback, not a step on the way. | 5 | **Built** as UltimateBot ([roadmap.md](../roadmap.md) item 6, [ultimate_bot_agent.h](../../engine/include/agent/ultimate_bot_agent.h), loop in [evidence_loop.h](../../engine/include/agent/evidence_loop.h)). Budget and threshold measurements wait on a trained head. |

## Open questions

- **Budget split**: whether later sims should use a smaller `S`, and the
  early-stopping threshold for the sequential schedule.
- **A fusion refinement for the move proposal model.** Feeding best-so-far
  as an input settles the one statistic a mean pool could not carry, so a
  mean+max or attention pooling of the evidence summary is not needed for
  that. One refinement remains open, cheap and unvalidated: direct
  cross-attention from each move query to the evidence tokens (`O(N·K)`), so
  a candidate compares itself to each simmed move by encoding rather than only
  through shared board squares. Today a leave-twin with a different footprint
  is visible only through the move scalars and the pooled summary.
- **A/B/T of the pool recipe.** A ≈ 15 and B ≈ 5, with evidence-set sizes up
  to the deployment budget, are starting points. On-policy depth per position
  trades against position count once sim throughput binds, and the stopping
  rule's statistics will say where the marginal sim is worth more.
- **Whether the spatial machinery pays at all.** Against the cheap option so
  far, it has not: the kill-test's `full` arm matched its `scalar` arm to
  ±0.0003, so at a root-WLD readout the planes are inert
  ([sim_obs_experiment_results.md](../sim_obs_experiment_results.md)). That
  is expected, since a position-level scalar has no use for per-move spatial
  discrimination. This plan still commits to spatial, per-move,
  prediction-paired evidence, because the effect it is built for is
  *promotion* (a move no earlier round ranked highly rising once a hot square
  is exposed), which a root readout structurally cannot show. The commitment
  is a bet, settled after the build by the placement-plane ablation in
  [evaluation_plan.md](../evaluation_plan.md): evidence tokens with and
  without the model's predicted planes, read at promotion rather than at root
  WLD. A null there sends the loop back to the scalar rung, not just back a
  step.
- **Sim reuse across rounds**: candidates kept across rounds keep their
  rollouts; whether to top up their counts as the evidence set grows.

# Sim-residual feedback: re-evaluating moves with Monte Carlo evidence

## Purpose

Move selection (roadmap track A) runs the move set evaluation model over all
legal moves, sends the top-`K` to Monte Carlo simulation, and picks the best
simmed move. This document makes that a **loop**: the sim results for each
simmed candidate are fed back into the model as *evidence*, and the move set
is re-evaluated conditioned on that evidence, yielding new candidates to sim
before the final pick.

The informative signal is the **residual** — the gap between what the sims
observed and what the network predicted. The network's primary blind spot is
lexical — it cannot search the lexicon over tiles not on the current rack
(see [lexical_features_for_value.md](lexical_features_for_value.md)) — while
GADDAG-driven rollouts do consider those possibilities. A sim surprise is
therefore a grounded clue about lexical structure. The residual is computed
inside the network: raw sim observations are fed in alongside the network's
own first-pass predictions.

## Where this sits

- **Roadmap track A** ([roadmap.md](roadmap.md)): wraps the one-round
  pipeline in an iteration.
- **[design.md](design.md) §8.1 (search-derived knowledge buffers)**: the
  evidence set is a concrete instantiation, with a natural training story.
- **The belief system's iterative particle generation** ([design.md](design.md)
  §3.5) follows the same propose/observe/condition idiom. At its sequential
  extreme the loop is a learned, amortized root search.
- **Engineered lexical features**
  ([lexical_features_for_value.md](lexical_features_for_value.md)) are
  complementary: they attack *recall* (getting lexically promising moves into
  the first candidate set); the evidence loop corrects rankings the sims
  expose as misjudged. Neither subsumes the other.
- **The position evaluation model's placement heads** already predict where
  each player's next move lands; the conjunction heads below conjoin that
  with the game outcome.

## The per-square outcome heads

Two 15×15 per-square Bernoulli heads on the value models (implemented:
`OppWinPlacementTarget` / `SelfWinPlacementTarget` in
[training_targets.h](../engine/include/training/training_targets.h), served
by the shared `mask_conv` stack in
[model.py](../py/scribblez/position_eval/model.py)):

- **Opponent danger**: `Pr[opponent's next move occupies S AND opponent wins]`.
- **Self opportunity**: `Pr[our next move occupies S AND we win]`.

Targets are free from self-play logs (the next move's placed squares times
the win indicator); loss is per-square BCE; a pass or exchange is an all-zero
plane. Each conjunction mixes "plays there often" with "playing there wins
often", so each is paired with a marginal occupancy head
(`opp_next_placement` / `self_next_placement`) to let the network
disentangle the two.

### Per-move placement planes

The heads above read whatever position they are given, so the position
evaluation model already yields a candidate's four planes when run on that
candidate's **post-move state** — which is the only form the evidence loop
can use (see [sim evidence](#sim-evidence) for why the root's planes are the
wrong comparison). The move set evaluation model, which scores all `N`
candidates against one board encode, therefore gains **per-move** versions of
the same four planes, distilled from the teacher's masks at each candidate's
post-move state exactly as its value heads are distilled from the teacher's
WLD.

Decoding a 15×15 plane from a per-move vector needs a spatial readout the
scoring path does not otherwise have: the move's attended embedding scores
against each of the 225 board tokens, one such readout per head. The heads
are *read* only for the handful of simmed candidates, but they are *trained*
over the labeled subset, which is where their cost lands
([open questions](#open-questions)).

## Sim evidence

The sim of candidate `M` (`S` rollouts from `M`'s post-move state) yields,
per head, an **empirical map**, plus a **sim value estimate** (empirical WLD)
and the **rollout counts** — a confidence signal, so the network knows how
much to trust maps built from few rollouts.

The map is comparable only to the network's prediction **for `M`'s post-move
state**, not to the root position's planes. The rollouts observe plies played
after `M`; the root planes predict the reply to whatever we end up playing.
Differencing the two would charge `M` with a change it caused — and for a
blocking candidate, causing that change *is* the merit being measured, so the
mismatch erases exactly the signal the loop hunts for. This is what the
[per-move placement planes](#per-move-placement-planes) exist to supply.

Sims across candidates at one position share their random draws (the same
sampled opponent racks; a fixed shuffled bag order consumed as needed) —
**common random numbers (CRN)**, implemented in
[sim_runner.h](../engine/include/sim/sim_runner.h). Rack and draw luck
then cancel in *comparisons* between candidates, which is what the final
pick and the stopping rule ride on.

**Evidence stays paired with its move.** Aggregating the candidates' maps
into shared board-level planes loses which moves suffer a danger — and the
move → does-the-danger-persist mapping is itself lexical (a move can kill a
threat without occupying its squares). The unit of evidence is the
(move, sim-result) pair; aggregation across pairs is left to the network.

## Evidence-set conditioning: the architecture

The re-evaluation is set-conditioned scoring:
`score(M′ | board, {(Mᵢ, sim-resultᵢ)})` for every legal move `M′` (an
attentive-neural-process shape):

- **Evidence tokens.** One per simmed candidate: its move encoding (the move
  set evaluation model's move encoder, reused) fused with its sim
  observations (maps, value, counts) **and the network's own predicted
  placement planes for that candidate**, the two plane stacks concatenated
  channel-wise so the encoder sees observation and prediction for the same
  square side by side. The residual is formed *inside* the encoder rather
  than precomputed, which keeps the raw halves available (a confident
  prediction contradicted by 40 rollouts and an unsure one contradicted by
  2000 have the same difference and warrant different updates).
- **Evidence self-attention.** Contrasts between pairs are the point.
- **Fusion into the board encoding.** The evidence tokens cross-attend with
  the board's spatial map `H`, producing an evidence-conditioned `H′`; the
  per-move scoring machinery is unchanged, attending into `H′` exactly as it
  attends into `H`.

**The predictions have to be inputs; the network cannot recover them.** An
evidence encoder that reads observations alone (the kill-test's
[model.py](../py/scribblez/sim_evidence/model.py), whose fusion is a plain
additive `x + ev_spatial` with the encoder blind to `x`) can express
`posterior = prior + g(observation)` but not
`posterior = prior + k·(observation − prior)`: the second needs a term that
scales the prior down, and nothing downstream of an additive merge can
separate the two summands again. Such a model learns a correction *marginal*
over its own belief states — the same shift whether it had already priced the
danger in or was blind to it, which double-counts confirmations and damps
genuine surprises toward the average. Feeding the predictions in as
channels restores the contrast without disturbing the fusion's placement.

An **empty evidence set** must degrade to the plain one-pass model; training
covers this case explicitly. The fusion stage sits between the shared trunk
and the heads, so the position evaluation model takes the same evidence
through the same stage (needed for the distillation story below).

**Incremental inference.** At one decision point only the evidence set
changes across rounds, so the trunk output `H` and the move encodings are
computed once and cached by the harness; per round, only the fusion stage
and the cheap re-scoring pass run, with outputs bit-identical to a full
recompute. This makes **late fusion a load-bearing constraint**: evidence
must not modulate the trunk's own layers, or the trunk cannot be cached
across rounds.

The predicted planes carried by an evidence token are the **evidence-free
first-pass** predictions, so they cache with the move encodings and a token
never changes once created. Taking them from the conditioned pass instead
would make round `r`'s stored prediction a function of round `r−1`'s
evidence — a token that drifts as the set grows, defeating the caching and
turning the residual into a difference against an already-corrected belief.

A learned recurrent memory (fixed-size state updated as evidence arrives) is
rejected: it is lossy compression, and the candidate that matters most here
is precisely the low-salience one that compression drops first; proposing a
candidate without the move list present requires the network to *generate* a
move, its demonstrated blind spot; and a stateful network makes training
sequential, where the evidence-set formulation keeps training rows
independent.

## The decision procedure

The proposer's job is uniform across rounds: given the sim results for the
candidates simmed so far (possibly none), pick the next candidate(s) to sim.
One generic loop, parameterized by a schedule (`B` candidates proposed per
round, `R` rounds):

1. GADDAG generates all legal moves.
2. Evaluate all of them with the current evidence set (initially empty) and
   propose the top `B` unsimmed candidates.
3. Sim the proposed candidates; append their (move, sim-result) pairs to the
   evidence set.
4. Repeat from 2 until `R` rounds have run (or the sim budget is spent).
5. Final pick: best move by simulation value over all simmed candidates.

The first sim slot is reserved for a **mechanical anchor** — the
highest-raw-score move — regardless of the proposer's ranking. It is cheap
insurance against model blind spots, and its sim is high-value evidence: the
residual on the obvious move calibrates the rest of the evidence set.

The payoff is **promotion, not re-scoring**: simmed moves are ranked by
their sims directly; conditioning matters because the next round can promote
moves no earlier round selected — e.g. the modest play that blocks a
newly-discovered hot spot. Helpfully, opponent hot spots are discovered by
simming *any* candidate that fails to block them, so danger coverage is not
sensitive to the first batch's composition.

### The schedule spectrum

- **(B = K, R = 2)** — two batched rounds. Coarsest evidence conditioning
  (two prefix sizes); works with no acquisition mechanism beyond the value
  ranking plus a footprint-novelty penalty for within-batch diversity.
- **(B = 1, R = K)** — fully sequential. Every sim is maximally informed by
  all prior evidence, and the loop admits early stopping (halt when no
  unsimmed move is likely to prove best); a greedy proposer tends to propose
  near-duplicates of the current best, so an acquisition mechanism
  ([candidate selection](#candidate-selection)) is load-bearing at small
  `B`.
- Training must cover every evidence-prefix size (including zero), so data
  generation records the trajectory that produced each set. A sequential
  trajectory supplies rows at every prefix size.

Wall-clock barely distinguishes the schedules at the intended sim budgets:
rollouts parallelize *within* a candidate, and at hundreds-to-thousands of
rollouts per candidate a single sim saturates the hardware, so total sim
compute is `K·S` either way and sequential rounds add only `R` cheap
fusion-and-rescore passes plus barrier waits. The design center is therefore
**(B = 1, R = K)** — sequential proposal behind the mechanical anchor — with
batch mode retained as the fallback if the sequential proposer fails to beat
it.

## Candidate selection

Suppose we have simmed `N` candidates thus far. We must choose the `(N+1)`st
candidate to sim. How do we select this? This is an exploration problem.

Roughly speaking, we want to sim candidates that are likely to prove best.
A candidate is likely to prove best if:

A. It looks good on its own.
B. It looks different from previous candidates (if it sims identically to some
   previous candidate, it won't appear strictly better than it).

We considered two different approaches:

1. **Covariance modeling.** If we can model the covariance between candidates,
   this helps with B.

2. **Directly modeling "proves-best".** Predict the likelihood that a given
   next candidate will yield a sim that strictly exceeds the best-so-far.

Option 1 faces some challenges. Defining a target that corresponds to
covariance seems difficult. It also only helps with requirement B; it is unclear
how to blend that with requirement A in a principled way. For these reasons, we
favor Option 2.

Two structural facts about the proves-best target:

- **It is a thin transform of the conditioned value.** "Proves best" is
  approximately `P(conditioned value of the candidate + sim noise >
  best-so-far)` — a calibrated comparison of the evidence-conditioned value
  against a known scalar, at a noise level given by the rollout counts. The
  hard part is the conditioned value, which dense distillation trains; the
  head is a thin output on that backbone, fine-tuned on sim outcomes. The
  head cannot be the sole training path: its labels exist only for simmed
  candidates — a handful per position, chosen by the data-generation
  proposer, at thousands of rollout-games each — whereas the distillation
  oracle labels *any* move under *any* evidence prefix at one forward pass,
  unbiased over the full move set.
- **At an empty evidence set it reduces to value ranking.** With best-so-far
  at the floor, the expected-gain form `E[max(0, p(w) − best)]` collapses to
  `E[p(w)]` — the value prediction itself (the probability form instead
  degenerates to 1 for every move). First-round proposal by value score is
  the empty-evidence special case of the acquisition rule, not a separate
  mechanism.

Details to be worked out with Option 2:

- **Probability vs expected gain.** The final pick is by sim value, so what a
  sim contributes is `E[max(0, p(w) − best)]`; the Bernoulli target ignores
  the margin of improvement. Start with the probability (cleaner target),
  keeping the expected-gain variant as a target swap.
- **Winner's curse.** The best-so-far is a max of noisy sim estimates and is
  biased upward, and near-ties make the label a coin flip driven by rollout
  noise. The rollout-count inputs exist for exactly this; the head is
  calibrated to the sim configuration that produced its labels.
- **Policy dependence.** The (evidence set, best-so-far) distribution
  reflects whatever proposer generated the trajectories, so the head trains
  progressively off-policy as the proposer improves — handled generationally,
  like the scorer.
- **Batch diversity at `B` > 1.** The head scores candidates independently,
  so a top-`B` batch can be near-duplicates *of each other*; the footprint
  novelty penalty supplies within-batch diversity. At `B` = 1 the issue
  vanishes.
- **Training rows.** From the simmed candidates a position already has
  (`.sobs`), any evidence prefix plus a held-out simmed candidate is a
  labeled row — combinatorially many (correlated) rows per position, no new
  generation machinery.
- **Scope.** The head only picks the next candidate to sim; the stopping rule
  and the final pick between simmed contenders still ride on the paired
  (CRN) sim estimates.

## Training

### Evidence semantics

The evidence input for a training row is the set of (move, sim-result) pairs
gathered at the decision point — **uniform for every candidate being
scored**, whether or not that candidate is in the set. This keeps
distillation targets well-defined across the whole move set.

### The position evaluation model with evidence

Per labeled position: run the proposal/sim schedule, record the evidence
trajectory, and store the **raw sim observations** alongside the `.slog`
data. Raw observations are model-independent and never go stale; the
network's own predictions (the other half of each evidence token) are
recomputed live at train time — one extra evidence-free forward per simmed
candidate, at the post-move state the replay already reconstructs, so no new
head and no new stored artifact. Targets are unchanged (WLD, ScoreDiff,
conjunction heads). Rows train at multiple evidence-prefix sizes including
**zero** — the zero-evidence rows keep the evidence-free pass from degrading
and are free.

### The move set evaluation model with evidence

Distillation from the evidence-conditioned position evaluation model, as in
Phase 4, with the evidence set present on both sides through the shared
fusion stage; the label-a-subset/mask-the-loss strategy applies unchanged.
The proves-best head is the one output trained directly on sim outcomes
(see [candidate selection](#candidate-selection)); the value heads train by
distillation alone.

The [per-move placement planes](#per-move-placement-planes) distill the same
way, from the teacher's four masks at each labeled candidate's post-move
state, and on the same labeled subset. Once they exist the proposer supplies
its own evidence tokens' predictions, and the position evaluation model's
extra per-candidate forwards are needed only for its own training rows.

### Evidence-trajectory generation

The evidence sets in training data come from running the deployment
schedule itself (anchor slot, then sequential proposal) at labeled
positions, with exploration noise: the proposer samples from a
temperature-softmax over its proposal scores rather than taking the argmax,
and the trajectory length is randomized to spread coverage across prefix
sizes and best-so-far levels. Uniform-random candidates are poor
exploration — with thousands of legal moves nearly all terrible, a uniform
sim spends its rollouts on evidence about moves no proposer will ever
propose; the region needing coverage is the plausible-but-not-top tail that
temperature sampling reaches. Generation 0, with no trained proposer, uses
the hasty-equity top-`K`.

This exploration is bias-free with respect to outcome targets: evidence
labeling is a side-computation on positions from ordinary self-play, so the
choice of which candidates to sim never alters the played move or the game
outcome — unlike move-sampling diversification, which changes the
trajectory itself. The only cost is sim compute spent on less informative
evidence. Coverage matters most for the proves-best head, whose labels
exist only for simmed candidates; temperature exploration is what puts
labels on the "not the top pick, but proved best" rows that head exists to
predict.

### The cost elephant

Evidence-carrying rows require running the sims at data-generation time —
thousands of rollout-games per labeled position, a ~10³–10⁴× slowdown over
plain generation. Mitigations, all compatible: label a sparse subset of
positions (the rest train with empty evidence, needed anyway); cheap sims
(small `S` is a soft degradation — the counts tell the network the noise
level); and the generational pipeline
([generational_training.md](generational_training.md)), which exists for
exactly this reuse pattern.

## Limitations and caveats

- The sim is not unbiased ground truth: the evidence mixes lexical blindness
  (expected to dominate — it is the one systematic network-vs-rollout gap)
  with rack-sampling mismatch, rollout-policy weakness, and Monte Carlo
  noise.
- Self-created opportunities stay out of reach: a move whose value exists
  only in structure it creates, and which no round proposes, is never
  simmed. That recall gap belongs to the lexical input features.
- In HastyBot self-play the logged reply and the rollouts share a policy, so
  the conjunction-head loss improves for a shallow reason; the metric that
  matters is WLD / calibration, not the spatial-head loss.
- Reply footprints are ~2–7 of 225 squares, so most squares are
  variance-dominated at practical `S`; genuine hot spots recur across
  rollouts, and the count inputs let the network discount the rest.
- Evidence-map encoding is an open design point: pooled vector (cheap) vs
  spatial planes (preserves the *where*). Start pooled.

## De-risking: the kill-test

The load-bearing hypothesis: *conditioning on sim evidence improves the
value model's outcome prediction.* Tested offline: the evidence-conditioned
position evaluation model vs the plain baseline on identical data, compared
on held-out WLD loss and calibration.

**Status: done — passed.** Evidence gain of −0.0063 CE at 5.7 SE with clean
controls; magnitude bounded by root-readout saturation, and an 8×
late-vs-early phase gradient supports the mechanism. Full numbers and
conclusions: [sim_obs_experiment_results.md](sim_obs_experiment_results.md).

The pipeline is [sim_obs_tool](../engine/apps/sim_obs_tool.cpp) (candidates
are the HastyBot-equity top-K, so each position's evidence contains the
played move's own sim) feeding [kill_test.py](../py/scripts/kill_test.py);
the evidence-conditioned model is
[sim_evidence/model.py](../py/scribblez/sim_evidence/model.py), a
zero-initialized fusion stage on the regular post-move model, so the arms
are parameter-identical and differ only in their inputs.

```
# Generates self-play data + sim observations until stopped; resumable.
./py/scripts/generate_kill_test_data.py -t apple

# 4-armed test; can run (and rerun) while generation continues.
./py/scripts/kill_test.py -t apple
```

Data accumulates under `<mount>/tags/kill_test/<tag>/data/slogs` (`.slog`
batches plus `.sobs` sidecars, written atomically). Reading the results
(per-arm history in `<mount>/tags/kill_test/<tag>/cache/results/<arm>.json`;
decision metric is best held-out `wld_ce`):

- **`full` < `none`** by a margin that dwarfs seed noise → the hypothesis
  survives.
- **`shuffled` ≈ `none`** is the validity check (shuffled evidence carries
  the same marginals, position-mismatched); measure `full`'s gain against
  `shuffled`.
- **`scalar` vs `full`** locates how much of the win needs the spatial
  planes.
- Evidence arms are **leave-one-out** by default — the played move's own sim
  is masked, because deployment only ever re-scores *unsimmed* moves, so LOO
  gains are the transfer gains that matter. The optional `ownsim` arm
  (`--arms ownsim`) prices that shortcut.

### The face-up-leaves mode

`--open-leaves` on both commands (under a dedicated tag) runs the same
experiment in **face-up-leaves Scrabble**: the tiles a player retained from
their last move are public, replenishment draws stay hidden. This is the
variant the project now develops in ([roadmap.md](roadmap.md)), so it is the
mainline mode rather than an instrument; the hidden arm remains runnable and
the gap between the two is what belief would have to close. Compare arm deltas
within a mode only.

Mechanics: the model input gains the opponent-leave counts block
(`kOppLeaveCounts`, input_encoder.h); the leave is derived at replay time
from the `.slog` draws (no game-runner changes); the `.sobs` header records
the condition, so mixing modes within a tag fails loudly.

```
./py/scripts/generate_kill_test_data.py -t apple-open --open-leaves
./py/scripts/kill_test.py -t apple-open --open-leaves
```

## Implementation roadmap

| Step | Build | Depends on | Status |
|------|-------|-----------|--------|
| 1 | Conjunction heads on the position evaluation model (targets from logs; per-square BCE). Independent value as probes even if the loop is never built. | — | **Done** — `opp_win_placement` / `self_win_placement`, plus the `self_next_placement` marginal so both conjunctions have an occupancy partner, through the full pipeline (target registry, decoder, FFI, model heads + BCE losses, ONNX export, TensorRT binding, dashboard loss series). |
| 2 | Sim machinery emits per-square empirical maps + value estimates + counts; **common random numbers across candidates at a position**; storage format for sim observations alongside `.slog`. | 1 | **Done** — [sim_runner.h](../engine/include/sim/sim_runner.h) (CRN rollouts over PLAY/EXCHANGE/PASS candidates, count planes mirroring the placement-mask targets, W/D/L + delta moments) and [sim_observation_log.h](../engine/include/data/sim_observation_log.h) (the versioned `.sobs` sidecar). |
| 3 | **Kill-test** (above): evidence-conditioned position evaluation model vs. baseline. **Go/no-go gate for everything below.** | 2, the eval machinery | **Done — passed** (see above). |
| 4 | Evidence encoder + fusion stage in the shared trunk, with tokens carrying the model's post-move placement planes beside the observed maps (extra evidence-free forwards, no new head); multi-prefix-size training; evidence labeling integrated into generational data generation at a sparse position fraction. | 3 | — |
| 5 | The move set evaluation model inherits the heads and the fusion stage, and gains per-move placement planes (spatial readout against the board tokens) so it predicts its own evidence tokens' half; distillation from the evidence-conditioned position evaluation model. | 4, roadmap track A | — |
| 6 | The sequential agent (the decision procedure above at `B = 1, R = K`) and the proves-best acquisition head that drives it; budget tuning and the early-stopping threshold. Batched multi-round scheduling is the fallback, not a step on the way ([roadmap.md](roadmap.md)). | 5 | — |

## Open questions

- **Budget split** — whether later sims should use smaller `S`, and the
  early-stopping threshold for the sequential schedule.
- **Proves-best head details** — the bulleted list under
  [candidate selection](#candidate-selection).
- **Whether the spatial machinery pays at all.** Settled against the cheap
  option so far: the kill-test's `full` arm matched its `scalar` arm to
  ±0.0003, so at a root-WLD readout the planes are inert
  ([sim_obs_experiment_results.md](sim_obs_experiment_results.md)) — as
  expected, since a position-level scalar has no use for per-move spatial
  discrimination. The plan above nonetheless commits to spatial, per-move,
  prediction-paired evidence, because the effect it is built for is
  *promotion* — a move no earlier round ranked highly rising once a hot
  square is exposed — which the root readout structurally cannot exhibit.
  That commitment is a bet. It is settled after the build, by the
  placement-plane ablation in [evaluation_plan.md](evaluation_plan.md) —
  evidence tokens with and without the model's predicted planes, read at
  promotion rather than at root WLD; a null there sends the loop back to the
  scalar rung, not just back a step.
- **The cost of per-move placement targets.** A `.mset` record is 5 floats
  today; four 15×15 planes add 900. The format is head-extensible by design
  (`record_floats` in
  [move_set_eval_target_log.h](../engine/include/training/move_set_eval_target_log.h)),
  so this is a size question, not a format one — but a 180× record blow-up
  wants a decision on quantization (probabilities fit a byte), sparsity
  (only the hot squares carry signal — reply footprints are ~2–7 of 225), or
  labeling the planes on a narrower subset than the value heads use.
- **Sim reuse across rounds** — candidates retained across rounds keep their
  rollouts; whether to top up counts as the evidence set grows.

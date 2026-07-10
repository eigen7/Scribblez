# Sim-residual feedback: re-evaluating moves with Monte Carlo evidence

## Purpose

Move selection (roadmap Phase 4) runs the move set evaluation model over all
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

- **Roadmap Phase 4** ([roadmap.md](roadmap.md)): wraps the one-round
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

## Sim evidence

The sim of candidate `M` (`S` rollouts from `M`'s post-move state) yields,
per head, an **empirical map** directly comparable to the network's
prediction for `M`, plus a **sim value estimate** (empirical WLD) and the
**rollout counts** — a confidence signal, so the network knows how much to
trust maps built from few rollouts.

Sims across candidates at one position share their random draws (the same
sampled opponent racks; a fixed shuffled bag order consumed as needed) —
**common random numbers (CRN)**, implemented in
[sim_runner.h](../engine/include/selfplay/sim_runner.h). Rack and draw luck
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
  observations (maps, value, counts) plus the network's own first-pass
  predictions for it — handing the network the residual contrast directly.
- **Evidence self-attention.** Contrasts between pairs are the point.
- **Fusion into the board encoding.** The evidence tokens cross-attend with
  the board's spatial map `H`, producing an evidence-conditioned `H′`; the
  per-move scoring machinery is unchanged, attending into `H′` exactly as it
  attends into `H`.

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

A learned recurrent memory (fixed-size state updated as evidence arrives) is
rejected: it is lossy compression, and the candidate that matters most here
is precisely the low-salience one that compression drops first; proposing a
candidate without the move list present requires the network to *generate* a
move, its demonstrated blind spot; and a stateful network makes training
sequential, where the evidence-set formulation keeps training rows
independent.

## The decision procedure

One generic loop, parameterized by a schedule (`B` candidates proposed per
round, `R` rounds):

1. GADDAG generates all legal moves.
2. Evaluate all of them with the current evidence set (initially empty) and
   propose the top `B` unsimmed candidates.
3. Sim the proposed candidates; append their (move, sim-result) pairs to the
   evidence set.
4. Repeat from 2 until `R` rounds have run (or the sim budget is spent).
5. Final pick: best move by simulation value over all simmed candidates.

The payoff is **promotion, not re-scoring**: simmed moves are ranked by
their sims directly; conditioning matters because the next round can promote
moves no earlier round selected — e.g. the modest play that blocks a
newly-discovered hot spot. Helpfully, opponent hot spots are discovered by
simming *any* candidate that fails to block them, so danger coverage is not
sensitive to the first batch's composition.

### The schedule spectrum

- **(B = K, R = 2)** — two batched rounds. Simplest data generation and
  training distribution; batch diversity for free.
- **(B = 1, R = K)** — fully sequential. Every sim is maximally informed by
  all prior evidence; latency serializes on the sims (the network side is
  one trunk pass plus `R` cheap incremental passes); a greedy proposer tends
  to propose near-duplicates of the current best, so an acquisition
  mechanism ([candidate selection](#candidate-selection)) is load-bearing at
  small `B`.
- Training must cover every evidence-prefix size (including zero), so data
  generation records the trajectory that produced each set.

The recommended starting point is **(B = K, R = 2)**, moving toward smaller
`B` only if measurement shows targeted sims beat batch diversity.

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
recomputed live at train time. Targets are unchanged (WLD, ScoreDiff,
conjunction heads). Rows train at multiple evidence-prefix sizes including
**zero** — the zero-evidence rows keep the evidence-free pass from degrading
and are free.

### The move set evaluation model with evidence

Distillation from the evidence-conditioned position evaluation model, as in
Phase 4, with the evidence set present on both sides through the shared
fusion stage; the label-a-subset/mask-the-loss strategy applies unchanged.
(Training later rounds directly against sim values instead is coherent but
departs from the Phase 4 pipeline; open question.)

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

### The open-leaves information condition

`--open-leaves` on both commands (under a dedicated tag) runs the same
experiment in **open-leaves Scrabble**: the tiles a player retained from
their last move are public, replenishment draws stay hidden. The leave is
the Bayesian-inferable part of a rack, so this hands the model and the sims
an exact rack posterior — the exact endpoint of the belief-quality axis. The
open-leaves gain minus the hidden gain prices the entire belief line of work
interventionally. A research instrument, not the product path; compare arm
deltas within a mode only.

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
| 2 | Sim machinery emits per-square empirical maps + value estimates + counts; **common random numbers across candidates at a position**; storage format for sim observations alongside `.slog`. | 1 | **Done** — [sim_runner.h](../engine/include/selfplay/sim_runner.h) (CRN rollouts over PLAY/EXCHANGE/PASS candidates, count planes mirroring the placement-mask targets, W/D/L + delta moments) and [sim_observation_log.h](../engine/include/selfplay/sim_observation_log.h) (the versioned `.sobs` sidecar). |
| 3 | **Kill-test** (above): evidence-conditioned position evaluation model vs. baseline. **Go/no-go gate for everything below.** | 2, Phase 3 eval machinery | **Done — passed** (see above). |
| 4 | Evidence encoder + fusion stage in the shared trunk; multi-prefix-size training; evidence labeling integrated into generational data generation at a sparse position fraction. | 3 | — |
| 5 | The move set evaluation model inherits the heads and the fusion stage; distillation from the evidence-conditioned position evaluation model. | 4, roadmap Phase 4 | — |
| 6 | Multi-round agent (the decision procedure above); schedule tuning (`B`, `R`); acquisition — footprint novelty penalty first, then the proves-best head if the small-`B` schedule shows value; match-play eval vs. the one-round agent. | 5 | — |

## Open questions

- **Schedule (`B`, `R`) and budget split** — including whether later rounds
  should use smaller `S`, and where on the acquisition ladder (novelty
  penalty → proves-best head) each schedule needs to sit.
- **Proves-best head details** — the bulleted list under
  [candidate selection](#candidate-selection).
- **Evidence-map encoding** — pooled vs spatial; cheap to ablate inside the
  kill-test.
- **Whether scalar evidence alone captures most of the win** — if so, the
  spatial machinery can be deferred (cheap-before-rich).
- **Later-round training targets** — evidence-conditioned distillation vs
  training directly against sim values for simmed candidates.
- **Sim reuse across rounds** — candidates retained across rounds keep their
  rollouts; whether to top up counts as the evidence set grows.

# Scribblez Project Roadmap

Scribblez aims to beat existing Scrabble engines by replacing their
context-blind static evaluation and naive rack inference with learned,
belief-aware evaluation ([design.md](design.md)). This document is the
**implementation plan**: the agent we are building, what already exists, what is
left to write, and the models that have to be trained to feed it.

It deliberately contains no experiments. Measurement — what has been established,
and how the finished agent gets evaluated — lives in
[evaluation_plan.md](evaluation_plan.md) and happens once the build is done. The
plan below is committed to, not gated: every component listed is part of the
destination agent, and nothing here exists to decide whether to build something
else.

Legacy track labels (A2, A3, C2, D1, …) are kept where code comments and other
docs already reference them, but the ordering below is implementation order, not
track order.

## The variant: face-up leaves

Development happens in **face-up-leaves Scrabble**, where each player reveals
their leave after every turn and only the replenishment draws stay hidden.
Symmetrically: both seats see, and both may use, the other's retained tiles.

Rack uncertainty is the dominant confound in everything downstream, and removing
it by rule lets the effort go where the novelty is: the move set evaluation
model, evidence conditioning, and sim scheduling. None of those components are
specific to an information condition, so returning to standard Scrabble later is
a data regeneration rather than a redesign.

What this parks is the *belief* half of the thesis in [design.md](design.md).
That is a sequencing decision, not a retraction; see [rack inference](#rack-inference--parked).

## The destination

The agent this plan builds, per turn:

```
GADDAG generates all N legal moves
      │
      ▼
move proposal model scores all N in ONE pass
  → per-candidate WLD, score differential, and placement planes
      │
      ▼
sim the highest-SCORING move (the greedy anchor)   ← model-independent
      │
      ▼
  ┌─→ append (move, sim observation, that move's predicted planes)
  │        to the evidence set
  │   │
  │   ▼
  │  fusion stage + proves-best head re-score every UNSIMMED candidate
  │   │   (trunk and move encodings cached; only fusion re-runs)
  │   ▼
  │  sim the argmax proves-best candidate
  └───┘  repeat until the sim budget is spent, or no unsimmed
         candidate's predicted gain clears the stopping threshold
      │
      ▼
play the best simmed candidate by simulation value
```

The model in the loop is the **move proposal model** — a copy of the move set
evaluation model carrying the fusion stage and the proves-best head
([models](#models-and-how-they-are-trained)). The plain student never runs at
the root; its readouts reach the loop through the copy's anchored plain pass.

Rollouts inside the loop are value-truncated at a lexically sufficient
horizon ([item 2](#2-value-truncated-rollouts-d1)) and climb the rest of the
[policy ladder](#7-self-model-plies-and-the-endgame-solver-d2-d3): self-model
plies, then the endgame solver once the bag empties.

**The first sim is a mechanical anchor**, not a model choice: the
highest-raw-score move, as the greedy agent would pick it, regardless of how the
model ranks it. Two reasons, both from
[sim_residual_feedback.md](sim_residual_feedback.md). It is cheap insurance
against model blind spots — the one candidate guaranteed to be simmed is chosen
by a rule the model cannot be wrong about. And its sim is unusually informative
evidence: the residual on the obvious move calibrates the rest of the evidence
set, which a pick correlated with the model's own errors would not do.

**The loop is sequential by design.** Every sim is informed by all prior
evidence, and if the agent sims `N` candidates it queries the proves-best head
`N − 1` times — the anchor needs no query, and every pick after it is
evidence-conditioned. This is `(B = 1, R = K)` in
[sim_residual_feedback.md](sim_residual_feedback.md)'s schedule spectrum, which
that document already identifies as the design center; batched multi-round
variants are a fallback, not a step on the way.

The serial queries are free. One full-candidate-set forward pass costs **0.37 ms**
at M = 4000 ([model_specs.h](../engine/include/nn/model_specs.h), measured),
against **~16.8 thread-seconds** of rollouts per turn at K=10 × 400 (measured)
— and the deployment budget is nearer ~1,000 rollouts per candidate (~42
thread-seconds by that scaling), which is what makes value truncation
([item 2](#2-value-truncated-rollouts-d1)) and early stopping load-bearing.
Ten sequential passes are four orders of magnitude below the rollouts they
schedule.

**Promotion, not re-scoring, is the payoff.** Simmed candidates are ranked by
their own sims; conditioning matters because the loop can promote a candidate no
earlier round would have picked — the modest play that blocks a hot spot the
sims just revealed.

**No upfront candidate filtering.** `N` ranges from 1 to 10,000+ (blanks), and
collapsing near-duplicate blank designations before scoring is both unnecessary
and risky: the single linear pass makes large `N` a non-problem, differing blank
letters produce genuinely different crosswords and hooks, and an upfront filter
risks dropping exactly the move the model exists to find. Every legal move is
scored, every iteration. The `O(N)` argument assumes full-set evaluation happens
once per decision, at the root — a neural *rollout* policy prunes instead, for
the separate reasons under [D2](#7-self-model-plies-and-the-endgame-solver-d2-d3).

## What is already built

- **The position evaluation model** — the teacher. Evaluates a post-move,
  pre-draw board from the mover's POV: WLD, a Gaussian over the final score
  differential, and four 15×15 placement masks (opponent/self next-move
  occupancy, each conjoined with that player winning). Trained on HastyBot
  self-play under the generational lifecycle
  ([architecture.md](architecture.md),
  [generational_training.md](generational_training.md)).
- **The move set evaluation model** — the student. Board trunk once, one
  cheap vector per candidate, cross-attention scoring all `N` in one pass
  ([model_architectures.md](model_architectures.md)); v2 carries the per-move
  placement planes ([move_set_eval_v2_results.md](move_set_eval_v2_results.md)).
- **Target generation** (A2) — the `.mset` sidecar, its generator, and the
  `move_set_eval` dashboard workload, running in-variant with a frozen
  hash-stamped teacher.
- **Engine inference** — the move-set arm of `NeuralNet<Spec>` and its
  evaluation service ([model_specs.h](../engine/include/nn/model_specs.h)), the
  P=1 ONNX export
  ([onnx_export.py](../py/scribblez/move_set_eval/onnx_export.py)), and the
  `--type=mset-sim` agent
  ([mset_sim_agent.h](../engine/include/agent/mset_sim_agent.h)): scores a
  turn's whole candidate set in one pass and sims the model's top K. This is the
  destination agent minus the evidence loop.
- **Sim machinery** — [sim_runner.h](../engine/include/sim/sim_runner.h) runs
  common-random-number rollouts;
  [sim_observation_log.h](../engine/include/data/sim_observation_log.h) stores
  them in `.sobs` sidecars.
- **Evidence machinery** — the fusion stage and its exactness tests
  ([evidence_fusion.py](../py/scribblez/evidence_fusion.py)), the proves-best
  head, the trajectory generator and `evidence_trajectories` workload, and the
  evidence trainer (`py/scribblez/evidence/`). Items 4–5 revise how these are
  fed and trained, not what they are.
- **The sim agent baseline** and the endgame solver, plus face-up leaves in the
  game loop.
- **Infrastructure** — the master dashboard and workload registry, the
  match harness (A1/E2), and the cloud fleet (E1).

## What needs implementing

In dependency order. Each item is part of the destination agent.

### 1. Per-move placement planes

**Done** — the format, readouts, regenerated corpus, and trained student v2
all exist; [move_set_eval_v2_results.md](move_set_eval_v2_results.md) records
the run and its gate metrics.

The four placement maps the position evaluation model already predicts, but
predicted **per candidate**, for that candidate's post-move state. The scoring
path holds one vector per move, so decoding a 15×15 map means scoring that
vector against the 225 board tokens, one readout per head.

These are the quantity sim evidence is differenced against, so they gate
everything below.

- **Model**: four per-move readouts on the move set evaluation model.
- **Targets**: the teacher's own masks at the same post-move states — the
  generator already runs the teacher there for the value targets.
- **Format** (settled, `.mset` v2): each stratified candidate record carries the
  four planes dense and absmax-quantized — per plane a float32 scale
  (max/255) plus 225 bytes, ~950 B per record against 36 B for v1 and ~3.6 KB
  for float32 planes. Dense-over-sparse because the masks are sigmoid outputs
  (near-zero, not zero, so a nonzero-mask encoding saves an unreliable amount),
  and fixed-size records keep both readers' vectorized indexing; full-sweep
  files stay plane-less (evaluation-only, value-based metrics).
- **Consequence**: the existing 600-pair corpus cannot be reused. Regeneration
  is required, which is why the format decision comes first.

### 2. Value-truncated rollouts (D1)

Promoted from the rollout-policy ladder to the head of the queue: truncation
is now a **data prerequisite**, not just an agent speedup.

Rollouts sim a few plies, then read the position evaluation model's value at
the horizon. An **anchor fraction** of terminal rollouts per candidate stays
as a ground-truth tether — and as the instrument that measures both what the
leaf model hides and the truncation bias in the CRN duplicate cancellation
([sim_residual_feedback.md](sim_residual_feedback.md)).

- **Why first**: the deployment budget is ~1,000 rollouts per candidate, and
  the trajectory corpus must carry deployment-quality evidence — the count
  inputs let the model discount noisy maps, but a head trained only on
  200-rollout evidence has never seen the maps it will be asked to trust.
  Against the v1 recipe that is ~5× the rollouts per sim and ~3× the sims
  per position; truncation is what makes the regeneration affordable. The
  kill-test's phase gradient
  ([sim_obs_experiment_results.md](sim_obs_experiment_results.md)) says the
  same move also *cleans* the evidence: truncation manufactures
  late-game-quality, low-variance observations at every phase.
- **Horizon**: deep enough for the lexical contingencies the loop hunts to
  resolve. A contingent draw is realized as draw-then-play — plies 2–3 after
  the candidate — so the horizon is ≥ 3–4 plies; shallower, and the horizon
  is handed straight back to the lexically blind leaf model, leaving the sim
  nothing to observe that the model did not already know.
- **Costs to accept** (unchanged from the ladder): `.sobs` artifacts become
  model-versioned, and sims start contending for the GPU.

### 3. Engine runtime for the evidence path

- **ONNX export** of the move proposal model, split so the cached parts
  (trunk, move encodings, evidence-free predictions) are computed once per
  turn and only the fusion stage plus re-scoring run per loop iteration.
  Outputs must be bit-identical to a full recompute.
- **Engine-side evidence staging**: `SimObservation` → model input, alongside
  the stored per-candidate predicted planes.
- Extends `MoveSetEvaluationSpec` or lands as a second spec beside it.

Moved ahead of data generation: the revised recipe's on-policy side (item 4)
*is* the deployment loop, so the generator needs this runtime before the
corpus can be made.

### 4. Evidence-trajectory generation

The data for item 5. **Machinery built, recipe revised** — the v1 chain
(self-play → trajectories → labeling as the `evidence_trajectories`
workload's generate role:
[evidence_trajectory_generator](../engine/apps/evidence_trajectory_generator.cpp)
writing trajectory `.sobs` v2, `--sobs` force-inclusion in the `.mset`
labeling) is implemented end to end and produced the 200-rollout corpus the
frozen trial consumed. The recipe below replaces v1's — a temperature
softmax over the student's top 64 plus one uniform tail sim; regeneration
waits on items 2–3.

Per labeled position, one simmed **pool**, every sim at the deployment
rollout configuration (truncated per item 2, CRN across the pool):

- **The anchor** — the highest-raw-score move, exactly as deployed.
- **A ≈ 15 on-policy picks** — the deployment loop itself: iterative
  proves-best proposals conditioned on the sims so far, with temperature in
  the proposal argmax for exploration. Generation 0, with no trained gain
  head, selects this side by a temperature softmax over the plain student's
  values on the **full** candidate set. No proposal-pool cap in any
  generation: deployment argmaxes over every unsimmed candidate, and a cap
  keeps deep promotions out of the corpus (the egotize-lane set's
  structurally unsimmable GAVE was the exhibit).
- **B ≈ 5 off-policy draws** — held out from evidence sets by construction,
  stratified the way the `.mset` sampler is: a uniform draw plus draws from
  the strata where surprising-but-good moves live (low-score / high-leave,
  setups, exchanges). This is the bounded floor against the proposer's echo
  chamber; the rationale is in
  [sim_residual_feedback.md](sim_residual_feedback.md).

Training rows are **assembled from the pool, not replayed from it**: the
evidence set is permutation-invariant and the gain label is a max over the
set, so any subset of {anchor} ∪ A that contains the anchor and fits the
deployment sim budget is a valid evidence set, and every pool member outside
it is a labeled held-out row (gain measured against the CRN max over the
set). One pool yields combinatorially many rows — the right response to sims
that now cost ~1,000 rollouts each. The B draws never enter an evidence set:
deployed evidence holds only the anchor and proposer picks, so keeping the
floor labels-only buys its coverage at zero input-distribution cost.

- The same tool reads hand-maintained `.gcg` position sets (`--gcg`,
  [positions/NWL23/face-up-trajectory-set](../positions/NWL23/face-up-trajectory-set/README.md))
  for the exhibits and the position-set metric of item 5: the dashboard's
  Trajectories tab ([react_dashboard.md](react_dashboard.md#the-evidence-trajectories-trajectories-tab))
  replays a set position through any checkpoint at every evidence-set size,
  and the trainer charts the set's sim-best rank (`posset_*`) per pass.
- The value-labeled `.mset` subset must always include the position's simmed
  pool, the way the mset sampler always includes the played move — otherwise
  dense value labels stay at the static strata's rate while the proposer
  explores elsewhere.

### 5. The move proposal model

The evidence consumer, and the model the sequential agent runs at the root:
a **copy** of the move set evaluation student — trunk, move encoder, value
and plane heads, fusion stage — plus the **proves-best head**, predicting
the expected improvement `E[max(0, v − best-so-far)]` a candidate's sim
would contribute over the best simmed so far. This is the acquisition
function that drives the loop; the expected-gain form (not probability), its
CRN pairing, and the truncation caveat are settled in
[sim_residual_feedback.md](sim_residual_feedback.md#candidate-selection).

A separate model rather than new heads on the student, so that each keeps
one job and one lifecycle: the student stays a pure distillation vessel (the
dense prior, the backbone source, and — under D2 — the rollout policy),
while the copy is free to follow the sim signal.

Trained on item 4's assembled rows, three loss components:

- **Gain** (primary): Huber against the held-out candidate's CRN-paired gain
  over its evidence set's best.
- **Conditioned WLD / score-diff** (auxiliary): soft-CE / Huber against the
  held-out candidate's own sim outcome, on the same rows. Sim outcomes and
  never the plain teacher, whose readout is a function of the board alone —
  as a target on evidence-bearing rows it would train the fusion stage to
  ignore evidence. The gain is a thin transform of the conditioned value, so
  these auxiliaries feed the head at no extra sim cost.
- **Self-distillation anchor** (stabilizer, small coefficient): the plain
  pass pressured toward the **frozen student's outputs**, recomputed on the
  fly by a frozen-student forward over **all** legal candidates of the
  batch's positions — no stored labels, full-set coverage, extendable to
  sim-less positions if drift shows up. The anchor is what keeps the gain
  argmax safe over the thousands of candidates that never receive a sim
  label: the backbone must not drift off the teacher-shaped ranking on the
  strength of a handful of sim rows. Plain pass only — pressuring
  conditioned outputs toward a board-only function is the
  ignore-the-evidence failure again.

The gen-1 **frozen-backbone trial** (fusion stage + head only, over the
200-rollout v1 corpus) is the recorded floor this regime replaces:
conditioned − plain soft-CE −0.0008, acquisition hit rate 0.57 against the
plain value's 0.61. The mechanism demonstrably engages on exhibits, but
200-rollout evidence is too noisy — and a zero-initialized fusion stage over
a frozen trunk too weak — to beat the plain ranking. Both diagnoses are
addressed above: deployment-count rollouts (item 2), and a trainable
backbone held by the anchor.

Two fusion refinements to validate during this build (cheap, unimplemented):
evidence reaches the global summary by mean pooling, but best-so-far — the
one statistic the gain head must know — is a max, so mean+max (the trunk's
own pooling convention) or attention pooling with a learned query; and a
direct move-query → evidence-token cross-attention (`O(N·K)`), letting a
candidate compare itself to each simmed move by encoding rather than only
through shared board squares — a leave-twin with a different footprint is
currently visible only through the move scalars and the pooled summary.

### 6. The sequential agent

The loop from [the destination](#the-destination), as a new `--player
--type=`. Reuses `mset_sim_agent`'s candidate generation, encoding, and
endgame handoff, and shares its loop machinery with item 4's generator (the
generator's on-policy side is this loop); what is left here is the
packaging, the budget, and the stopping rule.

- **First sim: the greedy anchor** — the highest-raw-score candidate, taken
  straight off the generated move list, not from the model's ranking. It is
  the one pick in the turn that does not depend on a network.
- **Every later sim**: argmax of the proves-best head over the unsimmed
  candidates, conditioned on the evidence so far.
- **Early stopping**: halt when no unsimmed candidate's predicted gain
  clears a threshold. At ~1,000 rollouts per sim this is where a budget
  saving turns directly into strength per second.
- **Final pick**: best simmed candidate by simulation value.

### 7. Self-model plies and the endgame solver (D2, D3)

The rest of the rollout-policy ladder (D1 moved to item 2). Each rung
changes sim semantics, so each lands behind a `.sobs` flag.

- **D2 — self-model plies.** Plies 1–2 played by our own stack over racks
  built from the public leave, then HastyBot to the horizon. This beats a
  generic policy upgrade because the evidence maps read *exactly* plies 1–2.
  Inside rollouts the model scores only the top-`k` by static equity for
  small fixed `k` — nothing is cached across plies, and fixed `k` gives
  static tensor shapes for batching plies across concurrent rollouts. D2 is
  also what closes the loop AlphaZero-style: once rollouts play with the
  model, sim quality — and every label derived from it — improves with each
  generation. Until then the sim is a fixed HastyBot oracle, and training
  distills it.
- **D3 — endgame solver for late-game rollouts.** A port of Macondo's
  negamax solver into the engine (the WMP precedent); rollouts switch to it
  when the bag empties, with budget scaled by the root's distance to the
  end.

D2 before D3: D2 depends on the trained student and carries the generational
payoff; D3 is the largest port with the most localized payoff.

### 8. Cloud generation

The generate role is GPU and local-only until the cloud fleet can host TensorRT
— the GPU-workloads item in [cloud_compute.md](cloud_compute.md), which also
needs a way to ship the teacher model to pods. Both corpus regenerations above
want it.

## Models and how they are trained

Three networks plus a spin-off copy, trained in this order; each depends on
the one above it.

### The position evaluation model (teacher)

- **Trained on**: HastyBot self-play `.slog` data, generational
  generate→train ([generational_training.md](generational_training.md)).
- **Planned second target stream**: sim values. A simmed candidate's `.sobs`
  record is a many-rollout estimate of the value at that candidate's
  post-move state — under face-up leaves the sim samples the same draw
  distribution the game did, so it is the same target as the game outcome at
  a fraction of the variance. Positions with sims train on both streams.
- **Predicts**: WLD, score-differential Gaussian, four placement masks.
- **Roles**: teacher for the student's distillation and, once item 2 lands,
  the rollout leaf evaluator — which is what puts it inside the generational
  improvement loop (stronger self-play → better value → better leaves →
  better sims → better labels).
- **Status**: trained and in use. Advancing it by promotion rather than by a
  new tag and full regeneration is
  [generational_teacher.md](generational_teacher.md), still deferred.

### The move set evaluation model (student)

- **Trained on**: `.mset` sidecars — the teacher's readouts at each
  candidate's post-move state (WLD, score differential, the four placement
  planes), paired with pre-move board inputs reconstructed by replay.
  Distillation only: the student carries no sim-outcome losses, and the
  fusion stage it hosts in code trains only in its move-proposal copy.
- **Predicts**: per candidate, WLD + score differential + the four placement
  planes.
- **Roles**: the dense prior over full candidate sets; the backbone the move
  proposal model is copied from and anchored to; under D2, the rollout
  policy.
- **Status**: v2 (with planes) trained
  ([move_set_eval_v2_results.md](move_set_eval_v2_results.md)).

### The move proposal model

- **Is**: the student copy plus the proves-best head
  ([item 5](#5-the-move-proposal-model)); the model at the root of the
  deployed loop.
- **Trained on**: evidence-set rows assembled from item 4's pools, under the
  three-part loss of item 5 — gain first, sim-outcome auxiliaries,
  self-distillation anchor.
- **Bootstrapping**: the gen-0 pool's on-policy side is selected by the
  plain student (temperature softmax over the full candidate set) — correct
  at the empty evidence set, and the greedy anchor supplies the first sim
  regardless of proposer. Later generations select with the current move
  proposal model, and each new student generation refreshes the copy's
  starting point and anchor target.

## Rack inference — parked

Face-up leaves removes the need to infer anything, so this is dormant until the
project returns to standard Scrabble.

What exists: a port of the algorithm behind Macondo's `SIMMING_INFER_BOT` — the
hypergeometric prior over draws from the unseen pool, a temperature-softened
static-equity likelihood, exhaustive enumeration of small leave spaces with
importance sampling above them, and the posterior a simulation would sample
racks from ([belief/rack_inference.h](../engine/include/belief/rack_inference.h)).
Tested; its one consumer is offline: the hidden-leaves Monte-Carlo ground truth
of the position-evaluation test sets samples the opponent's leave from this
posterior ([sim/monte_carlo_sim.h](../engine/include/sim/monte_carlo_sim.h)),
at the default (Macondo) temperature — nothing in play uses it.

Resuming means pricing the posterior against ground truth (a `.slog` replay
recovers the leave the opponent actually held), which is also what sets the
likelihood temperature, then wiring it into `SimRunner`, whose per-rollout-index
sampling already preserves common random numbers. Beyond that lies the learned
belief system of design.md §3.

## What is deliberately not here

- **Batched multi-round scheduling.** The sequential loop subsumes it; batch
  mode returns only if sequential proposal underperforms it.
- **Interim diversity heuristics** (the old C1 footprint/lane-overlap penalty),
  and footprint dedup at sim-selection time. Redundancy is handled twice over
  without them: the expected-improvement target rates a CRN-duplicate at ~0 by
  construction, and evidence conditioning propagates a disappointing sim to
  every candidate sharing its blind spot, since corrections are written onto
  board squares and near-duplicates attend to the same ones. A hand-built
  novelty penalty is both redundant and lossier — it can only express
  "identical footprint or not", where the model grades similarity by degree.
- **A frozen-backbone move proposal model.** The gen-1 frozen trial stays in
  the trainer as the diagnostic it was and in the record as the floor
  ([item 5](#5-the-move-proposal-model)); the plan trains the backbone,
  held by the self-distillation anchor.
- **Backtracking self-play** — rewind to a decision point and play out a
  different candidate. Needs a `.slog` branch-point extension and a branching
  `GameRunner` mode; parked until training signal is demonstrably
  data-diversity-limited.
- **Standard (hidden-leave) Scrabble**, and with it everything belief. Returning
  means regenerating data and retraining, not redesigning.
- **Search-derived knowledge buffers beyond the evidence loop** — still the
  long-range shape, but every nearer rung must fail first.

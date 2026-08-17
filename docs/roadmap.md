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
move set evaluation model scores all N in ONE pass
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
  └───┘  repeat until the sim budget is spent
      │
      ▼
play the best simmed candidate by simulation value
```

Rollouts inside the loop climb the [policy ladder](#7-rollout-policy-ladder):
value-truncated, then self-model plies, then the endgame solver once the bag
empties.

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
against **~16.8 thread-seconds** of rollouts per turn at K=10 × 400 (measured).
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
the separate reasons under [D2](#7-rollout-policy-ladder).

## What is already built

- **The position evaluation model** — the teacher. Evaluates a post-move,
  pre-draw board from the mover's POV: WLD, a Gaussian over the final score
  differential, and four 15×15 placement masks (opponent/self next-move
  occupancy, each conjoined with that player winning). Trained on HastyBot
  self-play under the generational lifecycle
  ([architecture.md](architecture.md),
  [generational_training.md](generational_training.md)).
- **The move set evaluation model v1** — the student. Board trunk once, one
  cheap vector per candidate, cross-attention scoring all `N` in one pass
  ([model_architectures.md](model_architectures.md)). Distilled from the teacher
  over 600 pairs (A3).
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

### 2. Evidence conditioning

The fusion stage from
[sim_residual_feedback.md](sim_residual_feedback.md), on the move set evaluation
model.

**Stage built, training pending** — the fusion stage, the staged
(cache-splitting) forward, the `.sobs`-plus-first-pass input builder, and the
exactness tests exist ([evidence_fusion.py](../py/scribblez/evidence_fusion.py),
[model_architectures.md](model_architectures.md)); the trained conditioned
model waits on evidence trajectories (item 4).

- **Evidence tokens** — one per simmed candidate: its move encoding (the move
  encoder, reused) fused with its sim observations (maps, value, counts) **and
  the model's own predicted planes for that candidate**, the two plane stacks
  concatenated channel-wise so the encoder forms the residual itself. Keeping
  both raw halves matters: a prediction contradicted by 40 rollouts and one
  contradicted by 2000 share a difference but warrant different updates.
- **The predictions must be inputs.** An encoder reading observations alone can
  express `posterior = prior + g(obs)` but never
  `posterior = prior + k·(obs − prior)`; nothing downstream of an additive merge
  can separate the summands again.
- **The predictions must be the evidence-free first pass**, read at the
  candidate's *post-move* state — so a token never changes once created (it
  caches), and so the conditional matches what the rollouts actually observed.
- **Evidence self-attention**, then cross-attention with the board's spatial map
  `H` producing an evidence-conditioned `H′`. Per-move scoring is unchanged and
  attends into `H′` exactly as it attended into `H`.
- **Late fusion is load-bearing**: evidence must not modulate the trunk's own
  layers, or the trunk cannot be cached across loop iterations.
- **Empty evidence must degrade to the plain one-pass model**, and training
  covers that case explicitly. Every turn starts there: the unconditioned pass
  is what produces the predicted planes the first evidence token carries, even
  though the candidate it is carried for is chosen by the greedy anchor rather
  than by that pass.

### 3. The proves-best head

A per-move head on the evidence-conditioned model predicting the **expected
improvement** a candidate's sim would contribute over the best simmed so far —
`E[max(0, p(w) − best)]`, not the probability that it improves at all (C2). This
is the acquisition function that drives the loop, and it is what keeps the loop
off near-duplicates of the current best.

That distinction is load-bearing at `B = 1`. Under common random numbers two
cosmetically-different candidates — a blank designated differently, placed tiles
reordered — place the same tiles, leave the same rack, and refill from the same
pool, so rollout `i` returns the *same* outcome for both. Their paired
difference is near-zero with almost no spread, so their expected gain is ~0 and
they are suppressed automatically. The probability form is only correct on an
*exact* tie (which never strictly exceeds, so reads 0); once the difference is
small but non-zero it reads ~0.5 when noise-dominated, or ~1 when the duplicate
is reliably a hair better, and spends the sim either way.

Once D1 truncates rollouts, the cancellation stops being exact: two near-identical
leaves get slightly different values from the leaf model, and because that
differential is deterministic given the boards it is a *bias* — more rollouts
converge to it rather than average it away. It is bounded by the leaf model's
local smoothness, measurable against D1's anchor fraction of terminal rollouts
(which do cancel exactly), and does not change the ordering unless it exceeds a
real candidate's improvement. See
[sim_residual_feedback.md](sim_residual_feedback.md).

**Labels come free from CRN sims**: any evidence prefix plus a held-out simmed
candidate is a labeled row. The improvement must be measured against the
best-so-far **over the same seed set** — that pairing is what makes a
duplicate's target ~0 rather than a small random number, and it is satisfied
automatically by labels drawn from one position's `.sobs`.

### 4. Evidence-trajectory generation

The data for items 2 and 3. A trajectory records the sequential loop that
produced each evidence set, which supplies rows at **every prefix size**,
including zero — the distribution the deployed agent actually walks.

**Machinery built, corpus pending** — the trajectory recipe (greedy anchor →
temperature-softmax student proposals at randomized length → one
uniform-random tail sim, per
[sim_residual_feedback.md](sim_residual_feedback.md)) is implemented end to
end:
[evidence_trajectory_generator](../engine/apps/evidence_trajectory_generator.cpp)
writes trajectory `.sobs` (v2: trajectory-ordered records, per-position
legal-move counts and uniform-tail flag, proposer hash), the `.mset`
labeling force-includes the simmed candidates (`--sobs`), and the
`move_set_eval` workload runs the chain per `traj_every`-th pair. No corpus
has been generated yet.

- Requires `.sobs` at scale; the tooling above is how it gets made. The same
  tool reads hand-maintained `.gcg` position sets (`--gcg`,
  [positions/NWL23/face-up-trajectory-set](../positions/NWL23/face-up-trajectory-set/README.md))
  for the exhibits and the position-set metric of items 2–3.
- The proposer follows the current model, so a conditionally-strong but
  equity-buried candidate gains coverage as generations advance.
- The value-labeled subset must always include the proposer's simmed
  candidates, the way the mset sampler always includes the played move —
  otherwise dense value labels stay at the static tail stratum's rate while the
  proposer explores elsewhere.

### 5. Engine runtime for the evidence path

- **ONNX export** of the evidence-conditioned model, split so the cached parts
  (trunk, move encodings, evidence-free predictions) are computed once per turn
  and only the fusion stage plus re-scoring run per loop iteration. Outputs must
  be bit-identical to a full recompute.
- **Engine-side evidence staging**: `SimObservation` → model input, alongside
  the stored per-candidate predicted planes.
- Extends `MoveSetEvaluationSpec` or lands as a second spec beside it.

### 6. The sequential agent

The loop from [the destination](#the-destination), as a new `--player --type=`.
Reuses `mset_sim_agent`'s candidate generation, encoding, and endgame handoff;
replaces its fixed top-K sim set with the proves-best loop.

- **First sim: the greedy anchor** — the highest-raw-score candidate, taken
  straight off the generated move list, not from the model's ranking. It is the
  one pick in the turn that does not depend on a network.
- **Every later sim**: argmax of the proves-best head over the unsimmed
  candidates, conditioned on the evidence so far.
- **Final pick**: best simmed candidate by simulation value.

Admits early stopping — halt when no unsimmed candidate is likely to prove best
— which is where a budget saving turns directly into strength per second.

### 7. Rollout policy ladder

Each rung changes sim semantics, so each lands behind a `.sobs` flag.

- **D1 — value-truncated rollouts.** Sim a few plies, then read the position
  evaluation model's value at the horizon. Keep an **anchor fraction** of
  terminal rollouts per candidate as a ground-truth tether. Costs to accept:
  `.sobs` artifacts become model-versioned, and sims start contending for the
  GPU.
- **D2 — self-model plies.** Plies 1–2 played by our own stack over racks built
  from the public leave, then HastyBot to the horizon. This beats a generic
  policy upgrade because the evidence maps read *exactly* plies 1–2. Inside
  rollouts the model scores only the top-`k` by static equity for small fixed
  `k` — nothing is cached across plies, and fixed `k` gives static tensor shapes
  for batching plies across concurrent rollouts.
- **D3 — endgame solver for late-game rollouts.** A port of Macondo's negamax
  solver into the engine (the WMP precedent); rollouts switch to it when the bag
  empties, with budget scaled by the root's distance to the end.

Order D1 → D2 → D3: D1 is the biggest sim-quality lever, D2 depends on the
trained student, D3 is the largest port with the most localized payoff.

### 8. Cloud generation

The generate role is GPU and local-only until the cloud fleet can host TensorRT
— the GPU-workloads item in [cloud_compute.md](cloud_compute.md), which also
needs a way to ship the teacher model to pods. Both corpus regenerations above
want it.

## Models and how they are trained

Three networks, trained in this order. Each depends on the one above it.

### The position evaluation model (teacher)

- **Trained on**: HastyBot self-play `.slog` data, generational
  generate→train ([generational_training.md](generational_training.md)).
- **Predicts**: WLD, score-differential Gaussian, four placement masks.
- **Status**: trained and in use. Advancing it by promotion rather than by a
  new tag and full regeneration is
  [generational_teacher.md](generational_teacher.md), still deferred.

### The move set evaluation model (student)

- **Trained on**: `.mset` sidecars — the teacher's readouts at each candidate's
  post-move state, paired with pre-move board inputs reconstructed by replay.
- **Predicts**: per candidate, WLD + score differential, **plus the four
  placement planes from item 1**.
- **Retrain required.** v1 has no placement planes, and the corpus carries no
  targets for them. Sequence: settle the `.mset` record format → regenerate the
  corpus → train v2.
- More epochs on the v1 corpus buy nothing; that run plateaued with training
  loss flat alongside the held-out metrics.

### The evidence-conditioned model + proves-best head

- **Trained on**: evidence trajectories (item 4) — `.sobs` observations paired
  with the model's own evidence-free predictions, at every prefix size.
- **Predicts**: everything the student predicts, conditioned on an evidence set,
  plus the proves-best probability per candidate.
- **Hosted on the student**, sharing its trunk and move encoder; the fusion
  stage sits between trunk and heads, which is also what lets the position
  evaluation model take the same evidence through the same stage.
- **Bootstrapping**: the first trajectory corpus is generated with the
  unconditioned student as proposer — it is already correct at the empty
  evidence set, and the greedy anchor supplies the first sim regardless of the
  proposer. Later generations propose with the current conditioned model.

## Rack inference — parked

Face-up leaves removes the need to infer anything, so this is dormant until the
project returns to standard Scrabble.

What exists: a port of the algorithm behind Macondo's `SIMMING_INFER_BOT` — the
hypergeometric prior over draws from the unseen pool, a temperature-softened
static-equity likelihood, exhaustive enumeration of small leave spaces with
importance sampling above them, and the posterior a simulation would sample
racks from ([belief/rack_inference.h](../engine/include/belief/rack_inference.h)).
Tested, with no consumer — expected rather than an oversight.

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
- **Backtracking self-play** — rewind to a decision point and play out a
  different candidate. Needs a `.slog` branch-point extension and a branching
  `GameRunner` mode; parked until training signal is demonstrably
  data-diversity-limited.
- **Standard (hidden-leave) Scrabble**, and with it everything belief. Returning
  means regenerating data and retraining, not redesigning.
- **Search-derived knowledge buffers beyond the evidence loop** — still the
  long-range shape, but every nearer rung must fail first.

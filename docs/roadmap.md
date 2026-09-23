# Scribblez roadmap

Scribblez aims to beat existing Scrabble engines by replacing their
context-blind static evaluation and naive rack inference with learned,
belief-aware evaluation ([design.md](design.md)). This document is the
**implementation plan**: the agent being built, what exists, what is left, and
the models that have to be trained to feed it. Read it to know where a piece
of work fits and why the pieces are ordered as they are.

It contains no experiments. What past measurements established, and how the
finished agent will be evaluated, is in
[evaluation_plan.md](evaluation_plan.md). The plan is committed to, not gated:
every item is part of the destination agent, and nothing here exists to decide
whether to build something else.

## Status at a glance

| Item | Status |
|------|--------|
| [1. Per-move placement planes](#1-per-move-placement-planes) | Done |
| [2. Value-truncated rollouts (D1)](#2-value-truncated-rollouts-d1) | Done |
| [3. Engine runtime for the evidence path](#3-engine-runtime-for-the-evidence-path) | Done |
| [4. Evidence-trajectory generation](#4-evidence-trajectory-generation) | Done; regenerated as improvements are tested |
| [5. The move proposal model](#5-the-move-proposal-model) | Done; retrained as improvements are tested |
| [6. The sequential agent](#6-the-sequential-agent) | Built; waits on a trained model from item 5 |
| [7. Self-model plies and the endgame solver (D2, D3)](#7-self-model-plies-and-the-endgame-solver-d2-d3) | D2 not started; D3 partly built |
| [8. Cloud generation](#8-cloud-generation) | Done for `move_set_eval`; `evidence_trajectories` is local-only |

What remains is mostly compute: generate the item-4 corpus, train the item-5
model on it, then run the measurements in
[evaluation_plan.md](evaluation_plan.md).

**Track labels.** Code comments and older documents refer to work by track
label. The numbering in this document is implementation order; the labels map
as follows:

- **A** — the move set evaluation track. A1 the match harness, A2 `.mset`
  target generation, A3 the distilled student and its gate metrics, A4 engine
  inference and the `mset-sim` agent.
- **B** — rack inference ([parked](#rack-inference-parked)).
- **C** — sim candidate selection. C1 was an interim diversity heuristic, now
  [rejected](#what-is-deliberately-not-here).
- **D** — the rollout-policy ladder: D1 value truncation (item 2), D2
  self-model plies and D3 the endgame solver (item 7).
- **E** — infrastructure: E1 the cloud fleet, E2 match discipline.

## The variant: face-up leaves

Development happens in **face-up-leaves Scrabble**: each player reveals their
leave after every turn, and only the replenishment draws stay hidden. Both
seats see, and may use, the other's retained tiles.

Rack uncertainty is the dominant confound in everything downstream. Removing it
by rule puts the effort where the novelty is: the move set evaluation model,
evidence conditioning, and sim scheduling. None of those components depends on
the information condition, so returning to standard Scrabble later means
regenerating data, not redesigning.

This parks the *belief* half of [design.md](design.md). That is a sequencing
decision, not a retraction; see [rack inference](#rack-inference-parked).

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

The model in the loop is the **move proposal model**: a copy of the move set
evaluation model carrying the fusion stage and the proves-best head
([models](#models-and-how-they-are-trained)). The plain student never runs at
the root; its readouts reach the loop through the copy's plain pass.

Rollouts inside the loop are value-truncated at a horizon deep enough for the
lexical contingencies to play out ([item 2](#2-value-truncated-rollouts-d1)),
and climb the rest of the
[rollout-policy ladder](#7-self-model-plies-and-the-endgame-solver-d2-d3):
self-model plies, then the endgame solver once the bag empties.

The design choices behind the loop:

- **The first sim is a mechanical anchor**, not a model choice: the
  highest-raw-score move, as the greedy agent would pick it, however the model
  ranks it. It is cheap insurance against model blind spots, because the one
  candidate guaranteed a sim is chosen by a rule the model cannot get wrong.
  Its sim is also unusually informative: the residual on the obvious move
  calibrates the rest of the evidence set, which a pick correlated with the
  model's own errors would not do
  ([sim_residual_feedback.md](plans/sim_residual_feedback.md)).
- **The loop is sequential.** Every sim is informed by all prior evidence. If
  the agent sims `N` candidates it queries the proves-best head `N − 1` times:
  the anchor needs no query, and every later pick is evidence-conditioned. In
  [sim_residual_feedback.md](plans/sim_residual_feedback.md)'s schedule
  spectrum this is `(B = 1, R = K)`, which that document identifies as the
  design center; batched multi-round variants are a fallback, not a step on
  the way.
- **The serial queries are free.** One full-candidate-set forward pass costs
  0.37 ms at M = 4000 (measured; [model_specs.h](../engine/include/nn/model_specs.h)),
  against ~16.8 thread-seconds of rollouts per turn at K = 10 candidates × 400
  rollouts (measured). The deployment budget is nearer ~1,000 rollouts per
  candidate (~42 thread-seconds by the same scaling), which is what makes value
  truncation and early stopping load-bearing. Ten sequential passes are four
  orders of magnitude below the rollouts they schedule.
- **Promotion, not re-scoring, is the payoff.** Simmed candidates are ranked by
  their own sims. Conditioning matters because the loop can promote a
  candidate no earlier round would have picked, such as the modest play that
  blocks a hot spot the sims just revealed.
- **No upfront candidate filtering.** `N` ranges from 1 to 10,000+ with blanks.
  Collapsing near-duplicate blank designations before scoring is unnecessary
  and risky: the single linear pass makes large `N` cheap, different blank
  letters produce genuinely different crosswords and hooks, and a filter risks
  dropping exactly the move the model exists to find. Every legal move is
  scored at every iteration. This `O(N)` argument assumes full-set evaluation
  happens once per decision, at the root; a neural *rollout* policy prunes
  instead, for the reasons under [D2](#7-self-model-plies-and-the-endgame-solver-d2-d3).

## What is already built

- **The position evaluation model**, the teacher. It evaluates a post-move,
  pre-draw board from the mover's point of view: WLD, a Gaussian over the final
  score differential, and four footprint-categorical placement heads (where
  each seat's next move lands, and the same conjoined with that seat winning).
  Trained on HastyBot self-play under the generational lifecycle
  ([architecture.md](architecture.md),
  [generational_training.md](generational_training.md)).
- **The move set evaluation model**, the student. The board trunk runs once,
  each candidate gets one cheap vector, and cross-attention scores all `N` in
  one pass ([model_architectures.md](model_architectures.md)). It carries the
  per-move placement readouts of item 1.
- **Target generation** (A2): the `.mset` sidecar, its generator, and the
  `move_set_eval` dashboard workload, run in-variant against a teacher pinned
  by content hash.
- **Engine inference** (A4): the move-set arm of `NeuralNet<Spec>` and its
  evaluation service ([model_specs.h](../engine/include/nn/model_specs.h)), the
  P = 1 ONNX export
  ([onnx_export.py](../py/scribblez/move_set_eval/onnx_export.py)), and the
  `--type=mset-sim` agent
  ([mset_sim_agent.h](../engine/include/agent/mset_sim_agent.h)), which scores
  a turn's whole candidate set in one pass and sims the model's top K: the
  destination agent without the evidence loop.
- **Sim machinery**: [sim_runner.h](../engine/include/sim/sim_runner.h) runs
  common-random-number (CRN) rollouts, to game end or value-truncated;
  [sim_observation_log.h](../engine/include/data/sim_observation_log.h) stores
  them in `.sobs` sidecars.
- **The evidence path**: the fusion stage
  ([evidence_fusion.py](../py/scribblez/evidence_fusion.py)), the proves-best
  head, the trajectory generator and the `evidence_trajectories` workload, the
  evidence trainer (`py/scribblez/evidence/`), the engine runtime, and the
  UltimateBot agent (items 3–6).
- **The sim agent baseline**, the endgame solver, and face-up leaves in the
  game loop.
- **Infrastructure**: the master dashboard and workload registry, the match
  harness (A1/E2), and the cloud fleet (E1).

## The items

In dependency order.

### 1. Per-move placement planes

**Done.** [move_set_eval_v2_results.md](move_set_eval_v2_results.md) records
the corpus, the trained student, and its gate metrics.

The placement distributions the position evaluation model predicts, predicted
instead **per candidate**, for that candidate's post-move state. The scoring
path holds one vector per move, so decoding a board-shaped distribution means
scoring that vector against the 225 board tokens, one readout per head. These
are the quantity sim evidence is differenced against, so everything below
depends on them.

- **Model**: four per-move readouts on the move set evaluation model.
- **Targets**: the teacher's own distributions at the same post-move states;
  the generator already runs the teacher there for the value targets.
- **Format**: each stratified `.mset` record carries the four distributions
  dense and absmax-quantized, one byte per footprint class plus a float32
  scale per head, about 11.8 KB per record. Dense because the masked footprint
  softmax is broad (a top-128 truncation keeps only ~0.8–0.9 of the mass), and
  fixed-size records keep both readers' vectorized indexing. Full-sweep files
  carry no planes; they are evaluation-only and their metrics are value-based.
  [move_set_eval_target_log.h](../engine/include/training/move_set_eval_target_log.h)
  is the authoritative layout.
- **Consequence**: adding planes invalidated every existing corpus, which is
  why the format was settled before regenerating.

### 2. Value-truncated rollouts (D1)

**Done.** `SimRunner` truncates at a configurable horizon and reads the
position evaluation model there, at the post-move, pre-draw state of the last
ply's mover (the state the model is trained on). Games that end earlier keep
their exact result. It is exposed as `--sim-horizon` and `--leaf-model` on the
sim, neural-sim, mset-sim and ultimatebot agents, and as `--horizon` and
`--leaf-model` on `sim_obs_tool` and the trajectory generator. A truncated
`.sobs` stamps the leaf model's content hash and the horizon, and its
observations carry fractional (probability-weighted) outcomes.

Rollouts sim a few plies, then read the model's value at the horizon. The plies
before the horizon supply what the model cannot, the near-root lexical facts
sims exist to observe, and the leaf value stands for everything after.

- **Why it comes first**: truncation is a data prerequisite, not only an agent
  speedup. The deployment budget is ~1,000 rollouts per candidate, and the
  trajectory corpus must carry deployment-quality evidence: the rollout-count
  inputs let the model discount noisy maps, but a head trained only on
  200-rollout evidence has never seen the maps it will be asked to trust.
  Against the first trajectory recipe that is ~5× the rollouts per sim and ~3×
  the sims per position, and truncation is what makes that affordable. The
  kill-test's phase gradient
  ([sim_obs_experiment_results.md](sim_obs_experiment_results.md)) says it also
  *cleans* the evidence: truncation produces late-game-quality, low-variance
  observations at every phase.
- **Horizon**: deep enough for the lexical contingencies the loop hunts to
  resolve. A contingent draw is realized as draw-then-play, plies 2–3 after the
  candidate, so the horizon is at least 3–4 plies (`SimRunner` enforces a
  minimum of 3). Any shallower and the question is handed straight back to the
  lexically blind leaf model, leaving the sim nothing to observe that the
  model did not already know.
- **Costs accepted**: `.sobs` artifacts become model-versioned, and sims
  contend for the GPU.

### 3. Engine runtime for the evidence path

**Done.** The move proposal model runs incrementally in the engine as two
graphs; [model_architectures.md](model_architectures.md#4-side-by-side) has
their inputs and outputs.

- **Split ONNX export**
  ([proposal_export.py](../py/scribblez/move_set_eval/proposal_export.py)): a
  `move_proposal_cache` graph (trunk, move encodings, evidence-free
  predictions; once per turn) and a `move_proposal_step` graph (the fusion
  stage plus re-scoring; per loop iteration, over the cache tensors). Their
  composition is bit-identical to `MoveSetEvalModel.forward` in PyTorch
  (`test_move_set_eval_evidence.py`); across independently built TensorRT
  plans it is tolerance-bounded.
- **Evidence staging**
  ([evidence_staging.h](../engine/include/agent/evidence_staging.h)): turns
  `SimObservation`s, moves and the cache's per-candidate predictions into the
  fusion stage's padded `(1, E, …)` inputs.
- **Runtime**: two specs beside `MoveSetEvaluationSpec`
  (`MoveProposalCacheSpec`, `MoveProposalStepSpec`), served at FP32 through
  `NeuralNet<Spec>` directly, because the handoff tensors do not fit
  `TrtEvalService`'s row-uniform decode. One shared engine pair per run
  ([move_proposal_nets.h](../engine/include/agent/move_proposal_nets.h)) is
  driven through per-consumer sessions
  ([move_proposal_session.h](../engine/include/agent/move_proposal_session.h))
  behind the GPU-free
  [move_proposal_service.h](../engine/include/agent/move_proposal_service.h)
  interface. Verified against the PyTorch reference over empty, partial and
  full evidence sets by `test_proposal_inference_parity.cpp`, with
  `proposal_infer_smoke` checking GPU liveness. The cache graph bounds its
  chunks at 1024 rows so that a per-thread agent fits a 4 GiB match GPU.

This item precedes data generation because item 4's on-policy side *is* the
deployment loop, so the generator needs the runtime before the corpus can be
made.

### 4. Evidence-trajectory generation

**Done.** Corpora are regenerated continually as improvements are tested. The
`evidence_trajectories` workload's generate role runs self-play, then the
[evidence trajectory generator](../engine/apps/evidence_trajectory_generator.cpp)
(selection in
[evidence_trajectory_select.h](../engine/include/training/evidence_trajectory_select.h)),
then `.mset` labeling. An earlier recipe produced the 200-rollout corpus the
frozen trial of item 5 consumed.

Per labeled position the generator sims one **pool**, every candidate under
common random numbers and at the tag's rollout and truncation configuration
(the deployment configuration of item 2 is the target):

- **The anchor**: the highest-raw-score move, exactly as deployed.
- **A ≈ 15 on-policy picks** (`on_policy_min`/`on_policy_max`): the deployment
  loop itself, iterative proves-best proposals conditioned on the sims so far,
  with temperature in the argmax for exploration. Generation 0 has no trained
  gain head, so it selects these by a temperature softmax over the plain
  student's values on the **full** candidate set. There is no proposal-pool cap
  in any generation: deployment argmaxes over every unsimmed candidate, and a
  cap keeps deep promotions out of the corpus. The exhibit is the trajectory
  set's `egotize-lane` position, where a top-64 cap left the key move GAVE
  unsimmable.
- **B ≈ 3 off-policy draws** (`off_policy_count`): drawn uniformly over the
  legal moves the anchor and on-policy picks did not take, and held out of
  evidence sets by construction. This is the bounded floor against the
  proposer's echo chamber, kept assumption-free: a uniform draw samples
  exchanges and the tail at their natural frequency, so no stratum has to be
  hand-specified. Stratified or semantic draws (contention zone, high leave,
  setups) are a later refinement if the floor proves too coarse; the rationale
  is in [sim_residual_feedback.md](plans/sim_residual_feedback.md).

Training rows are **assembled from the pool, not replayed from it**. The
evidence set is permutation-invariant and the gain label is a max over the
set, so any subset of {anchor} ∪ A that contains the anchor and fits the sim
budget is a valid evidence set, and every pool member outside it is a labeled
held-out row, its gain measured against the CRN max over the set. One pool
yields combinatorially many rows, the right response to sims that cost ~1,000
rollouts each. The B draws never enter an evidence set: deployed evidence
holds only the anchor and proposer picks, so keeping the floor labels-only buys
its coverage at no cost to the input distribution.

- The same tool reads hand-maintained `.gcg` position sets (`--gcg`,
  [positions/NWL23/face-up-trajectory-set](../positions/NWL23/face-up-trajectory-set/README.md))
  for the exhibits and item 5's position-set metric. The dashboard's
  Trajectories tab ([react_dashboard.md](react_dashboard.md#trajectories-evidence_trajectories))
  replays a set position through any checkpoint at every evidence-set size,
  and the trainer charts the set's sim-best rank (`posset_*`) per pass.
- The `.mset` labeling force-includes each position's simmed pool (`--sobs`),
  the way the mset sampler always includes the played move. Otherwise dense
  value labels would stay at the static strata's rate while the proposer
  explores elsewhere.

### 5. The move proposal model

**Done.** The model is retrained continually as improvements are tested. The
trainer is `py/scribblez/evidence/`, the `evidence_trajectories` workload's
train role;
[model_architectures.md](model_architectures.md#training-the-evidence-path-scribblezevidence)
has its modes and loss table.

The evidence consumer, and the model the sequential agent runs at the root: a
**copy** of the move set evaluation student (trunk, move encoder, value and
placement heads, fusion stage) plus the **proves-best head**, which predicts
the expected improvement `E[max(0, v − best-so-far)]` a candidate's sim would
add over the best simmed so far. This is the acquisition function that drives
the loop. The expected-gain form (rather than a probability of being best),
its CRN pairing, and the truncation caveat are settled in
[sim_residual_feedback.md](plans/sim_residual_feedback.md#candidate-selection).

It is a separate model rather than new heads on the student so that each keeps
one job and one lifecycle. The student stays a pure distillation vessel (the
dense prior, the backbone source, and under D2 the rollout policy), while the
copy is free to follow the sim signal.

It trains on item 4's assembled rows with two loss components:

- **Gain** (primary): Huber against the held-out candidate's CRN-paired gain
  over its evidence set's best. Best-so-far is a known scalar at inference, the
  max sim value over the evidence gathered so far, so it is fed to the head as
  an input rather than left to be reconstructed from the pooled evidence (a
  mean pool cannot carry the max the target depends on). The head computes it
  from the evidence tokens' observed win values (`evidence_fusion.best_so_far`)
  identically in training, in the dashboard's Trajectories pane, and in the
  exported step graph.
- **Conditioned WLD and score differential** (auxiliary): soft-CE and Huber
  against the held-out candidate's own sim outcome, on the same rows. The
  target is always the sim outcome, never the plain teacher: the teacher's
  readout is a function of the board alone, so on evidence-bearing rows it
  would train the fusion stage to ignore evidence. The gain is a thin
  transform of the conditioned value, so these auxiliaries help the head at no
  extra sim cost.

The backbone trains, starting from the student, and the empty-evidence
(prefix-0) rows keep the evidence-free pass calibrated as a board-only prior on
the simmed candidates. There is deliberately **no self-distillation anchor**.
Its only added job would be extending that calibration to the *unsimmed* legal
moves the gain argmax ranges over, and doing that cleanly needs a live
frozen-student forward over all `N` candidates per position. The replay
pipeline (encoded inputs, no move list) cannot supply that without a new
engine move generator, a large build for a speculative stabilizer. The gain
head instead generalizes from a diverse held-out set (anchor, on-policy, and
low-value off-policy draws) and the student starting point, with
`backbone_lr_mult` as the drift knob. Whether the argmax over unsimmed moves
holds up is measured at the agent (item 6); adding the self-distillation anchor
is the known fallback if it does not.

**The recorded floor.** The gen-1 frozen-backbone trial (fusion stage and head
only, over the 200-rollout corpus) measured conditioned − plain soft-CE
−0.0008, and an acquisition hit rate of 0.57 against the plain value's 0.61.
The mechanism visibly engages on exhibits, but 200-rollout evidence is too
noisy, and a zero-initialized fusion stage over a frozen trunk too weak, to
beat the plain ranking. The plan addresses both: deployment-count rollouts
(item 2) and a trainable backbone.

One fusion refinement remains open, cheap and unimplemented: a direct
move-query → evidence-token cross-attention (`O(N·K)`), letting a candidate
compare itself to each simmed move by encoding rather than only through shared
board squares. Today a leave-twin with a different footprint is visible only
through the move scalars and the pooled summary.

A proposed extension, keeping evidence per sampled opponent rack so knowledge
transfers across candidates and rollouts, is
[rack_conditional_evidence.md](plans/rack_conditional_evidence.md).

### 6. The sequential agent

**Built; waits on a trained model from item 5.** `--player
"--type=ultimatebot"` ([ultimate_bot_agent.h](../engine/include/agent/ultimate_bot_agent.h))
is the loop from [the destination](#the-destination) as a playing agent, over
the item-3 runtime. It reuses `mset-sim`'s candidate generation, encoding, and
endgame handoff. The loop itself
([evidence_loop.h](../engine/include/agent/evidence_loop.h)) holds no network
and runs no sims: it drives a `MoveProposalService` and a `CandidateSimmer` it
is handed, with the pick rule as a policy, so item 4's generator can later run
its on-policy side through the same loop with a tempered pick.

The evidence trainer exports the cache/step pair every pass, stamped with the
evidence width it trained at; the agent checks its `--max-sims` against that
width at load. The workload's `match_eval` role plays every Nth exported pair
as UltimateBot against a fixed opponent. It runs locally only, until the
truncation leaf model can be shipped to an ssh worker.

- **First sim: the greedy anchor**, the highest-raw-score candidate, taken
  straight off the generated move list rather than from the model's ranking.
- **Every later sim**: the argmax of the proves-best head over the unsimmed
  candidates, conditioned on the evidence so far.
- **Early stopping**: halt when no unsimmed candidate's predicted gain clears
  `--gain-threshold` (in the gain head's win-probability units; 0 never stops
  early). At ~1,000 rollouts per sim, this is where a budget saving turns
  directly into strength per second.
- **Final pick**: the best simmed candidate by simulation win rate, the value
  the gain head is trained in, so the pick and the stopping rule agree. The
  agent has no spread objective. `--max-sims` (default 10, anchor included;
  the counterpart of `mset-sim`'s `--sim-top-k`) bounds the loop, and 1 plays
  the anchor unsimmed.

What remains is compute: a trained head, then the budget and threshold
measurements in [evaluation_plan.md](evaluation_plan.md).

### 7. Self-model plies and the endgame solver (D2, D3)

The rest of the rollout-policy ladder. Each rung changes what a sim means, so
each lands behind a `.sobs` flag.

- **D2: self-model plies.** Not started. Plies 1–2 are played by our own stack
  over racks built from the public leave, then HastyBot plays to the horizon.
  This beats a generic policy upgrade because the evidence maps read exactly
  plies 1–2. Inside rollouts the model scores only the top `k` moves by static
  equity, for a small fixed `k`: nothing is cached across plies, and a fixed
  `k` gives static tensor shapes for batching plies across concurrent
  rollouts. D2 is also what closes the loop AlphaZero-style: once rollouts play
  with the model, sim quality, and every label derived from it, improves with
  each generation. Until then the sim is a fixed HastyBot oracle, and training
  distills it.
- **D3: the endgame solver in late-game rollouts.** Partly built. The engine
  has an endgame solver ([endgame_solver.h](../engine/include/endgame/endgame_solver.h)),
  and `SimRunner::Params::solve_endgames` hands every rollout's endgame to it
  once the bag empties, at the solver's default budget. Only the sim candidate
  survey (and the `blind_spots` workload built on it) turns this on today; the
  agents, the trajectory generator and `.sobs` do not expose it yet. Scaling
  the budget by the root's distance to the end is unbuilt.

D2 comes before D3: D2 depends on the trained student and carries the
generational payoff; D3's payoff is more localized.

### 8. Cloud generation

**Done for `move_set_eval`.** The engine worker image hosts TensorRT, and the
generate role declares its teacher export as an out-of-tag input
(`RoleSpec.inputs`, [cloud_compute.md](cloud_compute.md)) that the controller
stages for a remote slot. The train role pulls the pair store and delivers its
exports through its sink, so a whole `move_set_eval` run can sit on rented GPUs
with only the dashboard local. The `evidence_trajectories` roles (which need
the teacher, the proposer and the leaf model) remain local-only because they do
not declare their inputs yet.

## Models and how they are trained

Three networks, trained in this order; each depends on the one above it.

### The position evaluation model (teacher)

- **Trained on**: HastyBot self-play `.slog` data, generational generate→train
  ([generational_training.md](generational_training.md)).
- **Predicts**: WLD, a score-differential Gaussian, and four footprint
  placement heads.
- **Roles**: the teacher for the student's distillation, and the rollout leaf
  evaluator of item 2. The leaf role puts it inside the generational
  improvement loop: stronger self-play, better value, better leaves, better
  sims, better labels.
- **Planned second target stream: sim values.** A simmed candidate's `.sobs`
  record is a many-rollout estimate of the value at its post-move state. Under
  face-up leaves the sim samples the same draw distribution the game did, so it
  is the same target as the game outcome at a fraction of the variance.
  Positions with sims would train on both streams. Constraint: the stream must
  come from untruncated sims, because a truncated sim value embeds the model's
  own leaf readouts and the model must not train on its own outputs. The plan
  is [sim_labeled_candidates.md](plans/sim_labeled_candidates.md).
- **Advancing it** by promotion, rather than by a new tag and full
  regeneration, is [generational_teacher.md](plans/generational_teacher.md),
  deferred.

### The move set evaluation model (student)

- **Trained on**: `.mset` sidecars, the teacher's readouts at each candidate's
  post-move state (WLD, score differential, the four placement
  distributions), paired with pre-move board inputs reconstructed by replay.
  Distillation only: the student has no sim-outcome losses, and the fusion
  stage its code hosts trains only in the move proposal copy.
- **Predicts**: per candidate, WLD, score differential, and the four placement
  distributions.
- **Roles**: the dense prior over full candidate sets; the backbone the move
  proposal model is copied from; under D2, the rollout policy.

### The move proposal model

- **Is**: the student copy plus the proves-best head
  ([item 5](#5-the-move-proposal-model)), the model at the root of the
  deployed loop.
- **Trained on**: evidence-set rows assembled from item 4's pools, under item
  5's loss: gain first (best-so-far fed as an input), with the sim-outcome
  auxiliaries.
- **Bootstrapping**: the gen-0 pool's on-policy side is selected by the plain
  student (a temperature softmax over the full candidate set). That is correct
  at the empty evidence set, and the greedy anchor supplies the first sim
  regardless of proposer. Later generations select with the current move
  proposal model, and each new student generation refreshes the copy's
  starting point.

## Rack inference (parked)

Face-up leaves removes the need to infer anything, so this is dormant until
the project returns to standard Scrabble.

What exists: a port of the algorithm behind Macondo's `SIMMING_INFER_BOT`
([belief/rack_inference.h](../engine/include/belief/rack_inference.h)). It
combines a hypergeometric prior over draws from the unseen pool with a
temperature-softened static-equity likelihood, enumerates small leave spaces
exhaustively and importance-samples above them, and yields the posterior a
simulation would sample opponent racks from. It is tested, and its one consumer
is offline: the hidden-leaves Monte Carlo ground truth of the position
evaluation test sets samples the opponent's leave from this posterior
([sim/monte_carlo_sim.h](../engine/include/sim/monte_carlo_sim.h)), at the
default (Macondo) temperature. Nothing in play uses it.

Resuming means pricing the posterior against ground truth (a `.slog` replay
recovers the leave the opponent actually held), which also sets the likelihood
temperature, and then wiring it into `SimRunner`, whose per-rollout-index
sampling already preserves common random numbers. Beyond that lies the learned
belief system of [design.md](design.md) §3.

## What is deliberately not here

- **Batched multi-round scheduling.** The sequential loop subsumes it; batch
  mode returns only if sequential proposal underperforms it.
- **Diversity heuristics** (C1's footprint/lane-overlap penalty) and footprint
  dedup at sim-selection time. Redundancy is already handled twice: the
  expected-improvement target rates a CRN duplicate at ~0 by construction, and
  evidence conditioning propagates a disappointing sim to every candidate
  sharing its blind spot, since corrections are written onto board squares and
  near-duplicates attend to the same ones. A hand-built novelty penalty would
  be redundant and lossier: it can only express "identical footprint or not",
  where the model grades similarity by degree.
- **A frozen-backbone move proposal model.** The frozen mode stays in the
  trainer as the diagnostic behind the [recorded floor](#5-the-move-proposal-model);
  the plan trains the backbone, relying on the sim signal and a small backbone
  learning rate rather than an explicit anchor.
- **Backtracking self-play**: rewinding to a decision point to play out a
  different candidate. It needs a `.slog` branch-point extension and a
  branching `GameRunner` mode, and is parked until training signal is
  demonstrably limited by data diversity.
- **Standard (hidden-leave) Scrabble**, and with it everything belief.
  Returning means regenerating data and retraining, not redesigning.
- **Search-derived knowledge buffers** beyond the evidence loop
  ([design.md](design.md) §8.1): still the long-range shape, but every nearer
  rung must fail first.

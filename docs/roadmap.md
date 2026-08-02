# Scribblez Project Roadmap

Scribblez aims to beat existing Scrabble engines by replacing their
context-blind static evaluation and naive rack inference with learned,
belief-aware evaluation ([design.md](design.md)). This document is the plan for
getting there: the variant we develop in, what is built, the agent everything
converges on, and the tracks that remain — each with its own experiment and its
own fallback, priced before it earns permanence.

## The variant: face-up leaves

Development happens in **face-up-leaves Scrabble**, where each player reveals
their leave after every turn and only the replenishment draws stay hidden.
Symmetrically: both seats see, and both may use, the other's retained tiles.

The reason is that rack uncertainty is the dominant confound in every
measurement this roadmap depends on. The kill-test's open-leaves pilot showed a
**~5× larger transfer gain** than the hidden-information arm, which says the
evidence loop's signal is being swamped by noise about what the opponent holds.
Removing that uncertainty by rule makes every downstream readout — the
re-ranking experiment above all — sharper and faster, and it lets the effort go
where the novelty is: the move set evaluation model, evidence conditioning, and
sim scheduling. None of those components are specific to an information
condition, so returning to standard Scrabble later is a data regeneration
rather than a redesign.

What this parks is the *belief* half of the thesis in
[design.md](design.md) — its second named weakness of existing engines. That is
a sequencing decision, not a retraction; see the rack inference track below.

Benchmark comparability survives the change. Macondo's published OracleBot
result (~53.3% against BestBot, which is its SimmingBot) pits a leave-knowing
bot against one that plays without leave knowledge. An opponent that declines
to read a public leave produces exactly the games an opponent that never knew
about it would, since its policy does not model what its opponent knows — so
our agent against a leave-ignoring BestBot measures the same thing. Only
self-play differs, both seats there reading the leave.

## What is built

**The position evaluation model.** It evaluates a board state *after* the mover
places tiles but *before* drawing replacements, from that seat's POV,
predicting:

- **Win/Draw/Loss probabilities** — the primary value signal.
- **Score differential** — a Gaussian over the clipped final differential,
  trained by Gaussian NLL.
- **Placement masks** — four 15×15 heads for where each player's next move
  places tiles, and the per-square win conjunctions the sim-evidence loop reads
  ([sim_residual_feedback.md](sim_residual_feedback.md)).

Starting here was deliberate: applying a candidate move and asking "how good is
the resulting state?" is exactly the Q-value that move selection needs. Training
runs on HastyBot self-play under the generational generate→train lifecycle
([architecture.md](architecture.md),
[generational_training.md](generational_training.md)).

**Self-play diversification.** HastyBot plays one deterministic strategy, so
training only on its games teaches values only for HastyBot-reachable positions.
Move sampling (a softmax over static equity) and random openings both ship;
positions preceding a game's last random ply are excluded from the
training-eligible region, so random play never pollutes an outcome target.

**Validation machinery.** No single metric suffices; four complementary evals,
and the failure mode each catches:

| Eval | What it tests | Failure mode it catches |
|---|---|---|
| Monotonicity probes | Structural coherence | Nonsensical evaluations |
| Calibration testing | Probabilistic accuracy | Structurally sound but biased |
| Monte-Carlo comparison | Absolute value accuracy | Divergence from deep-search ground truth |
| Match play | Downstream utility | Calibrated but not *useful* for move selection |

All four are built and render on the dashboard: match play is A1's
`match_eval` worker, which plays sequential-test-checked paired matches for
each exported checkpoint against a fixed baseline. Automating it first was a
prerequisite rather than a nicety — every remaining track's final readout is
match play, because root cross-entropy demonstrably cannot see re-ranking
value.

**Sim machinery.** [sim_runner.h](../engine/include/selfplay/sim_runner.h) runs
common-random-number rollouts over a position's candidates and
[sim_observation_log.h](../engine/include/selfplay/sim_observation_log.h) stores
them in `.sobs` sidecars alongside the `.slog` data.

## What the kill-test established

The sim-evidence kill-test
([sim_obs_experiment_results.md](sim_obs_experiment_results.md),
[sim_residual_feedback.md](sim_residual_feedback.md)) passed, and its numbers
shape everything below:

- **The mechanism is real.** Conditioning the position evaluation model on sim
  evidence improves held-out outcome prediction with clean controls. The
  deployment-shaped (leave-one-out) transfer gain is smaller but significant,
  tail-concentrated, and late-game-loaded — the signature of evidence mattering
  exactly where decisions are contested.
- **Root-value accuracy is saturating.** A 200-rollout sim and the trained trunk
  are roughly equal, highly correlated estimators of root WLD; fusing them buys
  thousandths. The remaining prizes are *decision quality* (which move gets
  played) and *sim quality* (what the rollouts know), not root cross-entropy.
- **Sim quality had two independent limiters, and the variant removes one.**
  Opponent-rack uncertainty is settled by rule now that leaves are face up;
  what remains is rollout variance, which value-truncated rollouts attack (D1).

## The destination

One picture of the agent this roadmap builds toward, with each component labeled
by its track:

```
GADDAG generates all N moves
      │
      ▼
the move set evaluation model scores all N in one pass    [A]
      │
      ▼
proves-best scheduler picks what to sim               [C]
      │
      ▼                     rollout policy ladder      [D]
sims: opponent racks are    ── ply 1..2: our own stack (move set eval)
      the public leave       ── middle plies: HastyBot (WMP greedy)
      plus hidden draws      ── bag empty: depth-limited endgame solver
      │
      ▼
evidence-conditioned re-rank of all N                 [A/e]
      │
      ▼
sim the promoted moves, pick by sim
```

The tracks are separable, but they compound: sharper sims make evidence and the
proves-best labels more trustworthy, and a self-model rollout policy makes the
ply-1 reply distribution (which is what the evidence maps actually read) match
reality.

---

## Track A: the move set evaluation model — the spine

Everything else attaches to the move set evaluation model: it is the candidate
filter, the host of the proves-best head, the re-ranking surface for evidence,
and eventually the first plies of the rollout policy.

### Architecture

Evaluating every legal move with the position evaluation model means a full
board re-encode per candidate — redundant, since all candidates share the board.
The move set evaluation model takes the board plus all `N` candidates and
predicts in one pass what the position evaluation model would say about each
post-move state.

- **Board encoder** (once per position): the position evaluation model's trunk,
  producing a spatial map `H` (one vector per square) and a pooled global vector
  `g`.
- **Move encoder**: each move becomes a vector from its placed tiles (letter +
  board square, so spatial patterns like "lands on a triple-letter" are
  learnable) and a small scalar block
  ([move_set_encoder.h](../engine/include/training/move_set_encoder.h)).
- **Cross-attention scoring**: each move embedding queries into `H` (moves do
  not attend to each other), fuses with `g`, and projects to `Q(s, aᵢ)`. All `N`
  score in one batched pass — `O(N)` with the board encode amortized.
- **Exchange head**: the exchange and pass moves leave the board unchanged, so a
  dedicated head reads `g` and outputs the best keep-mask and its value,
  competing directly with the best word play.

**No upfront candidate filtering.** `N` ranges from 1 to 10,000+ (blanks), and
collapsing near-duplicate blank designations before scoring is both unnecessary
and risky: the single linear pass makes large `N` a non-problem, differing blank
letters produce genuinely different crosswords and hooks, and an upfront filter
risks dropping exactly the move the model exists to find (the modest play that
blocks a triple-word lane). Top-K diversity for simulation is handled *after*
scoring, by C1's footprint dedup of the ranked handful.

The `O(N)` argument assumes full-set evaluation happens roughly once per
decision, at the root. A neural *rollout* policy (D2) instead scores only the
top-`k` by static equity for a small fixed `k`: nothing is cached across plies,
so the pruning caps what a 20,000-move two-blank position would otherwise cost
and the fixed `k` gives static tensor shapes for batching plies across
concurrent rollouts. Context-blind pruning is second-order inside rollouts —
both simulated players share the policy, so residual bias largely cancels in
candidate comparisons — whereas at the root it would be fatal.

### Steps

Two prerequisites came first, neither of them model work, and both are done:

- **Face-up leaves in the game loop.** The variant is playable, not just
  replayable: `Game::set_face_up_leaves` makes each player's retained tiles
  public until they move again, `MoveRequest.opp_rack` carries what the mover
  legitimately knows (the public leave, or the whole rack once the bag is
  empty), agents and the encoder read it at play time, and the `.slog` header
  records the variant.
- **The sim agent baseline**
  ([sim_agent.h](../engine/include/agent/sim_agent.h)): top-K candidates by
  static equity → CRN rollouts → pick by sim, handing the turn to the endgame
  solver once the bag empties. It is the baseline the move-set-evaluation agent
  must beat, the harness in which C and D get their match readouts, and —
  being simming plus the endgame solver we already have — our equivalent of
  Macondo's BestBot, so it doubles as the opponent that makes published results
  comparable.

Then the spine proper:
- **A1 — automated match eval.** Built: the `match_eval` worker
  ([runner.py](../py/scribblez/match_eval/runner.py)) plays paired matches for
  each exported checkpoint against a fixed opponent, with win-rate curves and
  a sequential significance test (`scribblez/stats.py`, the E2 discipline) on
  the dashboard's Match tab.
- **A2 — target generation at scale.** Built: the `.mset` sidecar and its
  generator
  ([move_set_eval_target_log.h](../engine/include/training/move_set_eval_target_log.h),
  [move_set_eval_target_generator.cpp](../engine/apps/move_set_eval_target_generator.cpp)),
  and the `move_set_eval` dashboard workload
  ([workloads/move_set_eval.py](../py/scribblez/workloads/move_set_eval.py)):
  each cycle plays a self-play batch, runs the generator with the tag's frozen
  teacher ONNX (hash-stamped into every sidecar), and delivers `.slog`/`.mset`
  pairs to the tag's store. Remaining, found by plan review: the variant is
  not actually wired through this pipeline — the workload never passes
  `face_up_leaves` to self-play (so every corpus generated so far is standard
  hidden-leave Scrabble), and the target generator neither honors the
  teacher's `opp_leave_input` metadata nor stamps the `.mset` open-leaves
  flag from the source `.slog` header. Beyond the wiring: an opp-leave-input
  teacher must be trained and exported (none exists — it needs face-up-leaves
  position-eval training first), and the corpus accumulated so far is
  regeneration debt once the wired pipeline and that teacher exist.
  Separately, the generate role is GPU and local-only until the cloud fleet
  can host TensorRT — the GPU-workloads item in
  [cloud_compute.md](cloud_compute.md), which now also needs a way to ship the
  teacher model to pods.
- **A3 — move-set-evaluation v1.** The model, dataset, trainer, and gate
  metric exist as a lean local loop
  ([py/scribblez/move_set_eval/](../py/scribblez/move_set_eval/)). This lands
  in two slices. First, an **offline shakeout**: train once on the corpus
  local generation has already produced and measure — answering whether the
  distillation works at all before any infrastructure is built. That corpus
  predates A2's variant wiring, so shakeout numbers are provisional and get
  re-derived after regeneration. Second, only if the shakeout justifies it,
  the **generational fold-in**, whose real work is named here rather than
  discovered later: a data loader that is not the current synchronous
  per-batch replay-decode (extend the native loader or add a prefetcher), a
  file-level held-out reservation (position-level splits leak through shared
  game prefixes), and a reuse regime re-derived for frozen sidecar targets
  (the `turns_per_game` fresh-sampling lever does not exist here). The
  teacher is pinned for the whole A3–A4 arc; refreshing it is a deliberate
  new tag priced as a full corpus regeneration. "At scale" means the largest
  corpus local generation sustains — not gated on the cloud item above. The
  headline metrics are not target MSE but **top-K recall against the
  position evaluation model's ranking** and **teacher-value regret** — the
  teacher-value gap between its best candidate and the best candidate the
  filter retains; recall alone scores dropping a near-tie like dropping a
  uniquely winning move. Both are measured on a **full-sweep held-out slice**
  (a generator mode that scores every legal candidate for a few positions
  per game): the stratified ~15-candidate training samples are structurally
  blind to exactly the tail moves the filter exists to catch. A3 ships
  curves, not a bar — the bar is set by A4, which owns the measurement that
  defines it.
- **A4 — the move-set-evaluation agent.** First, the
  **position-evaluation-top-K agent** — exact per-candidate evaluation by the
  position evaluation model over a generous static-equity shortlist, then sim
  — which needs no new runtime and owns two measurements: the **sensitivity
  sweep** that sets A3's bar (vary K, or inject controlled recall degradation
  into the shortlist, and measure the match-play cost of a recall miss), and
  the equal-budget baseline the learned filter must beat. Then the
  move-set-evaluation agent itself: top-K by the move set evaluation model →
  sim → pick by sim, matched against the sim agent baseline and against that
  exact-evaluation agent. Its missing ONNX export path and engine-side
  runtime are a design task, not plumbing: the current model materializes a
  per-move copy of the board tokens (gigabytes at two-blank `N`) and the
  evaluation service API speaks flat fixed-width rows, so the export needs a
  chunked candidate axis or shared board keys, and a two-input service API —
  decided before the ONNX graph is frozen. Until the learned filter beats
  exact evaluation at equal budget, exact evaluation stays the selector.
- **A5 — evidence-conditioned move set evaluation** (steps 5–6 of
  [sim_residual_feedback.md](sim_residual_feedback.md)): the fusion stage
  migrates from the kill-test's position-evaluation harness onto the move set
  evaluation model, enabling the two-round re-rank. Gated on E3.

## Track B: rack inference — parked

Face-up leaves removes the need to infer anything, so this track is dormant
until the project returns to standard Scrabble.

What exists already: a port of the algorithm behind Macondo's
`SIMMING_INFER_BOT` — the hypergeometric prior over draws from the unseen pool,
a temperature-softened static-equity likelihood, exhaustive enumeration of
small leave spaces with importance sampling above them, and the posterior a
simulation would sample racks from
([belief/rack_inference.h](../engine/include/belief/rack_inference.h)). It is
tested but has **no consumer**, which is expected rather than an oversight.

What was never done, and is where this resumes: pricing the posterior against
ground truth (a `.slog` replay recovers the leave the opponent actually held,
so posterior log-loss against the prior's measures the information gain
directly), which is also what sets the likelihood temperature. Then wiring the
posterior into `SimRunner`, whose per-rollout-index sampling already preserves
common random numbers. Beyond that lies the learned belief system of
design.md §3, which only earns attention if cheap inference leaves a wide gap.

## Track C: sim scheduling — spend rollouts where they buy information

From the candidate-selection analysis in
[sim_residual_feedback.md](sim_residual_feedback.md): the next candidate to sim
should be the one most likely to *prove best* — good on its own, and different
enough from the already-simmed candidates to beat them.

- **C1 — v0 diversity.** A footprint/lane-overlap novelty penalty at top-K
  selection time. Hours of work, and it directly improves evidence diversity
  for A5.
- **C2 — proves-best head.** A per-move head on the evidence-conditioned move
  set evaluation model predicting the probability that a candidate's sim
  strictly exceeds the best so far. Labels are free from the CRN sims already
  sitting in every `.sobs`: any evidence prefix plus a held-out simmed candidate
  is a labeled row. The target is a function of the evidence set, so C2 wants
  A5 as its host, but the target and labels can be validated earlier on the
  kill-test's evidence-conditioned position-evaluation harness.
- **C3 — adaptive scheduling.** Propose by the proves-best head; evaluate
  batched against sequential schedules. **Readout:** decision quality at a fixed
  rollout budget, and budget required for fixed decision quality. The prize is
  real at deployment — sims dominate think time, so a 2× budget saving is a 2×
  stronger agent per second.

## Track D: the rollout policy ladder

Current rollouts are HastyBot-to-the-end. Each rung changes sim semantics, so
each lands behind a `.sobs` flag, gets validated by the paired kill-test
machinery, and then by match play.

- **D1 — value-truncated rollouts** (design.md §5.2): sim a few plies, then read
  the position evaluation model's value at the horizon. The kill-test's 8×
  late-vs-early phase gradient says this is the biggest sim-quality lever, and
  the sim agent's own strength curve says the same from the other end: rolling
  out to a natural game end, it needs some 400 rollouts per candidate just to
  pass the static evaluator it filters with, and fewer than 200 leave it worse
  than playing no rollouts at all. Keep
  an **anchor fraction** of terminal rollouts per candidate — a ground-truth
  tether and a free measurement of the value model's bias. Costs to accept:
  `.sobs` artifacts become model-versioned, and sims start contending for the
  GPU (the contention-manager regime
  [generational_training.md](generational_training.md) plans for).
- **D2 — self-model plies.** Plies 1–2 of each rollout played by our full stack
  (move set evaluation top-1 or a temperature sample, over racks built from
  the public leave),
  then HastyBot to the horizon. This beats a generic policy upgrade because the
  evidence maps *read exactly plies 1–2*. Cost: batched leaf evaluation on the
  game-pool substrate, at shallow plies only.
- **D3 — endgame solver for late-game rollouts.** Macondo's negamax solver has
  the needed shape (depth/time limits, first-win-only mode, transposition table,
  multithreading) but is Go, so this is a **port into the engine** (the WMP
  precedent). Rollouts switch to the solver when the bag empties, with budget
  scaled by the root's distance to the end. Validate with a bag-≤-N kill-test
  slice and endgame-position match play.

Ladder order is D1 → D2 → D3 by measured value per unit effort: D1 is supported
by existing evidence, D2 depends on A3, and D3 is the largest port with the most
localized payoff.

## Track E: scale and readouts — the enablers

- **E1 — cloud fleet** ([cloud_compute.md](cloud_compute.md), in flight). First
  consumer: A2's target generation.
- **E2 — match harness statistics** (with A1). Match play needs its own
  discipline — paired seeds and racks across agents (the CRN idea at the match
  level) and sequential stopping — owned by the harness
  ([harness.py](../py/scribblez/match_eval/harness.py) over the engine's
  `--paired` mode, `scribblez/stats.py`) so experiments do not reinvent it.
  Still open: full tile-order CRN, which needs the `Bag` reshaped into a
  seeded permutation (today the shared-seed draw streams diverge once the two
  arms' replenishment counts differ).
- **E3 — the re-ranking experiment.** The pivotal readout root-CE experiments
  structurally cannot provide: match play between (a) pick-by-sim over top-K and
  (b) a two-round agent that sims top-K, re-ranks the *unsimmed* candidates with
  the evidence-conditioned model, sims the promoted moves, and picks. Runnable
  with the kill-test's position-evaluation-based fusion model before the move
  set evaluation model exists; A5 productionizes it only if E3 says yes.

---

## Sequencing

Effort splits into three lanes that can run concurrently (one person + fleet:
lead with the spine, interleave the others as experiments block on data).

| Stage | Spine (A) | Sims (C/D) | Enablers (E) |
|---|---|---|---|
| 1 | Face-up leaves in the game loop; the sim agent baseline; A1 match eval | — | E1 fleet lands; E2 match statistics |
| 2 | A2 variant wiring + regeneration; A3 offline shakeout, then fold-in | D1 truncated rollouts; C1 novelty dedup | **E3 re-ranking experiment** |
| 3 | A4 sensitivity sweep → recall bar; move-set-evaluation agent | C2 proves-best head | — |
| 4 | A5 evidence-conditioned move set evaluation *if E3 says re-ranking pays* | D2 self-model plies; C3 proves-best scheduling | — |
| 5 | — | D3 endgame-solver port | volunteer-compute hardening |

Decision gates, stated so the results can veto the plan:

1. **E3** (stage 2): the two-round agent beats pick-by-sim at equal rollout
   budget → A5 and the full loop proceed; it doesn't → the evidence loop's
   deployment form is reconsidered. The fallback is still valuable: better sims,
   better scheduling, and the move set evaluation model alone are an engine
   improvement without any second round.
2. **A4** (stage 3): the recall/regret bar fires here, not at A3 — the
   sensitivity sweep on the position-evaluation-top-K agent prices what a
   recall miss costs in win rate and sets the bar, and the move set
   evaluation model must clear it *and* beat that agent at equal budget
   before it replaces exact evaluation anywhere. A3's stage-2 deliverable is
   the curves the bar is later read against; sequencing the gate at stage 2
   was circular, since the sensitivity measurement needs an agent A3 cannot
   provide.

## What is deliberately not here

- **Backtracking self-play**: rewind to a decision point and play out a
  *different* top-K move, for direct comparative signal from one position with
  the prefix cost amortized. Needs a `.slog` extension for branch points and a
  branching mode in `GameRunner`; parked until a training signal is demonstrably
  data-diversity-limited.
- **Standard (hidden-leave) Scrabble**, and with it everything belief: the
  posterior wired into the sims, sequential belief carried across turns, and
  the full design.md §3 system. The variant defers all of it. Returning means
  regenerating data and retraining, not redesigning, because nothing in this
  roadmap is specific to an information condition.
- **Search-derived knowledge buffers beyond the evidence loop** — still the
  long-range shape, but every nearer rung must fail to justify skipping to it.

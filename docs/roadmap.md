# Scribblez Project Roadmap

Scribblez aims to beat existing Scrabble engines by replacing their
context-blind static evaluation and naive rack inference with learned,
belief-aware evaluation ([design.md](design.md)). This document is the plan for
getting there: what is built, the agent everything converges on, and the tracks
that remain — each with its own experiment and its own fallback, priced before
it earns permanence.

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

The first three are built and render on the dashboard. Match play is run by hand
today; automating it is A1 below, and it is a prerequisite rather than a
nicety — every remaining track's final readout is match play, because root
cross-entropy demonstrably cannot see re-ranking value.

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
- **Sim quality has two independent limiters, now separately measurable.**
  Rollout variance (attacked by value-truncated rollouts) and opponent-rack
  uncertainty (attacked by belief, and bounded above by the open-leaves
  information condition, which hands the sims an exact rack posterior).

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
sims: racks from belief    ── ply 1..2: our own stack (move set eval + belief)
      inference        [B]  ── middle plies: HastyBot (WMP greedy)
      │                     ── bag empty: depth-limited endgame solver
      ▼
evidence-conditioned re-rank of all N                 [A/e]
      │
      ▼
sim the promoted moves, pick by sim
```

The tracks are separable, but they compound: better belief makes sims sharper,
sharper sims make evidence and the proves-best labels more trustworthy, and a
self-model rollout policy makes the ply-1 reply distribution (which is what the
evidence maps actually read) match reality.

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

- **A0 — the sim agent baseline.** `SimRunner` is offline-only today, and the
  engine has no simming agent at all; wrapping it as one (candidates → sim →
  pick by sim) is small and unblocks everything else, since it is both the
  baseline the move-set-evaluation agent must beat and the harness in which B,
  C, and D get their match readouts.
- **A1 — automated match eval.** Periodic `play_game` matches during training
  with win-rate curves and a sequential significance test on the dashboard.
- **A2 — target generation at scale.** The `.mset` sidecar and its generator
  ([move_set_eval_target_log.h](../engine/include/training/move_set_eval_target_log.h),
  [move_set_eval_target_generator.cpp](../engine/apps/move_set_eval_target_generator.cpp))
  are built and run locally. Remaining: a dashboard workload so the cloud fleet
  generates them, following the position-evaluation workload's shape
  ([position_eval_workload.md](position_eval_workload.md)).
- **A3 — move-set-evaluation v1 + the recall bar.** The model, dataset, trainer,
  and gate metric exist as a lean local loop
  ([py/scribblez/move_set_eval/](../py/scribblez/move_set_eval/)). Remaining:
  train at scale, fold into the generational lifecycle, and set the bar. The
  headline metric is not target MSE but **top-K recall against the position
  evaluation model's ranking** plus rank correlation over held-out candidate
  sets — recall is the filter's one job.
- **A4 — the move-set-evaluation agent.** Needs an ONNX export path and an
  engine-side runtime, which the model does not yet have. Then top-K by the move
  set evaluation model → sim → pick by sim, matched against A0 and against the
  position-evaluation-top-K agent.
- **A5 — evidence-conditioned move set evaluation** (steps 5–6 of
  [sim_residual_feedback.md](sim_residual_feedback.md)): the fusion stage
  migrates from the kill-test's position-evaluation harness onto the move set
  evaluation model, enabling the two-round re-rank. Gated on E3.

## Track B: rack inference — belief, cheapest first

The sim interface is already built and stable: `SimPosition.opp_leave` seeds any
known part of the opponent's rack, and per-rollout-index sampling preserves
common random numbers. Belief work is therefore *sampling policy*, not sim
plumbing.

Today's hidden-information sims draw the opponent's rack uniformly from the
unseen pool, which is only correct when they just bingoed. Everywhere else that
discards the information in their last move — a misspecification the
evidence-conditioned models would otherwise learn from. So inference is a
data-correctness fix for track A as much as a strength lever for the agent.

The plan is to port Macondo's Bayesian rack inference (its `rangefinder`
package, the machinery behind `SIMMING_INFER_BOT`) rather than build the
design.md belief system. Its scheme is posterior ∝ prior × likelihood, with a
multivariate hypergeometric prior over leaves drawn from the unseen pool, a
softmax likelihood P(observed play | leave) over how good every play would have
been under that hypothesis, exhaustive enumeration when the leave space is small
and importance sampling from the prior otherwise (where the proposal *is* the
prior, so the weight is likelihood only), and posterior sampling inside the sim
hedged by a fallback to a uniform rack when coverage is thin.

- **B1 — the inference core, with a static-equity likelihood.** Macondo scores
  each hypothesis with a 200-iteration two-ply mini-sim — roughly 100 ms per
  hypothesis, which is why it caps enumeration at 750 leaves and gives inference
  a 20-second budget. We use HastyBot static equity instead: one move generation
  plus an equity pass, some three orders of magnitude cheaper, which moves the
  enumerate-vs-sample knee out to thousands of leaves and makes the whole thing
  affordable inline. On our own data this is not a compromise but an
  improvement: the opponent in a `.slog` *is* the equity argmax, so the equity
  softmax is the exactly-correct generative model, where Macondo's mini-sim is
  an approximation of its own simming bot. The temperature does not carry over —
  Macondo's is on a win-probability log-odds scale, ours is in equity points —
  so it needs its own sweep.
- **B2 — price it offline against ground truth.** Replay recovers the
  opponent's true leave, so the posterior can be scored directly: its log-loss
  on the true leave against the prior's (the information gain in nats), the true
  leave's posterior mass, and per-tile marginal calibration, sliced by tiles
  kept and bag size. This sets the temperature, settles how to hedge thin
  coverage, and prices the entire track before a single sim or match runs.
  Caveat: on HastyBot self-play the likelihood model is exactly right, so these
  numbers are an upper bound; repeat on a neural-agent corpus for the
  misspecified case.
- **B3 — the three-arm information-condition study.** Regenerate the kill-test
  corpus with hidden, inferred, and open-leaves sims at matched scale.
  Open-leaves is the ceiling any belief system can reach and hidden is the
  status quo, but *inferred* is the arm that ships: two arms give the ceiling,
  three give the fraction cheap inference actually collects. **Gate:** if
  inference collects most of the open-leaves gain, the track is done and B4
  stays parked; if the gap is wide, B4 is funded; if open-leaves itself buys
  little, the budget moves to C and D. Macondo measures perfect-leave knowledge
  at about 3.3 points of win rate over its simming bot and its inference at
  about 1.9 of those, so a modest result is the expected one.
- **B4 — learned belief** (design.md §3: encoder/decoder + compressor +
  rejection traces). The full system, gated on B3.

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
  late-vs-early phase gradient says this is the biggest sim-quality lever. Keep
  an **anchor fraction** of terminal rollouts per candidate — a ground-truth
  tether and a free measurement of the value model's bias. Costs to accept:
  `.sobs` artifacts become model-versioned, and sims start contending for the
  GPU (the contention-manager regime
  [generational_training.md](generational_training.md) plans for).
- **D2 — self-model plies.** Plies 1–2 of each rollout played by our full stack
  (move set evaluation top-1 or a temperature sample, belief-sampled racks),
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
  consumers: B3's matched-scale runs, then A2's target generation.
- **E2 — match harness statistics** (with A1). Match play needs its own
  discipline — paired seeds and racks across agents (the CRN idea at the match
  level) and sequential stopping — which the harness should own so experiments
  do not reinvent it.
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

| Stage | Spine (A) | Sims (B/C/D) | Enablers (E) |
|---|---|---|---|
| 1 | A0 sim agent; A1 match eval | B1 inference core; **B2 offline pricing** | E1 fleet lands; E2 match statistics |
| 2 | A2 fleet target generation; A3 move-set-evaluation v1 + recall bar | **B3 three-arm study**; D1 truncated rollouts; C1 novelty dedup | **E3 re-ranking experiment** |
| 3 | A4 move-set-evaluation agent | C2 proves-best head | — |
| 4 | A5 evidence-conditioned move set evaluation *if E3 says re-ranking pays* | D2 self-model plies; C3 proves-best scheduling | — |
| 5 | — | D3 endgame-solver port; B4 learned belief *if B3 warrants* | volunteer-compute hardening |

Decision gates, stated so the results can veto the plan:

1. **B2** (stage 1): the posterior's information gain over the prior, measured
   against known ground truth, decides whether inference is worth simming with
   at all — for a fraction of the cost of finding out by match play.
2. **B3** (stage 2): inference recovers most of the open-leaves gain → the track
   is done; a wide gap → B4 is funded; a small open-leaves gain → belief is
   deprioritized and C and D take its budget.
3. **E3** (stage 2): the two-round agent beats pick-by-sim at equal rollout
   budget → A5 and the full loop proceed; it doesn't → the evidence loop's
   deployment form is reconsidered. The fallback is still valuable: better sims,
   better scheduling, and the move set evaluation model alone are an engine
   improvement without any second round.
4. **A3** (stage 2): the move set evaluation model's top-K recall against the
   position evaluation model must clear a bar — set from A4's sensitivity, how
   much win rate a recall miss costs — before it replaces
   position-evaluation-top-K anywhere.

## What is deliberately not here

- **Exporting or serving any information-condition model** (open-leaves is an
  instrument, not a product path).
- **Backtracking self-play**: rewind to a decision point and play out a
  *different* top-K move, for direct comparative signal from one position with
  the prefix cost amortized. Needs a `.slog` extension for branch points and a
  branching mode in `GameRunner`; parked until a training signal is demonstrably
  data-diversity-limited.
- **Sequential belief** — carrying the rack posterior forward across turns
  rather than recomputing it from the last move alone. Macondo has not built it
  either; it belongs with B4 if B3 says the gap is worth closing.
- **The full design.md belief system (B4) and search-derived knowledge buffers
  beyond the evidence loop** — both remain the long-range shape, but every
  nearer rung must fail to justify skipping to them.

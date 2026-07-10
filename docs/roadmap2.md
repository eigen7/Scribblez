# Roadmap 2: from evidence to a decision loop

The successor to [roadmap.md](roadmap.md), which carried the project through
the position evaluation model, the generational training pipeline, self-play diversification, and the
validation machinery. This document plans the next stage: turning a
well-validated value model into a *decision loop* — candidate scoring,
belief-aware simulation, evidence-conditioned re-ranking — and doing it the
way the sim-evidence work was done: every component priced by an experiment
before it earns permanence.

## Where the evidence stands

The sim-evidence kill-test
([sim_obs_experiment_results.md](sim_obs_experiment_results.md),
[sim_residual_feedback.md](sim_residual_feedback.md)) established:

- **The mechanism is real.** Conditioning the position evaluation model on sim evidence improves
  held-out outcome prediction with clean controls; the deployment-shaped
  (leave-one-out) transfer gain is smaller but significant, tail-concentrated,
  and late-game-loaded — the signature of evidence mattering exactly where
  decisions are contested.
- **Root-value accuracy is saturating.** A 200-rollout sim and the trained
  trunk are roughly equal, highly correlated estimators of root WLD; fusing
  them buys thousandths. The remaining prizes are *decision quality* (which
  move gets played) and *sim quality* (what the rollouts know), not root CE.
- **Sim quality has two independent limiters, now separately measurable.**
  Rollout variance (attacked by value-truncated rollouts) and opponent-rack
  uncertainty (attacked by belief; priced by the open-leaves information
  condition, which hands the sims an exact rack posterior — the ceiling any
  belief system can reach). An open-leaves pilot showed a ~5× larger transfer
  gain at small scale; the matched-scale run decides.

Meanwhile the distributed data-generation fleet
([cloud_compute.md](cloud_compute.md)) is being built in parallel, which turns
"matched-scale run" from a laptop-week into an overnight job.

## The destination

One picture of the agent this roadmap builds toward, with each component
labeled by its track:

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

The tracks are separable — each has its own experiment and its own fallback —
but they compound: better belief makes sims sharper, sharper sims make
evidence and the proves-best labels more trustworthy, a self-model rollout
policy makes the ply-1 reply distribution (which is what the evidence maps
actually read) match reality.

---

## Track A: the move set evaluation model — the spine

Everything else attaches to the move set evaluation model: it is the candidate filter, the host of
the proves-best head, the re-ranking surface for evidence, and eventually the
first plies of the rollout policy. The design (board encoder + move encoder +
single-pass cross-attention + exchange head) is specified in
[roadmap.md](roadmap.md) Phase 4 and unchanged.

- **A1 — quality gate + automated match eval.** Finish Phase 3.4: periodic
  automated `play_game` matches during training with win-rate curves (and a
  sequential significance test) on the dashboard. This is a prerequisite, not
  a nicety: every downstream track's final readout is match play, because
  root CE demonstrably cannot see re-ranking value.
- **A2 — target generation.** Label-a-subset position-evaluation targets stored alongside
  `.slog` (the Phase 4 plan), generated on the cloud fleet. The `.sobs`
  sidecar pattern (identity-addressed per-position artifacts, mode flags,
  atomic writes) is the template.
- **A3 — move-set-evaluation v1 + the recall metric.** Train the distillation model. The
  headline metric is not target MSE but **top-K recall against the position evaluation model's
  ranking** (does the move set evaluation model's top-K contain the moves the position evaluation model would pick?) and
  rank correlation over held-out candidate sets — recall is the filter's one
  job ([lexical_features_for_value.md](lexical_features_for_value.md)).
- **A4 — move-set-evaluation agent.** Top-K by the move set evaluation model → sim → pick by sim; match it against
  vanilla HastyBot and against the position-evaluation-top-K agent. This is the first
  end-to-end product of the new loop and the baseline every later component
  must beat.
- **A5 — evidence-conditioned move set evaluation** (steps 5–6 of
  [sim_residual_feedback.md](sim_residual_feedback.md)): the fusion stage
  migrates from the kill-test's position-evaluation harness onto the move set evaluation model, enabling the
  two-round re-rank. Gated on E3 (below) showing the re-rank is worth a
  round trip.

## Track B: rack inference — belief, cheapest first

The interface is already built and stable: `SimPosition.opp_leave` seeds any
known/inferred part of the opponent's rack, and per-rollout-index sampling
preserves CRN. Belief work is therefore *sampling policy*, not sim plumbing.

- **B0 — price the track before building it.** Two matched-scale runs
  (~100k positions each, fleet-generated): hidden vs open-leaves kill-tests.
  The open-leaves gain minus the hidden gain is the total value of perfect
  belief to the evidence loop. **Gate:** if the delta is small, B1–B3 drop
  down the priority list and the track's budget moves to D and C.
- **B1 — exact leave enumeration (keep-1/keep-2).** When the opponent's last
  move used 5–6 tiles (or exchanged all but 1–2), enumerate all ≤ ~378
  possible leaves, score each hypothesis with a forced-tile shadow search
  (max-equity must not beat the observed move by more than ε), and sample
  from the resulting exact posterior. Covers ~22% of positions on top of the
  ~24% already exact (post-bingo / not-yet-acted). On HastyBot self-play the
  likelihood is a near-indicator (the opponent *is* the greedy policy), with
  an ε-gap tolerance and a fallback to uniform when no hypothesis survives.
  Shares the "moves that must use tile X" search primitive with the
  contingent-map plan.
- **B2 — accept/reject for the general case.** Macondo-style plausibility
  weighting for the positions enumeration can't reach (keep-3+): sample a
  rack from the pool, reconstruct the pre-move rack, compute the best
  available move, and weight the sample by the observed move's plausibility
  (equity-gap softmax; near-indicator on self-play data). Unlike Macondo,
  fall back to the B1/B0-exact cases where available rather than to uniform
  filler. Note the self-model assumption: plausibility is scored against
  *our* policy's preferences, which is exactly right for self-play data and
  the honest default against unknown opponents.
- **B3 — learned belief model** (design.md §3: encoder/decoder + compressor +
  rejection traces). The full system. Gated on B1/B2 capturing meaningfully
  less than the B0 ceiling — if cheap inference closes most of the gap, B3
  stays parked.

## Track C: sim scheduling — spend rollouts where they buy information

From the candidate-selection analysis in
[sim_residual_feedback.md](sim_residual_feedback.md): the next candidate to
sim should be the one most likely to *prove best* — good on its own, and
different enough from the already-simmed candidates to beat them.

- **C1 — v0 diversity.** Footprint/lane-overlap novelty penalty at top-K
  selection time. Hours of work; also directly improves evidence diversity
  for A5.
- **C2 — proves-best head.** Per-move head on the evidence-conditioned move set evaluation model
  predicting the probability that the candidate's sim strictly exceeds the
  best-so-far. Labels are free from the CRN sims already sitting in every
  `.sobs`: any evidence prefix plus a held-out simmed candidate is a
  labeled row.
- **C3 — adaptive scheduling.** Propose by the proves-best head; evaluate
  batched (B=K) vs sequential (B=1) schedules. **Readout:** decision quality
  at a fixed rollout budget (match play), and budget required for fixed
  decision quality. The prize is real at deployment: sims dominate think
  time, so a 2× budget saving is a 2× stronger agent per second.

Dependency: the proves-best target is a function of the evidence set, so C2
wants the evidence-conditioned move set evaluation model (A5) as its host. A stopgap head can be
prototyped earlier on the kill-test's evidence-conditioned position-evaluation
harness (`sim_evidence/model.py`) to validate the target and labels.

## Track D: the rollout policy ladder

Current rollouts are HastyBot-to-the-end. Each rung below changes sim
semantics — so each lands behind a `.sobs` flag/version, gets validated by
the paired kill-test machinery, and then by match play.

- **D1 — value-truncated rollouts** (design.md §5.2): sim a few plies, read
  the position evaluation model's value at the horizon. The kill-test's 8× late-vs-early phase
  gradient is direct evidence this is the biggest sim-quality lever —
  truncation manufactures late-game-quality (low-variance) evidence at every
  phase. Keep an **anchor fraction** of terminal rollouts per candidate: a
  ground-truth tether in the evidence, and a free per-position measurement of
  the value model's bias. Costs to accept: `.sobs` artifacts become
  model-versioned (record the checkpoint hash; regenerate per generation),
  and sims start contending for the GPU — the contention-manager regime
  [generational_training.md](generational_training.md) already plans for.
- **D2 — self-model plies.** Model the opponent (and our own next reply) as
  using our full stack: ply 1–2 of each rollout are played by the move set evaluation model's top-1
  (or a temperature sample) with belief-sampled racks, then HastyBot to the
  horizon/end. This matters more than a generic policy upgrade because the
  evidence maps *read exactly plies 1–2* (the opponent's reply and our
  follow-up): upgrading those plies upgrades the evidence at its point of
  consumption. Cost: move-set-evaluation inference inside rollouts → batched leaf
  evaluation on the game-pool substrate; respect the roadmap.md rollout note
  (full-N passes at shallow plies only — deeper plies use cheap
  policies).
- **D3 — endgame solver for late-game rollouts.** Macondo's negamax solver
  is configurable in exactly the ways needed: `Solve(ctx, plies)`
  (depth-limited and time-limited), first-win-only mode (win/loss without
  exact spread — much faster), transposition table, iterative deepening, and
  multithreading. It is Go, so the path is a **port of a depth-limited
  negamax endgame solver into the engine** (the WMP precedent), not a config
  flag. Deployment shape: rollouts switch from HastyBot to the solver when
  the bag empties, with the budget scaled by the root's distance to the end —
  first-win-only and shallow plies for mid-game rollouts that happen to reach
  the endgame, deeper exact solves when the root itself is near the end
  (where the kill-test showed evidence matters most anyway). Validate with a
  bag-≤-N variant of the kill-test slices and endgame-position match play.

Ladder order is D1 → D2 → D3 by measured-value-per-effort: D1 is supported by
existing evidence, D2 depends on A3, D3 is the largest port with the most
localized payoff.

## Track E: scale and readouts — the enablers

- **E1 — cloud fleet** ([cloud_compute.md](cloud_compute.md), in flight).
  First consumers: B0's two matched-scale runs, then A2's target generation.
- **E2 — automated match harness** (with A1): every track's terminal metric.
  Match play needs its own statistics discipline (paired seeds/racks across
  agents — the CRN idea at the match level — and sequential stopping), which
  the harness should own so experiments don't reinvent it.
- **E3 — the re-ranking experiment (step-6-lite).** The pivotal readout the
  root-CE experiments structurally cannot provide: match play between
  (a) pick-by-sim over top-K and (b) a two-round agent that sims top-K,
  re-ranks the *unsimmed* candidates with the evidence-conditioned model,
  sims the promoted moves, and picks. The LOO transfer gain says the
  re-ranker has signal; E3 asks whether it changes picked moves often enough
  — and correctly enough — to win games. Runnable with the kill-test's
  position-evaluation-based fusion model before the move set evaluation model exists (re-scoring a few dozen
  candidates with the position evaluation model per decision is affordable in an eval harness);
  A5 productionizes it only if E3 says yes.

---

## Sequencing

Effort splits into three lanes that can run concurrently (one person + fleet:
lead with the spine, interleave the others as experiments block on data).

| Stage | Spine (A) | Sims (B/C/D) | Enablers (E) |
|---|---|---|---|
| 1 | A1 match harness; A2 target-gen prototype | — | E1 fleet lands; **B0 matched-scale hidden + open-leaves runs** |
| 2 | A3 move-set-evaluation v1 (recall metric) | D1 truncated rollouts (kill-test validated); C1 novelty dedup | E3 re-ranking match experiment (position-evaluation-based) |
| 3 | A4 move-set-evaluation agent baseline | B1 leave enumeration *if B0 says belief pays*; C2 proves-best head (prototyped on the kill-test harness) | — |
| 4 | A5 evidence-conditioned move set evaluation *if E3 says re-ranking pays* | D2 self-model plies; C3 proves-best scheduling | — |
| 5 | — | D3 endgame-solver port; B2 accept/reject; B3 learned belief *if the gap to B0's ceiling warrants* | volunteer-compute hardening |

Decision gates, stated so the results can veto the plan:

1. **B0** (stage 1): open-leaves gain ≫ hidden gain at matched scale →
   belief track funded (B1 next); roughly equal → belief deprioritized, D/C
   take its budget.
2. **E3** (stage 2): the two-round agent beats pick-by-sim at equal rollout
   budget → A5 and the full loop proceed; it doesn't → the evidence loop's
   deployment form is reconsidered (the fallback is still valuable: better
   sims + better scheduling + the move set evaluation model alone are an engine improvement without
   any second round).
3. **A3** (stage 2): the move set evaluation model's top-K recall against the position evaluation model must clear a bar
   (to be set from A4's sensitivity — how much win rate a recall miss costs)
   before it replaces position-evaluation-top-K anywhere.

## What is deliberately not here

- **Exporting/serving any information-condition model** (open-leaves is an
  instrument, not a product path).
- **Backtracking self-play** ([roadmap.md](roadmap.md) Phase 2) — still a
  good idea, still parked until a training signal is demonstrably
  data-diversity-limited.
- **The full design.md belief system (B3) and search-derived knowledge
  buffers beyond the evidence loop** — both remain the long-range shape, but
  every nearer rung must fail to justify skipping to them.

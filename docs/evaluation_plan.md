# Evaluation: what is established, and how the finished agent gets measured

[roadmap.md](roadmap.md) is an implementation plan and contains no experiments.
This document holds the other half: the measurements that already shaped the
design, the machinery that produces them, and the evaluation to run **once the
agent in the roadmap is built**.

The ordering is deliberate. Evaluation runs after implementation, not between
its steps. Nothing in the roadmap is gated on a result here.

## The evaluation machinery

No single metric suffices; four complementary evals, and the failure mode each
catches:

| Eval | What it tests | Failure mode it catches |
|---|---|---|
| Monotonicity probes | Structural coherence | Nonsensical evaluations |
| Calibration testing | Probabilistic accuracy | Structurally sound but biased |
| Monte-Carlo comparison | Absolute value accuracy | Divergence from deep-search ground truth |
| Match play | Downstream utility | Calibrated but not *useful* for move selection |

All four are built and render on the dashboard. Match play is the `match_eval`
worker, which plays sequential-test-checked paired matches for each exported
checkpoint against a fixed baseline.

**Match discipline** (A1/E2) lives in the harness
([harness.py](../py/scribblez/match_eval/harness.py) over the engine's
`--paired` mode, `scribblez/stats.py`): paired seeds and racks across arms — the
CRN idea at the match level — and sequential stopping, so experiments do not
reinvent it. The `match_arms` workload runs N named player-0 specs against one
fixed opponent under a shared base seed.

Two known gaps in that machinery, both worth closing before the final readouts:

- **Full tile-order CRN.** The `Bag` needs reshaping into a seeded permutation;
  today the shared-seed draw streams diverge once two arms' replenishment counts
  differ.
- **Per-pair result storage in `match_arms`.** The table stores a 5-bin
  pentanomial histogram per arm, which discards *which* pair is which. Arms
  share a base seed, so arm-vs-arm comparisons could be paired — but the stored
  aggregate throws that pairing away, and every cross-arm comparison is
  currently unpaired. Storing the per-pair score vector would sharpen those
  comparisons at no extra compute. (Within-arm CIs are already pentanomial and
  do exploit CRN.)

## Benchmark comparability

Macondo's published OracleBot result (~53.3% against BestBot, its SimmingBot)
pits a leave-knowing bot against one that plays without leave knowledge. An
opponent that declines to read a public leave produces exactly the games an
opponent that never knew about it would, since its policy does not model what
its opponent knows — so our agent against a leave-ignoring BestBot measures the
same thing. Only self-play differs, both seats there reading the leave.

Our own equivalent of BestBot is the sim agent
([sim_agent.h](../engine/include/agent/sim_agent.h)) — simming plus the endgame
solver — which doubles as the baseline and as the opponent that makes published
results comparable.

## What the kill-test established

The sim-evidence kill-test
([sim_obs_experiment_results.md](sim_obs_experiment_results.md),
[sim_residual_feedback.md](sim_residual_feedback.md)) passed, and its numbers
shaped the design:

- **The mechanism is real.** Conditioning the position evaluation model on sim
  evidence improves held-out outcome prediction with clean controls. The
  deployment-shaped (leave-one-out) transfer gain is smaller but significant,
  tail-concentrated, and late-game-loaded — the signature of evidence mattering
  exactly where decisions are contested.
- **Root-value accuracy is saturating.** A 200-rollout sim and the trained trunk
  are roughly equal, highly correlated estimators of root WLD; fusing them buys
  thousandths. The remaining prizes are *decision quality* and *sim quality*,
  not root cross-entropy.
- **The spatial half of the evidence is unproven at a root-WLD readout.** The
  `full` arm matched the `scalar` arm to ±0.0003. That readout structurally
  cannot show what the planes are for — promoting a move no earlier round ranked
  highly — which is why the roadmap carries them into the loop where promotion
  actually happens.
- **Open-leaves pilot.** ~5× larger transfer gain than the hidden-information
  arm, which is the measurement behind developing in the face-up-leaves variant.
- **Sim quality had two independent limiters**, and the variant removes one.
  Opponent-rack uncertainty is settled by rule; what remains is rollout
  variance, which value-truncated rollouts attack.

## What the distillation run established

[move_set_eval_results.md](move_set_eval_results.md) has the curves. In summary:
over 600 pairs the student reaches **recall@1 0.687** and **regret@1 0.0032** on
a full-sweep held-out slice, against the incumbent static-equity ranking's
**0.563** and **0.0090**.

The exchange-encoding decision was also settled by measurement: exchanges carry
their surrendered tiles in the uniform encoding, and a dedicated leave-encoder
head was **rejected** — on the post-fix holdout, exchange rank regret came in at
0.0026 against a 0.0098 leave-value baseline, at or below overall regret@1, so
the complexity had nothing left to buy.

## The sensitivity sweep: a null

Run 2026-08-13 on the `a4-sensitivity-sweep` tag: 8 arms × 400 games against
`--type=sim` (top-k 10 × 400 rollouts), face-up leaves, shared base seed. The
arms degraded the position-evaluation-top-K agent in controlled ways, the intent
being to price what a recall miss costs in win rate.

| arm | score | 95% CI |
|---|---|---|
| k1 | 0.5188 | [0.475, 0.562] |
| k2 | 0.5325 | [0.484, 0.581] |
| k3 | 0.5300 | [0.483, 0.577] |
| k5 | 0.5138 | [0.471, 0.557] |
| k10 | 0.5062 | [0.459, 0.554] |
| k10-drop05 | 0.4950 | [0.446, 0.544] |
| k10-drop10 | 0.4788 | [0.434, 0.523] |
| k10-drop25 | 0.4700 | [0.424, 0.516] |

**Nothing is significant.** Every interval contains 0.5 and every interval
overlaps every other. Two readings worth keeping:

- **`--sim-top-k` had no measurable effect** (weighted trend on log K:
  −0.005 per doubling, p = 0.59). `k1` — which returns the model's top-ranked
  move with *zero* rollouts — scored above `k10`, which spends 4000 rollouts a
  turn. The runtimes confirm `k1` simmed nothing: 50.4 min against ~100 min for
  the K=10 arms on identical game counts.
- **`--drop-best-prob` had the right sign and a monotone ordering**
  (0.506 → 0.495 → 0.479 → 0.470) but did not reach significance
  (slope −0.138 per unit drop, p = 0.28).

Resolving the observed 0.036 drop effect unpaired needs ~2.9× the pairs
(~585/arm, ~35 h). Paired cross-arm analysis would be cheaper, which is what the
per-pair storage gap above is about.

The sweep existed to set a recall bar for the distilled filter. It did not — the
honest output is an upper bound, that a recall miss is cheap, rather than a
number. The roadmap consequently does **not** gate on it, and the direct
measurement below replaces it.

## The evaluation to run when the build is done

In rough order of what each answers.

1. **The finished agent against the sim agent baseline**, face-up leaves, paired.
   The headline number, and the one comparable to published results via the
   BestBot argument above.
2. **The finished agent against `--type=mset-sim`** — the same stack with the
   evidence loop removed. This isolates what evidence conditioning and adaptive
   scheduling buy, which is the central claim of
   [sim_residual_feedback.md](sim_residual_feedback.md).
3. **Budget curves.** Decision quality at a fixed rollout budget, and budget
   required for fixed decision quality, against a fixed-top-K schedule. Sims
   dominate think time, so a 2× budget saving is a 2× stronger agent per second
   — this is where the sequential loop's early stopping shows up.
4. **The placement-plane ablation.** Evidence tokens with and without the model's
   predicted planes. The kill-test could not price them at a root-WLD readout;
   promotion is the readout that can, and this is the experiment that settles
   whether the plane-carrying `.mset` record (~950 B against v1's 36 B) earns
   its size.
5. **Rollout-ladder rungs**, each behind its `.sobs` flag: value truncation
   against terminal rollouts (with the anchor fraction giving a free read on the
   value model's bias), then self-model plies, then the endgame solver on a
   bag-≤-N slice.
6. **Whether the student can replace exact evaluation.** The distilled filter
   against per-candidate exact evaluation at equal rollout budget. Note this is
   properly a **non-inferiority** test: the student is ~13× cheaper per turn
   (measured: 497 moves/s against 38 at `--shortlist=0 --sim-top-k=1`), so a
   statistical tie is a win, and the margin should be declared before the run.

### Sizing

From the sweep above: at 200 pairs per arm, the pentanomial CI half-width on an
arm's score is ~0.043, and an unpaired arm-vs-arm difference has SE ~0.031. Any
comparison expected to land inside ~6 points needs either more pairs or the
paired cross-arm analysis the storage gap currently prevents. Head-to-head
matches (arm plays the comparison agent directly) avoid the problem entirely,
since the pentanomial then applies to the comparison itself.

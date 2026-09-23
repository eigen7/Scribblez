# Evaluation plan

[roadmap.md](roadmap.md) is an implementation plan and contains no
experiments. This document holds the other half: the machinery that produces
measurements, the measurements that already shaped the design, and the
evaluation to run **once the roadmap's agent is built**.

Evaluation runs after implementation, not between its steps. Nothing in the
roadmap is gated on a result here.

## The evaluation machinery

Four complementary evaluations were planned, each catching a different
failure mode:

| Eval | What it tests | Failure mode it catches | Status |
|---|---|---|---|
| Monotonicity probes | Structural coherence | Nonsensical evaluations | not built |
| Calibration testing | Probabilistic accuracy | Structurally sound but biased | not built |
| Monte-Carlo comparison | Absolute value accuracy | Divergence from deep-search ground truth | built |
| Match play | Downstream utility | Calibrated but not *useful* for move selection | built |

The Monte-Carlo comparison is the position_eval trainer's per-checkpoint eval
against the committed ground truth (`eval_win_mae`, the `eval_place_*`
metrics) and the dashboard's Positions tab
([react_dashboard.md](react_dashboard.md)). Match play is the `match_eval`
role, which plays a fixed-length paired match against a fixed baseline for
each exported checkpoint. Probes and calibration would belong in the same
per-checkpoint eval, rendered on the dashboard.

**Match discipline** lives in the harness
([harness.py](../py/scribblez/match_eval/harness.py), over the engine's
`--paired` mode, with statistics in `py/scribblez/stats.py`) so experiments do
not reinvent it: paired seeds and racks across arms (common random numbers at
the match level) and a fixed pair count per match. The `match_arms` workload
runs N named player specs against one fixed opponent under a shared base
seed.

Two gaps in that machinery are worth closing before the final readouts:

- **Full tile-order common random numbers.** `Bag` draws from a seeded RNG
  stream, so two arms given the same seed diverge as soon as their
  replenishment counts differ. Reshaping it into a seeded permutation of the
  tiles would keep the tile order shared.
- **Per-pair result storage in `match_arms`.** Each arm's result stores a
  5-bin pentanomial histogram (`pair_counts`), which discards which pair is
  which. Arms share a base seed, so arm-vs-arm comparisons could be paired,
  but the stored aggregate throws the pairing away and every cross-arm
  comparison is unpaired. Storing the per-pair score vector would sharpen
  those comparisons at no extra compute. (Within-arm confidence intervals are
  already pentanomial and do exploit the pairing.)

## Benchmark comparability

Macondo's published OracleBot result (~53.3% against BestBot, its simming
bot) pits a leave-knowing bot against one that plays without leave knowledge.
An opponent that declines to read a public leave produces exactly the games an
opponent that never knew it would, since its policy does not model what its
opponent knows. So our agent in the face-up-leaves variant, playing a
leave-ignoring BestBot equivalent, measures the same thing. Only self-play
differs, since there both seats read the leave.

Our BestBot equivalent is the sim agent
([sim_agent.h](../engine/include/agent/sim_agent.h), `--type=sim`): simming
plus the endgame solver. It is both the baseline and the opponent that makes
published results comparable.

## What the kill-test established

The sim-evidence kill-test
([sim_obs_experiment_results.md](sim_obs_experiment_results.md),
[plans/sim_residual_feedback.md](plans/sim_residual_feedback.md)) passed, and
its numbers shaped the design:

- **The mechanism is real.** Conditioning the position evaluation model on sim
  evidence improves held-out outcome prediction, with clean controls. The
  deployment-shaped (leave-one-out) transfer gain is smaller but significant,
  concentrated in the tail and in the late game: evidence matters exactly
  where decisions are contested.
- **Root-value accuracy is saturating.** A 200-rollout sim and the trained
  trunk are roughly equal, highly correlated estimators of root WLD, and
  fusing them buys thousandths. The remaining prizes are *decision quality*
  and *sim quality*, not root cross-entropy.
- **The spatial half of the evidence is unproven at a root-WLD readout.** The
  `full` arm matched the `scalar` arm to ±0.0003. That readout cannot show what
  the planes are for, promoting a move no earlier round ranked highly, which
  is why the roadmap carries them into the loop where promotion happens.
- **Open-leaves pilot.** The transfer gain was ~5× larger than in the
  hidden-information arm. This measurement is why development happens in the
  face-up-leaves variant.
- **Sim quality had two independent limiters**, and the variant removes one.
  Opponent-rack uncertainty is settled by rule; what remains is rollout
  variance, which value-truncated rollouts attack.

## What the distillation run established

[move_set_eval_results.md](move_set_eval_results.md) has the curves. In
summary: over 600 pairs the student reaches **recall@1 0.687** and
**regret@1 0.0032** on a full-sweep held-out slice, against the incumbent
static-equity ranking's **0.563** and **0.0090**.

Measurement also settled how exchanges are encoded. They carry their
surrendered tiles in the uniform move encoding; a dedicated leave-encoder head
was **rejected** because exchange rank regret on the holdout was already
0.0026, against a 0.0098 leave-value baseline and at or below overall
regret@1, leaving the extra head nothing to buy.

## The sensitivity sweep: a null

**Verdict: a null.** No arm differed significantly from any other, so the
sweep could not set the recall bar it was run for.

Run 2026-08-13 on the `a4-sensitivity-sweep` tag: 8 arms × 400 games against
`--type=sim` (top-k 10 × 400 rollouts), face-up leaves, shared base seed. The
arms degraded the position-evaluation top-K agent (`--type=neural-sim`) in
controlled ways, to price what a recall miss costs in win rate: `kN` sims the
model's top N moves (`--sim-top-k N`), and `-dropP` additionally excludes the
model's top-ranked move from the sim set with probability P per turn
(`--drop-best-prob`, P = 0.05, 0.10, 0.25).

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

Every interval contains 0.5 and every interval overlaps every other. Two
readings worth keeping:

- **`--sim-top-k` had no measurable effect** (weighted trend on log K:
  −0.005 per doubling, p = 0.59). `k1`, which plays the model's top-ranked
  move with *zero* rollouts, scored above `k10`, which spends 4000 rollouts a
  turn. The runtimes confirm `k1` simmed nothing: 50.4 min against ~100 min
  for the K=10 arms on identical game counts.
- **`--drop-best-prob` had the right sign and a monotone ordering**
  (0.506 → 0.495 → 0.479 → 0.470) but did not reach significance
  (slope −0.138 per unit drop, p = 0.28).

Resolving the observed 0.036 drop effect unpaired needs ~2.9× the pairs
(~585 per arm, ~35 h). Paired cross-arm analysis would be cheaper; that is
what the per-pair storage gap above costs.

The sweep existed to set a recall bar for the distilled filter. Its honest
output is an upper bound (a recall miss is cheap) rather than a number, so
nothing is gated on it, and the direct measurement below replaces it.

## The evaluation to run when the build is done

In rough order of what each answers.

1. **The finished agent against the sim agent**, face-up leaves, paired. The
   headline number, and the one comparable to published results by the
   BestBot argument above.
2. **The finished agent against `--type=mset-sim`**, the same stack with the
   evidence loop removed. This isolates what evidence conditioning and
   adaptive scheduling buy, the central claim of
   [plans/sim_residual_feedback.md](plans/sim_residual_feedback.md).
3. **Budget curves.** Decision quality at a fixed rollout budget, and the
   budget needed for a fixed decision quality, against a fixed-top-K schedule.
   Sims dominate think time, so a 2× budget saving is a 2× stronger agent per
   second. This is where the sequential loop's early stopping shows up.
4. **The placement-plane ablation.** Evidence tokens with and without the
   model's predicted planes. The kill-test could not price the planes at a
   root-WLD readout; promotion is the readout that can. This experiment settles
   whether the plane-carrying `.mset` record (~950 B against v1's 36 B) earns
   its size.
5. **Rollout-ladder rungs**, each behind its `.sobs` flag: value truncation
   against terminal rollouts, then self-model plies, then the endgame solver
   on a slice with at most N tiles in the bag.
6. **Whether the student can replace exact evaluation.** The distilled filter
   against per-candidate exact evaluation at equal rollout budget. This is a
   **non-inferiority** test: the student is ~13× cheaper per turn (measured:
   497 moves/s against 38 at `--shortlist=0 --sim-top-k=1`), so a statistical
   tie is a win, and the margin should be declared before the run.

### Sizing

From the sweep above: at 200 pairs per arm, the pentanomial CI half-width on
an arm's score is ~0.043, and an unpaired arm-vs-arm difference has SE
~0.031. Any comparison expected to land within ~6 points needs either more
pairs or the paired cross-arm analysis the storage gap prevents. Head-to-head
matches, where one arm plays the comparison agent directly, avoid the problem,
since the pentanomial statistics then apply to the comparison itself.

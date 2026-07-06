# Sim-evidence kill-test: results and conclusions

Results of the [sim_residual_feedback.md](sim_residual_feedback.md) kill-test
(roadmap step 3) — the go/no-go gate for the sim-evidence loop, testing its
load-bearing hypothesis in isolation: *does conditioning `M_post` on
Monte-Carlo sim evidence improve its outcome prediction?*

**Verdict: pass.** The evidence effect is real and statistically unambiguous
(5.7 standard errors on 10k paired holdout positions), the falsification
controls are clean, and the effect's *size* is bounded by an identified
ceiling — root-value information saturation between two redundant estimators —
rather than by any failure of the mechanism. The phase-sliced analysis shows
the gain scales with sim quality exactly as the mechanism predicts, which
directly motivates truncated (value-bootstrapped) rollouts as the next lever.

## Setup

- **Data**: 494 `.slog` files of HastyBot-vs-HastyBot self-play
  (`generate_kill_test_data.py`), one sampled eligible position per game:
  98,800 positions — 88,800 train / 10,000 holdout, split by file so games
  cannot leak.
- **Evidence** (`sim_obs_tool`): per position, top-10 candidates by HastyBot
  static equity, each simmed with 200 common-random-number HastyBot rollouts
  played to a natural game end, opponent racks sampled uniformly from the
  unseen pool. On this data the equity argmax is the move actually played, so
  candidate 0's sim directly estimates the training target.
- **Arms** (`kill_test.py`; identical architecture, parameters, and seed —
  only the evidence input differs):
  `none` (evidence zeroed — baseline), `shuffled` (real evidence permuted
  across positions — falsification control), `scalar` (sim summaries only),
  `full` (spatial planes + summaries).
- **Model**: `EvidencePostMoveModel`, 96-channel / 6-block trunk, 1.4 M
  params; AdamW 3e-4, batch 256, early stopping (patience 4). Decision
  metric: best held-out WLD cross-entropy.

## Results

Best held-out epoch per arm:

| arm | wld_ce | Δ vs none | brier | acc |
|---|---|---|---|---|
| none | 0.5110 | — | 0.3322 | 0.7473 |
| shuffled | 0.5116 | +0.0006 | 0.3324 | 0.7460 |
| scalar | 0.5045 | **−0.0065** | 0.3278 | 0.7469 |
| full | 0.5047 | **−0.0063** | 0.3282 | 0.7476 |

Paired per-position analysis (negative d = first arm better; SE over 10,000
paired holdout rows):

```
full      vs none       d=-0.0063 +/- 0.0011   win%=58.2   sign-p~0
scalar    vs none       d=-0.0065 +/- 0.0011   win%=57.1   sign-p~0
full      vs shuffled   d=-0.0069 +/- 0.0011   win%=58.9   sign-p~0
full      vs scalar     d=+0.0003 +/- 0.0002   win%=54.5
shuffled  vs none       d=+0.0006 +/- 0.0006   win%=48.6
```

Sliced (full vs none):

```
opp rack unbiased (n=2433)   d=-0.0043 +/- 0.0019
opp rack biased   (n=7567)   d=-0.0069 +/- 0.0014
late game  (<= 9 moves left) d=-0.0129 +/- 0.0026
mid game                     d=-0.0042 +/- 0.0017
early game (> 16 moves left) d=-0.0016 +/- 0.0014
```

Evidence-only yardstick (logistic regression on the sim scalars, **no board
input**), against the arms above:

```
played move (candidate 0 only)  holdout wld_ce = 0.5096
all candidates                  holdout wld_ce = 0.5098
```

## Analysis

**The gate is passed, with textbook controls.** The evidence gain is 5.7 SEs
from zero and holds against `shuffled`, which itself sits at a null vs `none`
(marginal statistics carry nothing; junk evidence is mildly harmful noise).
The two evidence arms agree with each other to ±0.0003 — the gain is
position-matched evidence, not artifact.

**The yardstick explains the small magnitude.** A bare logistic over one
candidate's raw sim scalars — no board, no trunk — matches the fully trained
baseline (0.5096 vs 0.5110). The trunk and a 200-rollout sim are therefore
two *roughly equal, highly correlated* estimators of root WLD, and fusing two
redundant estimators buys little by construction — exactly the observed few
thousandths. The constraint is information saturation at the root-value
readout, not fusion capacity. (Incidentally, this also shows the sims are
well-calibrated outcome predictors despite the uniform-rack bias.)

**The phase gradient is the mechanism's fingerprint.** The gain is 8×
larger in the endgame third (−0.0129) than the opening third (−0.0016) — a
monotone gradient tracking sim *quality*: near the endgame, rollouts are
close to exact and the model leans on them; early, a terminal rollout is 20+
moves of draw luck and adds little. This is direct empirical support for the
mechanism (the model uses evidence in proportion to its reliability) and
prices what better sims would buy everywhere: the late-game figure is the
gain where sims are trustworthy.

**Spatial planes are inert at this readout** (`full` ≡ `scalar` ± 0.0003).
Expected: the position-level WLD scalar has no use for per-move spatial
discrimination. The spatial planes' real test is per-move re-ranking (roadmap
steps 5–6), which this readout structurally cannot exhibit.

**The rack-bias slice is confounded and should not be over-read.** The gain
being *larger* on rack-biased positions (−0.0069 vs −0.0043) superficially
argues against the rack-inference-mismatch hypothesis, but the unbiased
subset (opponent just bingoed / has not yet acted) skews early-game, where
gains are small for phase reasons. The clean claim is only: there is no
positive evidence that rack bias is the binding limiter. A phase-controlled
cross-tab would settle it.

## Conclusions

1. **Proceed past the gate.** The loop's premise — sim evidence carries
   usable signal the network absorbs — is confirmed with clean controls.
2. **Truncated rollouts are the top-ranked next lever**: sim a small number
   of plies, then read `M_post`'s value at the horizon (the design.md §5.2
   sim shape). The phase gradient shows the payoff of trustworthy sims;
   truncation manufactures late-game-quality (low-variance) evidence at
   every phase, and cheaper per rollout besides. Costs to accept: evidence
   artifacts stop being model-independent (`.sobs` becomes
   generation-scoped), and everything past the horizon is scored by the
   value model itself — an anchor fraction of terminal rollouts keeps a
   ground-truth tether and measures the model's bias for free.
3. **Root-value CE is saturating — the next *experiment* should change
   readout, not polish this one.** The loop's actual payoff is re-ranking
   (promotion of moves the first pass misjudged), which root CE cannot
   measure. The step-6-lite version: match play between pick-by-sim and
   pick-by-evidence-conditioned-re-rank agents over HastyBot top-K.
4. **Poor-man's opponent-rack inference (leave enumeration) is deprioritized**
   pending a deconfounded rack slice — the data gives it no urgency.

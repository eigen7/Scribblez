# Sim-evidence kill-test: results and conclusions

The kill-test of [plans/sim_residual_feedback.md](plans/sim_residual_feedback.md):
the go/no-go gate for the sim-evidence loop. It tests the loop's
load-bearing hypothesis in isolation: *does conditioning the position
evaluation model on Monte-Carlo sim evidence improve its outcome prediction?*

**Verdict: pass.** The evidence effect is real and statistically unambiguous
(5.7 standard errors on 10k paired holdout positions), and the falsification
controls are clean. The effect's *size* is bounded by an identified ceiling,
root-value information saturation between two redundant estimators, not by
any failure of the mechanism. Sliced by game phase, the gain scales with sim
quality exactly as the mechanism predicts, which motivates truncated
(value-bootstrapped) rollouts as the next lever.

## Setup

- **Data.** 494 `.slog` files of HastyBot-vs-HastyBot self-play
  (`py/scripts/generate_kill_test_data.py`), one sampled eligible position per
  game: 98,800 positions, 88,800 train and 10,000 holdout, split by file so
  games cannot leak across the split.
- **Evidence** (`sim_obs_tool`). Per position, the top 10 candidates by
  HastyBot static equity, each simmed with 200 HastyBot rollouts under common
  random numbers, played to a natural game end, with opponent racks sampled
  uniformly from the unseen pool. On this data the equity argmax is the move
  actually played, so candidate 0's sim directly estimates the training
  target.
- **Arms** (`py/scripts/kill_test.py`; identical architecture, parameters and
  seed, differing only in the evidence input):
  - `none`: evidence zeroed (the baseline);
  - `shuffled`: real evidence permuted across positions (the falsification
    control);
  - `scalar`: sim summaries only;
  - `full`: spatial planes plus summaries.
- **Model.** `EvidencePositionEvalModel` (`py/scribblez/sim_evidence/model.py`),
  96-channel, 6-block trunk, 1.4M params; AdamW 3e-4, batch 256, early
  stopping with patience 4. Decision metric: best held-out WLD cross-entropy.

## Results

Best held-out epoch per arm:

| arm | wld_ce | Δ vs none | brier | acc |
|---|---|---|---|---|
| none | 0.5110 | — | 0.3322 | 0.7473 |
| shuffled | 0.5116 | +0.0006 | 0.3324 | 0.7460 |
| scalar | 0.5045 | **−0.0065** | 0.3278 | 0.7469 |
| full | 0.5047 | **−0.0063** | 0.3282 | 0.7476 |

Paired per-position analysis (negative d means the first arm is better; SE
over 10,000 paired holdout rows):

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

Evidence-only yardstick: logistic regression on the sim scalars, with **no
board input**, for comparison with the arms above:

```
played move (candidate 0 only)  holdout wld_ce = 0.5096
all candidates                  holdout wld_ce = 0.5098
```

## Analysis

**The gate is passed, with clean controls.** The evidence gain is 5.7 SEs
from zero and holds against `shuffled`, which itself is null against `none`.
The two evidence arms agree to ±0.0003. The gain comes from position-matched
evidence, not from an artifact.

**The yardstick explains the small magnitude.** A bare logistic regression
over one candidate's raw sim scalars, with no board and no trunk, matches the
fully trained baseline (0.5096 vs. 0.5110). The trunk and a 200-rollout sim
are two roughly equal, highly correlated estimators of root WLD, so fusing
them buys little by construction. The constraint is information saturation
at the root-value readout, not fusion capacity.

**The phase gradient is the mechanism's fingerprint.** The gain is 8× larger
in the late game (−0.0129) than in the early game (−0.0016), a monotone
gradient that tracks sim *quality*: the model uses evidence in proportion to
its reliability. The late-game figure prices what trustworthy sims would buy
everywhere.

**Spatial planes are inert at this readout** (`full` ≡ `scalar` ± 0.0003), as
expected: a position-level WLD scalar has no use for per-move spatial
discrimination. Their real test is per-move re-ranking, which this readout
cannot exhibit.

**The rack-bias slice is confounded.** A position is *unbiased* when the
sim's uniform sampling of the opponent's rack is exactly right: the opponent
has not moved yet, or their last move played all 7 tiles
(`py/scribblez/sim_evidence/slog_meta.py`). Elsewhere, what they kept biases
their rack. The gain is larger on biased positions, but the unbiased subset
skews toward the early game, where gains are small for phase reasons. The
defensible claim is only that there is no positive evidence that rack bias is
the binding limiter. A phase-controlled cross-tabulation would settle it.

## Conclusions

1. **Proceed past the gate.** The loop's premise, that sim evidence carries
   usable signal the network absorbs, is confirmed with clean controls.
2. **Truncated rollouts are the top-ranked next lever.** Sim a few plies,
   then read the position evaluation model's value at the horizon (the sim
   shape of [design.md](design.md) §5.2). The phase gradient shows the payoff
   of trustworthy sims, and truncation produces late-game-quality
   (low-variance) evidence at every phase, more cheaply per rollout. The
   costs: evidence stops being model-independent (`.sobs` files become tied
   to a model generation), and everything past the horizon is scored by the
   value model itself. (Since built as value-truncated rollouts;
   [roadmap.md](roadmap.md) item 2.)
3. **Root-value cross-entropy is saturating, so the next experiment should
   change the readout rather than polish this one.** The loop's real payoff
   is re-ranking, promoting moves the first pass misjudged, which root
   cross-entropy cannot measure. The lightweight version: match play between
   an agent that picks by sim and one that picks by evidence-conditioned
   re-ranking, both over HastyBot's top K.
4. **Cheap opponent-rack inference (enumerating their possible leaves) is
   deprioritized** until a deconfounded rack slice exists; the data gives it
   no urgency.

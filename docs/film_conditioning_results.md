# FiLM conditioning: results and post-mortem

**Verdict.** FiLM removed an expressivity ceiling but did not fix the defect.
With it, the trunk can gate a square's placement probability on the
opponent's leave, and it learns to. But the gated value stays anchored ~10×
below the truth, because the bottleneck is the training signal: a
bag-frequency prior learned into the weights, and a placement objective that
contributes ~1% of the trunk's gradient.

**The defect.** The face-up-leaves position evaluation model does not bind a
square's cross-check letters to the opponent's face-up leave. It reads the
letter masks through a fixed tile-frequency prior and ignores which letters
the opponent actually holds. At the time of these runs, the encoder already
carried each square's perpendicular cross-check set in the correct H/V
block; `use_film` added the model side, a multiplicative scalar→board
conditioning path alongside the trunk's additive injection (see
[model_architectures.md](model_architectures.md) and
[plans/sim_residual_feedback.md](plans/sim_residual_feedback.md)).

**Method.** The `face-up-leaves-film` run (`use_film: true`, otherwise
identical to the additive baseline `face-up-leaves-fixed5`), probed with
`py/scripts/position_eval/probe_crosscheck_binding.py`. That script reads the
placement heads as 15×15 planes, which the models of these runs had; current
position evaluation models predict move footprints instead, so it does not
run on today's checkpoints. The recurring test case is **pos-09 M7**: the
opponent holds G, GNU plays vertically there, and the Monte-Carlo truth for
the opponent placing a tile on M7 is **0.668**.

## Result 1 — FiLM engages, but does not close the gap

The binding mechanism the additive trunk could not express now exists.
Additive baseline vs. the FiLM run at generation 633 (M7, truth 0.668):

| signal | additive (fixed5) | FiLM (film) |
|---|---|---|
| M7 baseline Pr | 0.031 | 0.046 |
| availability sweep: G present → removed → leave-empty | 0.031 → 0.033 → 0.032 (flat) | 0.046 → 0.019 → 0.013 (gates) |
| letter-selectivity, G rank | 12/26 | 9/26 |
| tail p1 corr (large set) | 0.35 | 0.47 |
| cross-check-cell mean \|pred−truth\| | 0.029 | 0.021 |

The gate from the opponent's leave to placement is real and strengthened over
training: the ratio `Pr(G present) ÷ Pr(leave empty)` at M7 grew from ~1×
early to ~3.5× by generation 633, while the additive trunk shows no gating at
any generation. FiLM did what it was designed to do.

The *magnitude* did not converge. Over training, M7's baseline oscillates in a
**0.03–0.07 band and does not trend toward 0.668**:

```
gen   16    50   100   200   350   500   633
M7  0.023 0.033 0.034 0.028 0.072 0.075 0.046
```

The gate multiplies a baseline that stays anchored ~10× too low, and the
single-letter selectivity readout is still ordered by frequency (forcing the
mask to `{A}` scores higher than the real `{G}` hook). FiLM was necessary but
not sufficient.

## Result 2 — eval-mode / BatchNorm noise is not the explanation

The schedule-free optimizer arm recomputes every BatchNorm layer's statistics
for the averaged (deployed) weights before each checkpoint export
(`recalibrate_batchnorm` in `generational/optim.py`), over batches seeded per
generation. Re-running that recalibration on one checkpoint with six
different batch seeds moves M7 by **std 0.0006** (range 0.0541–0.0558), and
the gate ratio holds at ~2.4×. That is ~1% of the M7 value and 0.1% of the gap
to truth. The oscillation across generations is therefore genuine training
dynamics, not recalibration jitter, and the probe numbers are trustworthy.

## Result 3 — the tile-frequency prior is learned into the weights

Forcing M7's cross-check mask to each single letter, while zeroing different
input scalars (FiLM run, generation 650):

```
as-is scalars      : A.168 E.128 T.112 N.110 I.104 ... Q.000
unseen-pool zeroed : E.027 A.023 I.022 T.016 N.016 ... Q.000   (magnitude collapses ~6x)
ALL tile-counts 0  : E.100 S.092 N.056 R.055 T.052 ... Q.000   (order intact)
Spearman(all-zeroed order, Scrabble bag frequency) = 0.807
```

With every tile-count scalar zeroed (rack, unseen pool, opponent's leave) the
ranking is still shaped like bag frequency. So the network does not read a
frequency feature from its input. It has **learned** a readout weight for each
of the 26 cross-check letter planes, and gradient descent drove each weight
toward that letter's marginal placement rate, which is its bag frequency.
Separately, the input tile counts (chiefly the unseen pool) *scale* the
response: zeroing them collapses it ~6×.

This reframes the bottleneck. M7's hook letter G has a small learned weight
*because G is uncommon*. FiLM's leave gate multiplies that small weight, and
2.4× a small number is still small. To reach 0.668, the leave signal must
**override** the frequency prior for G, treating "opponent holds G" as making
G locally common, not merely scale it. The frequency prior is correct absent
leave information; it acts as a floor the current gate cannot lift far
enough.

## Result 4 — the placement objective barely reaches the shared trunk

Per-head gradient pull on the shared trunk, measured over a 128-position
held-out batch with Monte-Carlo targets (FiLM run, generation 650):

| head | ‖dL/d(trunk)‖ | λ | weighted pull |
|---|---|---|---|
| wld | 0.093 | 1.0 | 0.093 |
| score_diff | 71.9 | 0.0002 | 0.014 |
| opp_next_placement | 0.0016 | 0.5 | 0.0008 |
| self/win-placement heads | ~0.001 | 0.5 | ~0.0005 |

`opp_next_placement` contributes **0.8%** of the trunk's gradient; `wld`
outweighs it **113×**. Two effects compound. The trunk is overwhelmingly
shaped by the WLD objective (multi-task competition). And the placement
gradient is intrinsically tiny: its loss sits near its floor (0.061),
dominated by the easy majority of near-zero cells, so a rare high-value square
like M7 contributes almost nothing. (The per-head losses are flat from
generation 50 to 649; everything plateaued early.)

## Where the bottleneck is, and what to try next

FiLM removed the expressivity ceiling: the trunk can now form the conjunction
of leave and cross-check, and demonstrably does (Result 1). What remains is a
**training-signal** problem (Results 3–4). The placement objective exerts ~1%
of the trunk's gradient, that gradient is dominated by easy cells, and the
leave gate scales, rather than overrides, a frequency prior in which the
relevant letter is inherently small.

Two levers follow, in order:

1. **Isolate the objective** (tests multi-task dilution). Zero every loss
   weight except `opp_next_placement` and retrain a diagnostic tag. If M7
   climbs, the WLD and score-difference heads were starving the trunk and the
   fix is capacity allocation; if not, dilution is ruled out and the problem is
   the loss geometry below. This differs from per-cell reweighting of the
   placement loss, which had been considered and rejected: it removes
   competition between heads, not weighting within one. It yields a
   diagnostic-only model with no WLD head.
2. **Reachability renormalization.** Probe 3 (tail percentiles) shows FiLM
   improved the constrained tail (p1 correlation 0.35 → 0.47, cross-check-cell
   error −27%) but did not close it. Scaling each reachable cell's predicted
   marginal by `225/|reachable|` directly attacks the anchored-too-low
   magnitude: on a constrained board it scales up exactly the
   marginals of the few reachable cells, which is the M7 symptom.

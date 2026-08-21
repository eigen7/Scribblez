# FiLM conditioning: results and post-mortem

Context: [model_architectures.md](model_architectures.md) (the `use_film` trunk
option) and [sim_residual_feedback.md](sim_residual_feedback.md). The motivating
defect is that the face-up-leaves position-eval model does not bind a square's
cross-check letters to the opponent's face-up leave: it reads the letter masks
through a fixed tile-frequency prior and ignores which letters the opponent
actually holds. PR #90 fixed the encoder side (the cross-check planes now carry
the perpendicular cross-check set, in the correct H/V block); the `use_film` PR
added the model side (a multiplicative scalar→board conditioning path, the
missing half of the trunk's additive-only injection).

Method: the `face-up-leaves-film` run (`use_film: true`, otherwise identical to
the additive baseline `face-up-leaves-fixed5`), probed with
`py/scripts/position_eval/probe_crosscheck_binding.py`. The recurring test case is
**pos-09 M7**: the opponent holds G, GNU plays vertically there, and the
Monte-Carlo truth for the opponent placing a tile on M7 is **0.668**.

## Result 1 — FiLM engages, but does not close the gap

The binding mechanism the additive trunk could not express now exists. Comparing
the additive baseline against the FiLM run at gen 633 (M7, truth 0.668):

| signal | additive (fixed5) | FiLM (film) |
|---|---|---|
| M7 baseline Pr | 0.031 | 0.046 |
| availability sweep: G present → removed → leave-empty | 0.031 → 0.033 → 0.032 (flat) | 0.046 → 0.019 → 0.013 (gates) |
| letter-selectivity, G rank | 12/26 | 9/26 |
| tail p1 corr (large set) | 0.35 | 0.47 |
| cross-check-cell mean \|pred−truth\| | 0.029 | 0.021 |

The opp-leave→placement gate is real and strengthened over training — the ratio
`Pr(G present) ÷ Pr(leave empty)` at M7 grew from ~1× early to ~3.5× by gen 633,
where the additive trunk shows no gating at any generation. FiLM did what it was
designed to do.

But the *magnitude* did not converge. M7's baseline over training oscillates in a
**0.03–0.07 band and does not trend toward 0.668**:

```
gen   16    50   100   200   350   500   633
M7  0.023 0.033 0.034 0.028 0.072 0.075 0.046
```

The gate multiplies a baseline that stays anchored ~10× too low, and the
single-letter selectivity readout is still frequency-ordered (forcing the mask to
`{A}` scores higher than the real `{G}` hook). FiLM was necessary but not
sufficient.

## Result 2 — eval-mode / BatchNorm noise is not the explanation

The schedule-free arm recomputes every BatchNorm layer's statistics for the
averaged (deployed) weights before each checkpoint export (`recalibrate_batchnorm`
in `generational/optim.py`), over batches seeded per generation. Re-running that
recalibration on one checkpoint with six different batch seeds moves M7 by
**std 0.0006** (range 0.0541–0.0558); the gate ratio holds at ~2.4×. That is ~1%
of the M7 value and 0.1% of the gap to truth. The 0.03–0.07 oscillation across
generations is therefore genuine training dynamics, not recalibration jitter, and
the probe numbers are trustworthy. BatchNorm handling is ruled out.

## Result 3 — the tile-frequency prior is learned into the weights

Forcing M7's cross-check mask to each single letter and zeroing different input
scalars (film gen 650):

```
as-is scalars      : A.168 E.128 T.112 N.110 I.104 ... Q.000
unseen-pool zeroed : E.027 A.023 I.022 T.016 N.016 ... Q.000   (magnitude collapses ~6x)
ALL tile-counts 0  : E.100 S.092 N.056 R.055 T.052 ... Q.000   (order intact)
Spearman(all-zeroed order, Scrabble bag frequency) = 0.807
```

With every tile-count scalar zeroed (rack, unseen-pool, opp-leave) the ranking is
still bag-frequency-shaped. So the net did not read a frequency feature — it
**learned** a per-plane readout weight on each of the 26 cross-check letter planes,
and gradient descent drove those weights toward each letter's marginal placement
rate, which is bag frequency. Separately, the input tile-counts (chiefly the
unseen-pool) *scale* the response: zeroing them collapses it ~6×.

This reframes the bottleneck. M7's hook letter G has a small learned baseline
weight *because G is uncommon*. FiLM's leave-gate multiplies that small baseline;
2.4× a frequency-small number is still small. To reach 0.668 the leave signal must
**override** the frequency prior for G — treat "opponent holds G" as making G
locally common — not merely scale it. The frequency prior is correct behavior
*absent* leave information; it acts as a floor the current gate cannot lift enough.

## Result 4 — the placement objective barely reaches the shared trunk

Per-head gradient pull on the shared trunk, measured over a 128-position held-out
batch with Monte-Carlo targets (film gen 650):

| head | ‖dL/d(trunk)‖ | λ | weighted pull |
|---|---|---|---|
| wld | 0.093 | 1.0 | 0.093 |
| score_diff | 71.9 | 0.0002 | 0.014 |
| opp_next_placement | 0.0016 | 0.5 | 0.0008 |
| self/win-placement heads | ~0.001 | 0.5 | ~0.0005 |

`opp_next_placement` contributes **0.8%** of the trunk's gradient; `wld` dominates
it **113×**. Two compounding effects: the trunk is overwhelmingly shaped by the
wld objective (multi-task competition), and the placement gradient is *intrinsically*
tiny — its loss is near its floor (0.061), dominated by the easy near-zero majority
of cells, so a rare high-value square like M7 contributes almost nothing. (The
per-head losses are flat from gen 50 to 649; everything plateaued early.)

## Where the bottleneck is, and what to try next

FiLM removed the expressivity ceiling — the trunk can now form the
leave↔cross-check conjunction, and demonstrably does (Result 1). What remains is a
**training-signal** problem (Results 3–4): the placement objective exerts ~1% of
the trunk's gradient, that gradient is dominated by easy cells, and the leave-gate
scales rather than overrides a frequency prior in which the relevant letter is
inherently small.

Two levers follow, in order:

1. **Isolate the objective** (tests multi-task dilution). Zero every loss weight
   except `opp_next_placement` and retrain a diagnostic tag. If M7 climbs, the
   trunk was starved by wld/score-diff and the fix is capacity allocation; if it
   does not, dilution is ruled out and the problem is the loss geometry below.
   This is distinct from the previously rejected *cell-reweighting* — it removes
   inter-head competition, not intra-placement reweighting — and it yields a
   diagnostic-only model (no wld inference head).

2. **Reachability renormalization** (deferred in the roadmap: "revisit after FiLM
   using probe 3 as the read"). Probe 3 now shows FiLM improved the constrained
   tail (p1 0.35→0.47, cross-check-cell error −27%) but did not close it. The
   denominator-only `225/|reachable|` form directly attacks the anchored-too-low
   magnitude: on a constrained board it scales up exactly the marginals of the few
   reachable cells, which is the M7 symptom.

# Teacher POV calibration bias

**Status: diagnosis settled; fix not designed or built.** Nothing in the
trainer adds restoring force or the proposed gates yet ([Future
work](#future-work)).

**Problem.** The position evaluation model (the teacher) systematically
flatters the player whose point of view (POV) it evaluates from. At
`face-up-official` epoch 4414, the two seats' predicted final margins for the
*same game* sum to **+5.36 ± 0.12 points** instead of zero (30,510 adjacent
post-move row pairs, which alternate POV); the win-probability analog sums to
1.0163 ± 0.0006. Per seat that is about +2.5 points / +0.8% win probability.
It reproduces on the model's own training rows (dd +2.85 ± 0.15, dv
+0.0094 ± 0.0008), so it is in-distribution, not an artifact of the
measurement set.

**Diagnosis, in one paragraph.** The shipped +2.5 is essentially all
**drift**. The POV marginal is a nearly flat direction of the training
objective: removing the whole bias is worth about 1.5e-4 of the loss, 20×
below generation-to-generation loss noise. So the model's offset performs an
unpinned slow walk, and the deployed value is an accident of when the run
stopped. The same run's exports swung between −2.6 and +2.5 points and read
−0.6 at epoch 4000. Underneath the drift sit three stable, reproducible
components that net slightly *negative*: a Huber median-versus-mean wedge
(about −1 point, which the loss actively prefers); margin compression, S1
(the model's total response to the current lead is about 0.90 against
reality's 0.97); and a trailing-tempo bump, S2 (+4 points: "just moved, still
slightly behind" is overvalued). The schedule-free deployment then shapes
what ships: its unbounded 1/k weight average filters out fast jitter but lags
the slow walk and smooths it into large excursions.

**Decision.** The fix must add **restoring force**: a term the objective
actually feels. Optimizer and loss choices only move or reshape the bias.
Candidate terms, gates and a validation ladder are under [Future
work](#future-work).

This document corrects an earlier diagnosis
([PR #107](https://github.com/eigen7/Scribblez/pull/107)) whose central claim,
a "structural margin-expansion slope" of +0.21 points per point of current
lead, was an estimator artifact ([what it got
wrong](#what-the-earlier-diagnosis-got-wrong)). Two training runs were spent
on interventions derived from that claim; both were null. The diagnosis here
went through four rounds of adversarial review, and every load-bearing number
was cross-checked by at least two independent estimators.

## Methodology traps (read before measuring anything here)

These three traps produced the earlier misdiagnosis, and each is easy to fall
into again.

1. **Regressing prediction error on the current lead within a game is
   invalid.** Within one game the realized final takes two values (±D) while
   the current lead takes about 20, so centering on the game mean crushes
   d(realized)/d(lead) from 0.93 (pooled) to 0.69, while any smooth predictor
   keeps about 0.93. The estimator's own null is therefore about +0.24: an
   OLS fit of realized-on-lead, which has zero lead-conditional bias by
   construction, scores +0.245 ± 0.038 on it, *higher* than any real teacher.
   The earlier "+0.21 slope, identical across checkpoints" was this null.
   Use pooled estimators: bin by lead and compare within bins.
2. **Clustering by game within a bin changes per-bin profiles
   qualitatively.** The training estimand weights each (game, turn) by 1/n_g,
   which is within 0.3 points per bin of plain row weighting (verified).
   Clustering by games present in a bin up-weights a bin's brief visitors
   about 20× and flips signs (for example, lead in [80, 300): −1.9
   row-weighted against +8.5 clustered, on the same data). Report per-bin
   profiles row-weighted.
3. **Huber(δ=10) on a target spanning about ±60 points fits the conditional
   median.** 86% of residuals lie beyond the knee. The loss's own optimum has
   a mean bias of about −1 point (the outcome distribution's mean-minus-median
   gap), so a "biased" mean can be the loss working as intended. Report mean
   AND median calibration.

## The four components

Unless stated otherwise, numbers are for `face-up-official`, FP32, on fresh
self-play under the training configuration (`hastybot-endgame` in both
seats, `random_opening_mean` 2.0, face-up leaves). "dd" and "dv" are predicted
minus realized, for the score-diff and win-equity heads respectively, from
the POV seat at post-move rows.

**(a) Huber wedge, about −1 point.** The score-diff mean head's loss optimum
sits at a mean bias of −1.05. (An offset sweep puts Huber's minimum 3.5
points below the shipped point, whose residual mean is +2.45 against a median
of +3.54.) The WSD control run's long-run center, dd ≈ −1.5 / dv ≈ −0.0089
over 15 checkpoints, matches this equilibrium propagated through the
measured coupling between heads (about 155 points per unit of win
probability). The loss prefers this bias; only a mean-consistent loss removes
it.

**(b) Stable conditional structure: S1 and S2.** Replicated row-weighted
across two optimizers, two loss shapes, checkpoints from three independent
runs, and three disjoint game sets.

- **S1, margin compression.** The total response of the predicted final to
  the current lead, across states, is 0.899 ± 0.007 against a realized 0.970.
  The per-bin odd component reads −0.05 to −0.075 points per point beyond
  |lead| ≈ 30; two independent estimators give the same number. Perturbing
  the score-diff input scalar directly (±10 points at fixed states) shows
  the scalar channel itself responds at 0.974 ± 0.001. **The compression is
  therefore not in the scalar input channel.** The deficit lives in the net
  lead-correlated contribution of the other features (−0.075 per point),
  whose internal breakdown is unidentified. No input-encoding change can
  address it.
- **S2, trailing-tempo bump.** At post-move states where the mover is still
  slightly behind (lead in [−15, 0)), the realized E[final] is about −29
  while the model says about −22. Having just banked a move is overvalued
  precisely when that move failed to take the lead: +4 points, stable across
  checkpoints, and about +6 to +7 at ep4414 including its drift state.
  Conditional pair-sums, which are free of outcome noise, confirm it: the
  near-tied |lead| bin's pair-sum exceeds the mid-lead floor by +2.0 points
  dd / +0.023 dv, at more than 4σ. The leading-side twin, [0, 15), is
  approximately calibrated.

Weighted by the row distribution (mean POV lead +19.5, against which S1
acts), (a) and (b) **net about −0.5 to −1.5 points**. The stable structure
does not produce the shipped positive bias; the drift does.

**(c) Unpinned drift: the dominant term in what ships.** Flatness, measured:
removing the entire dv bias is worth 1.4e-4 nats of cross-entropy, against a
generation-to-generation CE standard deviation of 2.8e-3; the dd analog is
worth 1.6e-4 of weighted loss. Consequences, all measured:

- **The production number is where the run stopped.** The deployed exports
  (every 100 generations, across all 4414) wander in [−2.6, +2.5] with
  excursions of 700 to 1500 generations: a quiet band of ±0.4 until ep1500, a
  dive to −2.6 (ep1900), a climb to +2.4 (ep3300), back to −0.6
  (ep4000–4100), and +2.5 at 4414. The onset, ep1500–1600, precedes the
  trainer restart at generation 1773. Learning rate, rows per generation,
  loss weights and the data distribution (frozen HastyBot) were audited and
  flat. At onset the excursion is specific to the score-diff head (dv holds
  at −0.001 while dd dives): it lives in the least-pinned head
  (λ_sd = 2e-4).
- **The raw iterate's jitter is white.** Checkpoint-to-checkpoint σ is about
  1.6 points, white at a lag of one generation (lag-k RMS increments are flat
  from k=1 to 30; 40 consecutive WSD exports, paired rows). A WSD export is a
  lottery draw per checkpoint, centered on the (a)+(b) equilibrium, not on
  zero.
- **The drift is low-dimensional but not a constant offset.** Differences
  between checkpoints carry an offset *and* a tilt in response slope
  (ep4414 − ep1500 rises monotonically from +0.9 to +6.0 across lead bins,
  paired SE ±0.2).
- **Guard rail.** "Where the run stopped" applies to the aggregate only. The
  conditional structure (b) is stable across checkpoints and larger than any
  aggregate, so "the +2.5 was luck" must not be read as "nothing is wrong".

**(d) The 1/k averaging horizon (schedule-free runs).** The deployed model is
a uniform average of every iterate in the run. It suppresses (c)'s fast
jitter (the early deployed series stays within ±0.4 where raw WSD exports
swing ±1.6), but it lags and smooths the slow walk into the large
excursions. The live iterate's weight-space distance from the deployed
average grows monotonically (about 1500 → 2800 over the run) while deployed
quality stalls: eval_win_mae bottoms out at 0.0167 (ep3724), is flat from
about ep1000, and degrades to about 0.020 over the final stretch, at the same
time as the bias excursion. eval_sd_mean_mae is elevated at every excursion
peak; it already partly monitors (c) and belongs in the promotion gate now.

## Ruled out, each by direct measurement

- **Score-diff input resolution** (the earlier diagnosis's prescription),
  three ways. A full retrain with the scalar encoded on a 15-bump RBF basis
  left every metric unchanged
  ([PR #108](https://github.com/eigen7/Scribblez/pull/108), closed). The
  trained basis model demonstrably *uses* the basis (zeroing it moves win
  probability by 0.10) yet shows the same bias. And the direct perturbation
  test shows the scalar channel's response was never the deficit.
- **The mean-head loss shape as the driver of the shipped number.** Huber
  against MSE at matched ages ep100/200/339 differs by at most 0.2 points.
  The wedge (a) is an equilibrium shift of about −1, invisible under drift at
  young ages; the `sd-mean-mse` run also used a λ_sd 11× smaller, which
  weakens restoring force.
- **Generation-config mismatch** (greedy against solver endgames, random
  openings): changes realized POV outcomes by 0.03 points and dd by under 0.1,
  and the bias reproduces on the model's own training rows regardless.
- **The trainer's sampler.** Trainer-drawn WLD targets average
  0.4990 ± 0.0018, and score-diff targets about 0: no POV asymmetry in what is
  trained on.
- **BatchNorm recalibration at export:** +0.00014 win probability.
- **Trainer restarts.** The averaging weight is continuous through all three;
  the schedule-free state round-trips; the onset precedes the restart.
- **Non-stationary data.** The generator is frozen HastyBot, so the
  distribution is stationary by construction, and was audited.

## Impact while unfixed

The bias is common-mode within a position (the measured candidate-relative
margin is about 0), so **rankings, CRN-paired gains and the sim loop are
safe**. Absolute readouts are off by the current drift state (±2.5 points /
±0.8% win probability, with a sign that cannot be known in advance). That
blocks the planned sim-value second target stream for the teacher and any
absolute comparison across POVs. And by (d), long schedule-free runs quietly
pay a deployed-quality cost after about ep1000.

## Future work

Candidates, in intended order:

1. **F2′: an MSE (or other mean-consistent) score-diff mean loss, at restored
   effective weight.** MSE has about 11× Huber's gradient scale on these
   residuals; `sd-mean-mse` used λ = 1.77e-5 to match scales. Removes (a).
   Quality parity validated only to generation 358.
2. **F1″: a conditioned marginal-calibration term.** Match predicted against
   realized outcome marginals (all three WLD classes plus the score-diff mean)
   within bins of lead × turn, or of predicted value. A *global* marginal term
   cannot see S1 or the drift's tilt. Its weight is a real design problem:
   target a restoring force of about 1 to 10% of the CE gradient scale, tuned
   against the per-generation coherence metric, and gated on the EXPORTED
   model, since pinning the live iterate does not provably pin the average.
3. **F3: a bounded averaging horizon.** Because the jitter is white at lag 1,
   an average of about 4 *consecutive* exports already achieves the full √4
   reduction (σ 1.61 → 0.62, at the slow-center floor of ±0.6); striding buys
   nothing. This also caps (d)'s lag and stall.
4. **Gates, to start before any fix:** per-generation conditional pair-sum
   coherence on fresh self-play (SE 0.12 points per 1600 games, free of
   outcome noise), eval_sd_mean_mae, and row-weighted per-bin profiles (mean
   AND median). The pair-sum metric cancels structure that is odd in the lead
   by construction, so it cannot see S1 or the tilt; the coherence and
   profile gates complement each other. The coherence gate becomes a
   promotion gate under [generational_teacher.md](generational_teacher.md).

**Validation ladder, cheapest first.** Post-hoc per-bin recalibration is
already demonstrated: fitted on 1600 fresh games, it removes the aggregates
held out on two independent sets (dd +2.51 → −0.29 and +2.73 → −0.06, dv →
about 0) at zero CE cost. Next, a head-only fine-tune swapping Huber for MSE
(expected: about 1 point, the wedge). Then add F1″ terms, which tests whether
S1 and S2 can be removed in training. A full run only after those.

**Open items.** The roughly 1000-generation timescale of (c)'s excursions is
characterized, not explained; logging per-generation coherence on the next
run measures its spectrum for free. The true scalar-channel partial (about
1.0, against the measured 0.974) is unmeasured, and would only reshuffle S1's
internal attribution. Whether F1″ pins the *export* rests on an empirical
gate.

## What the earlier diagnosis got wrong

The earlier revision split the bias into a "structural slope" (+0.21 points
of over-prediction per point of lead, stable across checkpoints) plus a
drifting offset, and prescribed a nonlinear score-diff input basis. The
slope was trap 1: its stability across lineages and ages was the signature of
an estimator constant, not a model property. Its value equals each model's
own response minus the within-game-deflated realized response, about +0.21
for any smooth predictor.

Two interventions were run on its strength. Both are null on every corrected
metric, and both are informative in retrospect:

- **PR #108** (closed unmerged): the score-diff scalar on an RBF basis
  (encoding v2), teacher retrained 195 generations. The trained model uses
  the basis heavily and nothing about the bias changes: representation was
  never the constraint.
- **The `sd-mean-mse` tag** (358 generations): Huber → MSE for the mean loss
  at matched gradient scale. Quality parity; the aggregate is
  indistinguishable at matched young ages. Correct in direction (it removes
  (a)), but worth about 1 point where the headline is a ±2.5 drift.

## Reproduction

Needs a built engine, the mount lexica, and `onnxruntime` (CPU is fine).

**The encoder must match the checkpoints.** These checkpoints were trained on
an 85-plane input row. The encoder has since gained two reachability planes
(`spatial_planes()` is 87, commit `e8b4e01`), so decoding rows for them needs
a checkout from before that commit. At the time, decoding also had a
contingent-map arm, which must be off (it has since been deleted); the
opponent-leave arm is on.

Generate games under the training configuration:

    ./target/engine/play_game --games=1600 --seed=7 --threads=24 \
      --face-up-leaves --random-opening-mean=2.0 --binary-log-dir=<dir> \
      --player "--type=hastybot-endgame" --player "--type=hastybot-endgame"

Decode post-move rows with `ffi.decode_rows(post_move=True)` over each game's
eligible turns. The raw score-diff scalar is
`rows[:, spatial_planes()*225 + 127] * 100`, which is `85*225 + 127` for these
checkpoints (27 rack counts and 100 unseen-pool floats precede it).

1. **Aggregates** (expected at ep4414, seed 7): dd ≈ +2.8, dv ≈ +0.008
   row-weighted; clustered by game, ≈ +2.5 / +0.007.
2. **Pair-sum coherence** (the assumption-free core): within each game, pair
   adjacent post-move rows (t, t+1). Realized pair-sums cancel to under 1e-3
   by construction; predicted pair-sums average +5.36 ± 0.12 dd /
   +0.0163 ± 0.0006 WLD at ep4414, with the near-tied |lead| bin about +2
   points above the mid-lead floor.
3. **S1, directly**: copy the rows, overwrite the score-diff scalar with
   (lead ± 10)/100, run both; slope = Δpred/20. Expect about 0.974
   everywhere. The cross-state response (predictions regressed on lead
   across rows) reads about 0.90; the gap is S1.
4. **Lead-conditional profiles**: bin rows by lead (edges ±300, ±80, ±50,
   ±30, ±15, 0) and report row-weighted mean and median dd/dv per bin. Do NOT
   use within-game slopes or clustering by game within a bin (traps 1 and 2).
5. **The drift series**: score any tag's `models/model_epoch_*.onnx` over one
   fixed game set. `face-up-official` reproduces the [−2.6, +2.5] excursion;
   consecutive WSD exports of `face-up-leaves-fixed2` reproduce the σ ≈ 1.6,
   lag-1-white jitter.

## Pointers

- [PR #107](https://github.com/eigen7/Scribblez/pull/107): the superseded
  diagnosis. Parts of its evidence chain remain valid and are kept there: the
  paired rollout study, the tempo accounting, endgame invariance.
- [PR #108](https://github.com/eigen7/Scribblez/pull/108) (closed): the
  encoding intervention and why it was closed.
- The four position_eval runs whose checkpoints carry all of the above:
  `face-up-official`, `face-up-leaves-fixed2` (the WSD control),
  `sd-mean-mse` and `score-diff-basis`. None of them is under
  `/workspace/mount/tags/position_eval/` any longer (checked September 2026),
  so reproducing the numbers exactly needs them restored from elsewhere.
- [position_eval/trainer.py](../../py/scribblez/position_eval/trainer.py):
  trains on post-move rows; where the gates would land.

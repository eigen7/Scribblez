# Teacher POV calibration bias — corrected diagnosis

The position evaluation model (teacher) systematically flatters the player
whose POV it evaluates: at `face-up-official` epoch 4414, the two seats' final-
margin predictions for the *same game* sum to **+5.36 ± 0.12 points** instead
of zero (30,510 POV-alternating adjacent-row pairs; the win-probability analog
sums to 1.0163 ± 0.0006). Per seat: ~+2.5 pts / +0.8% win-prob, confirmed on
the model's own training rows (dd +2.85 ± 0.15, dv +0.0094 ± 0.0008), so it is
in-distribution, not a measurement-set artifact.

**This document supersedes its earlier revision, whose central claim — a
"structural margin-expansion slope" of +0.21 pts per point of current lead —
was an estimator artifact** (see "What the earlier diagnosis got wrong"). Two
training runs were spent on interventions derived from that claim, both null.
The corrected diagnosis below was developed against an adversarial review
process (four rounds; every load-bearing number cross-checked by at least two
independent estimators) and decomposes the bias into four measured components.
The diagnosis is settled; **the fix is future work** (see "Future work").

## The diagnosis in one paragraph

The shipped +2.5 is essentially all **drift**: the POV-marginal is a nearly
flat direction of the training objective (removing the entire bias is worth
~1.5e-4 of the loss, 20x below generation-to-generation loss noise), so the
model's offset executes an unpinned slow walk, and the deployed value is a
stopping-time accident — the same run's exported bias oscillated between
−2.6 and +2.5 pts, and reads −0.6 at epoch 4000. Underneath the drift sit
three stable, reproducible components that *net slightly negative*: a Huber
median-vs-mean wedge (~−1 pt, actively preferred by the loss), a margin-
compression profile S1 (the model's total response to the current lead is
~0.90 vs reality's 0.97), and a trailing-tempo bump S2 (+4 pts: "just moved,
still slightly behind" is overvalued). The schedule-free deployment then
shapes what ships: its unbounded 1/k weight average filters fast jitter but
lags and smooths the slow walk into large excursions.

## Methodology traps (read before measuring anything here)

Three traps produced the earlier misdiagnosis; each is easy to re-fall into.

1. **Within-game regression of prediction error on the current lead is
   invalid.** Within one game the realized final takes two values (±D) while
   the current lead takes ~20, so game-mean-centering crushes
   d(realized)/d(cur) from 0.93 (pooled) to 0.69 while any smooth predictor
   keeps ~0.93. The estimator's own null is therefore ~+0.24: an OLS fit of
   realized-on-lead — zero lead-conditional bias *by construction* — scores
   +0.245 ± 0.038 on it, *higher* than any real teacher. The earlier
   "+0.21 slope, identical across checkpoints" was exactly this null.
   Use pooled estimators (bin by lead, compare within bins).
2. **Game-in-bin clustering changes per-bin profiles qualitatively.** The
   training estimand weights each (game, turn) by 1/n_g, which is within
   ≤0.3 pts/bin of plain row weighting (verified). Clustering by
   games-present-in-bin up-weights a bin's brief visitors ~20x and flips
   signs (e.g. cur ∈ [80,300): row-weighted −1.9 vs clustered +8.5 on the
   same data). Report per-bin profiles row-weighted.
3. **Huber(δ=10) on a ±60-pt target fits the conditional median.** 86% of
   residuals sit beyond the knee. The loss's own optimum has mean bias
   ≈ −1 pt (the outcome distribution's mean−median gap); a "biased" mean can
   be the loss behaving correctly. Report mean AND median calibration.

## The four components

All numbers: `face-up-official` unless stated; FP32; fresh self-play under the
training configuration (hastybot-endgame both seats, `random_opening_mean`
2.0, face-up leaves); "dd"/"dv" = predicted − realized for the score-diff /
win-equity heads from the POV seat at post-move rows.

**(a) Huber wedge, ~−1 pt.** The score-diff mean head's loss optimum sits at
mean bias −1.05 (offset sweep: Huber's minimum is −3.5 pts from the shipped
point, whose residual mean is +2.45 vs median +3.54). The WSD control run's
long-run center, dd ≈ −1.5 / dv ≈ −0.0089 over 15 checkpoints, matches this
equilibrium propagated through the measured inter-head coupling (~155 pts per
unit win-prob). Loss-preferred: only a mean-consistent loss removes it.

**(b) Stable conditional structure: S1 + S2** (replicated row-weighted across
two optimizers, two loss shapes, checkpoints from 3 independent runs, and 3
disjoint game sets):

- **S1 — margin compression.** Total cross-state response of the predicted
  final to the current lead: 0.899 ± 0.007, vs realized 0.970; the per-bin
  odd component reads −0.05..−0.075 pts/pt beyond |lead| ≈ 30 — two
  independent estimators, one number. Direct input perturbation (±10 pts on
  the score-diff scalar at fixed states) shows the scalar channel itself
  responds at 0.974 ± 0.001: **the compression is not concentrated in the
  scalar input channel**; the deficit lives in the net lead-correlated
  contribution of the non-scalar features (−0.075/pt), whose internal
  decomposition is unidentified. No input-encoding change can address it.
- **S2 — trailing-tempo bump.** At post-move states with the mover still
  slightly behind (cur ∈ [−15,0)), realized E[final] ≈ −29 while the model
  says ≈ −22: having just banked a move is overvalued precisely when it
  failed to take the lead (+4 pts stable across checkpoints; ~+6–7 at ep4414
  including its drift state). Confirmed outcome-noise-free by conditional
  pair-sums: the near-tied |lead| bin's pair-sum exceeds the mid-lead floor
  by +2.0 pts dd / +0.023 dv at >4σ. The leading-side twin ([0,15)) is
  approximately calibrated.

Weighted by the row distribution (mean POV lead +19.5, against which S1 acts),
(a)+(b) **net ≈ −0.5..−1.5 pts** — the stable structure does not produce the
shipped positive bias; the drift does.

**(c) Unpinned drift — the dominant term in what ships.** Flatness, measured:
removing the entire dv bias costs 1.4e-4 nats of CE (generation-to-generation
CE stdev: 2.8e-3); the dd analog costs 1.6e-4 of weighted loss. Consequences,
all measured:
- The deployed export series (100-gen spacing, all 4414 generations) wanders
  in [−2.6, +2.5] with a ~700–1500-gen excursion: quiet band ±0.4 to ep1500,
  dive to −2.6 (ep1900), climb to +2.4 (ep3300), back to −0.6 (ep4000–4100),
  +2.5 at 4414. **The production number is where the run stopped.** Onset
  ep1500–1600 precedes the gen-1773 trainer restart; lr, rows/gen, loss
  weights, and the (frozen-HastyBot) data distribution were audited flat.
  At onset the excursion is score-diff-head-specific — dv holds at −0.001
  while dd dives — i.e. it lives in the *least-pinned* head (λ_sd = 2e-4).
- The raw iterate's checkpoint-to-checkpoint jitter is σ ≈ 1.6 pts and
  **white at lag 1 generation** (lag-k RMS increments flat from k=1 to 30;
  40 consecutive WSD exports, paired rows). A WSD export is a per-checkpoint
  lottery draw; its center is the (a)+(b) equilibrium, not zero.
- The drift is low-dimensional but not a constant: checkpoint differences
  carry an offset AND a response-slope tilt (ep4414 − ep1500 rises
  monotonically +0.9 → +6.0 across lead bins, paired SE ±0.2).
- Guard-rail: the stopping-time framing applies to the *aggregate only*. The
  conditional structure (b) is stable across checkpoints and larger than any
  aggregate — "the +2.5 was luck" must not be read as "nothing is wrong".

**(d) The 1/k averaging horizon (schedule-free deployments).** The deployed
model is a uniform average of the whole run's iterates. It suppresses (c)'s
fast jitter (the early deployed series is ±0.4 where raw WSD exports swing
±1.6) but lags and smooths the slow walk into the big excursions, and the
live iterate's weight-space distance from the deployed average grows
monotonically (~1500 → ~2800 over the run) while deployed quality stalls:
eval_win_mae bottoms at 0.0167 (ep3724), is flat from ~ep1000, and degrades
to ~0.020 over the final stretch, co-timed with the bias excursion.
eval_sd_mean_mae is elevated at every excursion peak — it already partially
monitors (c) and belongs in the promotion gate now.

## Ruled out (each by direct measurement)

- **Score-diff input resolution** (the previous diagnosis's prescription) —
  three ways: a full retrain on a 15-bump RBF basis encoding left every
  metric unchanged (PR #108, closed); the trained basis model demonstrably
  *uses* the basis (zeroing it moves win-prob by 0.10) yet shows the same
  bias; and the direct perturbation test shows the scalar channel's response
  was never the deficit.
- **Mean-head loss shape as the driver of the shipped number** — Huber vs
  MSE at matched ages ep100/200/339: differences ≤ 0.2 pts (the wedge (a) is
  an equilibrium shift of ~−1, invisible under young-age drift; the
  `sd-mean-mse` run also ran λ_sd 11x smaller, weakening restoring force).
- **Generation-config mismatch** (greedy vs solver endgames, random
  openings): moves realized POV outcomes by 0.03 pts and dd by <0.1 — and
  the bias reproduces on the model's own training rows regardless.
- **The trainer's sampler**: trainer-drawn WLD targets average 0.4990 ±
  0.0018 and score-diff targets ≈ 0. No POV asymmetry in what is trained on.
- **BatchNorm recalibration at export**: +0.00014 wp.
- **Trainer restarts**: averaging weight continuous through all three;
  schedule-free state round-trips; onset precedes the restart.
- **Nonstationary data**: the generator is frozen HastyBot; distribution
  stationary by construction and audited.

## Impact while unfixed

Unchanged from the earlier revision where it was right: the bias is
common-mode within a position (measured candidate-relative margin ≈ 0), so
**rankings, CRN-paired gains, and the sim loop are safe**. Absolute readouts
are off by the current drift state (±2.5 pts / ±0.8% wp, sign unknowable in
advance), which blocks the planned sim-value second target stream and any
cross-POV absolute comparison. New: (d) means long schedule-free runs are
also quietly paying a deployed-quality cost after ~ep1000.

## Future work (TODO — diagnosis settled, fix not designed)

The one-line requirement: **add restoring force**. Optimizer and loss choices
relocate or reshape the bias distribution (WSD+MSE would center it near
−0.5 ± jitter); only a term the objective actually feels can pin it.

Candidates, in intended order:
1. **F2' — MSE (or other mean-consistent) score-diff mean loss at restored
   effective weight** (MSE is ~11x Huber's gradient scale on these residuals;
   `sd-mean-mse` used λ 1.77e-5 to match scales). Removes (a). Validated for
   quality-parity to gen 358 only.
2. **F1'' — conditioned marginal-calibration term**: match predicted vs
   realized outcome marginals (all three WLD classes + sd mean) within bins
   of lead x turn or of predicted value — a *global* marginal term cannot see
   S1 or the drift's tilt. Weight is a real design problem (target: restoring
   force ~1–10% of the CE gradient scale, tuned against the per-generation
   coherence metric, gated on the EXPORTED model — pinning the live iterate
   does not provably pin the average).
3. **F3 — bounded averaging horizon.** Measured design input: the jitter is
   white at lag 1, so an average of ~4 *consecutive* exports already achieves
   the full √4 (σ 1.61 → 0.62, at the slow-center floor of ±0.6); stride
   buys nothing. This also caps (d)'s lag/stall.
4. **Gates** (start now, before any fix): per-generation conditional
   pair-sum coherence on fresh self-play — SE 0.12 pts per 1600 games,
   outcome-noise-free — plus eval_sd_mean_mae, plus row-weighted per-bin
   profiles (mean AND median). The pair-sum metric cancels odd-in-lead
   structure by construction (it cannot see S1 or the tilt), so the
   coherence and profile gates are complementary, not redundant.

Validation ladder (cheapest first): post-hoc per-bin recalibration is
already demonstrated (fitted on 1600 fresh games, it removes the aggregates
held-out on two independent sets — dd +2.51 → −0.29 / +2.73 → −0.06,
dv → ~0 — at zero CE cost); next a head-only fine-tune swapping Huber → MSE
(expected: ~1 pt, the wedge); then adding F1'' terms (tests S1/S2
removability in-training); a full run only after those.

Open items, honestly: the ~1000-gen timescale of (c)'s excursions is
characterized, not explained (per-generation coherence logging on the next
run measures its spectrum for free); the true scalar-channel partial
(≈1.0 vs measured 0.974) is unmeasured and would only reshuffle S1's internal
attribution; whether F1'' pins the *export* rests on an empirical gate.

## What the earlier diagnosis got wrong (and the null interventions)

The previous revision decomposed the bias into a "structural slope" (+0.21
pts over-prediction per point of lead, stable across checkpoints) plus a
drifting offset, and prescribed a nonlinear score-diff input basis. The slope
was trap #1 above: its stability across lineages and ages was the signature
of an estimator constant, not a model property (its value equals each model's
own response minus the within-game-deflated realized response, ~+0.21 for any
smooth predictor). Interventions run on its strength, both null on every
corrected metric and both informative in retrospect:
- **PR #108** (closed unmerged): kScoreDiff RBF basis, encoding v2, teacher
  retrained 195 gens. The basis is used heavily by the trained model and
  changes nothing about the bias — representation was never binding.
- **`sd-mean-mse` tag** (358 gens): Huber → MSE mean loss at matched gradient
  scale. Quality parity; aggregate indistinguishable at matched young ages —
  correct in direction (it removes (a)) but ~1 pt where the headline is a
  ±2.5 drift.

## Reproduction

Needs a built engine, mount lexica, `onnxruntime` (CPU fine). Generate games
under the TRAINING configuration (the earlier revision's recipe used plain
`--type=hastybot` and no random openings; the difference is negligible but
there is no reason to keep it):

    ./target/engine/play_game --games=1600 --seed=7 --threads=24 \
      --face-up-leaves --random-opening-mean=2.0 --binary-log-dir=<dir> \
      --player "--type=hastybot-endgame" --player "--type=hastybot-endgame"

Decode post-move rows exactly as before (`decode_rows(post_move=True)` over
each game's eligible turns; teacher arm: contingent off, opp-leave on; the
raw score-diff scalar is `rows[:, 85*225 + 127] * 100`).

1. **Aggregates** (expected, ep4414, seed 7): dd ≈ +2.8, dv ≈ +0.008
   (row-weighted; game-clustered ≈ +2.5/+0.007).
2. **Pair-sum coherence** (the assumption-free core): for each game, pair
   adjacent post-move rows (t, t+1); realized pair-sums cancel to <1e-3 by
   construction; predicted pair-sums average +5.36 ± 0.12 dd /
   +0.0163 ± 0.0006 wld at ep4414, with the near-tied |lead| bin ~+2 pts
   above the mid-lead floor.
3. **S1, directly**: copy rows, overwrite the score-diff scalar with
   (cur ± 10)/100, run both, slope = Δpred/20. Expected ≈ 0.974 everywhere
   (the *cross-state* response, regressing predictions on cur across rows,
   reads ≈ 0.90 — the gap is S1).
4. **Lead-conditional profiles**: bin rows by cur (edges ±300, ±80, ±50,
   ±30, ±15, 0), report row-weighted mean and median dd/dv per bin. Do NOT
   use within-game slopes or game-in-bin clustering (traps #1, #2).
5. **The drift series**: any tag's `models/model_epoch_*.onnx` scored over
   one fixed game set. `face-up-official` reproduces the [−2.6, +2.5]
   excursion; consecutive WSD exports (`face-up-leaves-fixed2`) reproduce
   the σ ≈ 1.6, lag-1-white jitter.

## Pointers

- [PR #107](https://github.com/eigen7/Scribblez/pull/107) — the superseded
  diagnosis this document corrects (kept for the evidence chain that remains
  valid: the paired rollout study, tempo accounting, endgame invariance).
- [PR #108](https://github.com/eigen7/Scribblez/pull/108) (closed) — the
  encoding intervention and its closure rationale.
- `face-up-official`, `face-up-leaves-fixed2` (WSD control), `sd-mean-mse`,
  `score-diff-basis` under /workspace/mount/tags/position_eval/ — the four
  runs whose checkpoints carry all of the above.
- [position_eval/trainer.py](../py/scribblez/position_eval/trainer.py) —
  post_move=True training; where the future gates land.
- [generational_teacher.md](generational_teacher.md) — where the coherence
  gate becomes a promotion gate.

# Teacher POV calibration bias — diagnosis and fix plan

The position evaluation model (teacher) systematically flatters the player
whose POV it evaluates: on fresh HastyBot self-play, `face-up-official`
epoch 4414 reads post-move states **+0.8% win-prob / +2.6–2.8 points**
better for the POV player than the realized outcomes. The bias decomposes
into a **structural margin-expansion slope** (+0.21 pts of over-prediction
per point of current score lead — identical across checkpoints and training
ages) and a **training-drifting constant offset**. The slope component is
the one the encoder's `ANALYSIS TODO` predicted: the single score-diff
scalar is too smooth a representation for the correction from current lead
to final outcome. That TODO is discharged — the basis it prescribed is the
encoding now, in
[score_diff_features.h](../engine/include/encoding/score_diff_features.h)
(Phase 2 below) — but every number here comes from a teacher trained before
it, and only a retrain on the new encoding can move them.

This document is self-contained: the full evidence chain, what is ruled
out, reproduction recipes with expected numbers, and a phased fix plan with
acceptance criteria. It exists so a fresh session can pick up the fix
without re-deriving the diagnosis.

## How it was found

Validating value-truncated rollouts (roadmap item 2,
[PR #106](https://github.com/eigen7/Scribblez/pull/106)): truncated and
terminal rollouts share seeds under CRN, so each truncated rollout's leaf
readout is directly paired with the realized terminal continuation of the
same trajectory. The paired comparison exposed a small aggregate bias whose
sign flips with horizon parity — which pinned it on the leaf model's POV
rather than on the truncation machinery. Everything below chases that
observation into the teacher itself; **none of it is a rollout defect**,
and PR #106 is not gated on it.

## The evidence chain

All errors are game-clustered standard errors (predictions within a game
share the game's outcome; clustering by game is mandatory). "dv" is
predicted-minus-realized win-prob (P(win) + P(draw)/2) from the POV player;
"dd" the same for the final score differential in points. Teacher:
`face-up-official/models/model_epoch_4414.onnx` unless stated, evaluated in
FP32 (FP16 has a separate NaN-overflow issue — see PR #106).

**1. Paired rollout study** (100 games, 400 pre-endgame decision points,
top-5 equity candidates, 200 CRN-paired rollouts; truth = terminal HastyBot
continuation):

| horizon (leaf POV)  | dv               | dd               |
|---------------------|------------------|------------------|
| 4 (root mover)      | +0.0058 ± 0.0007 | +2.66 ± 0.15 pts |
| 5 (opponent)        | −0.0043 ± 0.0006 | −2.14 ± 0.15 pts |
| 8 (root mover)      | +0.0047 ± 0.0005 | +2.30 ± 0.14 pts |

The sign follows whose POV the leaf is read from, and the magnitude is
depth-stable. Per-position spread is modest (std 0.013 win-prob; 0.2% of
positions beyond |0.05|), and late positions show exactly zero (their
rollouts end before the horizon and contribute exact terminal results).
Candidate-relative bias — the (equity-rank-0 minus rest) margin — is
**+0.0004 ± 0.0013, i.e. zero**: within a position the bias is common-mode.

**2. On-distribution reproduction, no rollouts involved** (the same 100
games' 2,031 eligible post-move rows — the training sample kind — decoded
through the training path (`decode_rows(post_move=True)`) and scored
against the games' real outcomes):

    dv +0.0083 ± 0.0019      dd +2.84 ± 0.37 pts

By-phase (root turn): dd grows +0.85 → +2.25 → +4.80 pts across turn
buckets [0,6) / [6,12) / [12,30); dv is flat.

Replication on 400 fresh games (seed 12; 8,052 rows): dv +0.0080 ± 0.0009,
dd +2.59 ± 0.18.

Statistical note: this estimator is far more precise than binary outcomes
suggest because consecutive rows alternate POV, and the game outcome
cancels exactly in POV-alternating pairs. Equivalently, the aggregate dv
measures a pure **coherence violation**: the model's win-probs for the two
seats, read from their alternating post-move states, sum to ~**1.017**
instead of 1. (Conditioned splits — by predicted leader, by value bucket —
forfeit this cancellation and need ~10× the rows for the same power.)

**3. Invariant to the endgame value definition**: regenerating the truth
games with `--type=hastybot-endgame` (solver endgames, as training
self-play uses) leaves the numbers unchanged (dd +2.82 vs +2.84). The
solver-vs-greedy endgame mismatch contributes nothing measurable.

**4. Tempo is priced, but incompletely**: at post-move states the POV
player is ahead by **+19.7 pts on average** (they banked a move the
opponent has not answered — pure tempo). Realized outcomes give back
−19.98 ± 0.36 of it by game end; the model gives back −17.15 ± 0.19 —
**86%**. The bias is the un-returned remainder, so the model clearly
"knows" the opponent moves next (a model blind to that would err by
~20 pts, not ~3).

**5. The decomposition — slope vs offset** (400-game set, within-game
regression of dd on the current-diff input, plus aggregate):

| checkpoint            | aggregate dd     | slope dd~cur (pts/pt) |
|-----------------------|------------------|-----------------------|
| fixed5 ep0047         | −0.43 ± 0.18     | **+0.223 ± 0.036**    |
| official ep3000       | +1.32 ± 0.18     | **+0.221 ± 0.036**    |
| official ep4414       | +2.59 ± 0.18     | **+0.210 ± 0.036**    |

The **slope is identical across lineages and training ages** — structural.
Under a linear model dd ≈ a + b·cur with mean cur = +20.3, the implied
offsets are a ≈ −4.9 (fixed5) → −3.2 (ep3000) → −1.7 (ep4414): the
**offset drifts upward with training** while the slope stays put. fixed5's
"zero" aggregate is an accidental cancellation of the two components at
its training age, not health. The win-prob analog of the slope (ep4414):
+0.00096 ± 0.00029 per point.

## Mechanism

**Component 1 — structural under-correction of the score-diff input
(the dominant, stable term).** The model's effective mapping from current
differential to predicted final outcome sits ~0.21 pts/pt too close to the
identity: it under-regresses current leads toward realized finals. This
surfaces as *POV* bias only because post-move POV states carry the +20-pt
tempo asymmetry; it is really a *lead-conditional* miscalibration
(slope × mean POV lead ≈ the aggregate). It is exactly the failure mode the encoder's
`ANALYSIS TODO` anticipated for the single smooth scalar, and the phase
gradient (worst late, where the win/diff structure over score-diff is
sharpest) matches that TODO's predicted failure zone. Phase 2 below is that
TODO's prescription, now implemented.

Note there is no explicit side-to-move input, and none is needed: the
teacher trains **exclusively on post-move rows**
([position_eval/trainer.py](../py/scribblez/position_eval/trainer.py)
loads `post_move=True`), and `PositionEncoder::replay_to_sampled`
([position_encoder.cpp](../engine/src/encoding/position_encoder.cpp))
applies the sampled turn's move whatever its type — so "the opponent moves
next" is a constant of the training distribution, carried by convention.
The score-diff scalar is nonetheless the *carrier of the POV asymmetry*,
which is why its under-resolution lands as POV bias.

**Component 2 — a training-drifting constant offset** (−4.9 → −1.7 pts
over ~4,400 epochs in the official run; the wld-head aggregate moved
+0.0100 → +0.0083 over ep3000 → ep4414, so the drift statement is cleanest
in the score-diff head). Unattributed. Suspects, in rough order:

- **Data-reuse sharpening**: long training under the reuse regime
  gradually over-weights the dominant, easy feature (current diff) relative
  to its correction — the same family as the July reuse/memorization
  regression that led to the reuse-4 regime (PR #46).
- **Schedule-free eval-regime gap**: the exported model is the averaged
  eval-mode point with separately recalibrated BN statistics (PR #83), not
  the point the training loss is minimized at. An offset introduced between
  those two points is invisible to the loss and free to drift. (This
  mechanism would also explain why an easily-learnable constant is not
  simply trained away.)

## Ruled out

- **Truncation machinery / POV flip logic** — the bias reproduces on plain
  training-kind rows with no rollouts (evidence 2).
- **Rollout-horizon distribution shift** — same reason.
- **Solver-vs-greedy endgame value definitions** — evidence 3.
- **Tempo/side-to-move blindness in the strong sense** — evidence 4
  (86% of tempo returned; a blind model would err ~7× larger).
- **A gross POV/encoding bug** — magnitudes are ~0.8%, not ~±40%; the
  serving encodes are pinned byte-identical to training replays by tests.
- **Structural-only or drift-only stories** — evidence 5 requires both
  components.

## Impact while unfixed

- **The sim loop and truncated corpora are safe**: the bias is common-mode
  within a position (measured margin bias ≈ 0), so rankings and CRN-paired
  proves-best gains cancel it. One horizon per corpus keeps the POV parity
  fixed, which the `.sobs` v3 header already enforces.
- **Absolute readouts are flattered by ~0.8% wp / ~2.8 pts toward the POV
  player**: eval dashboards, cross-POV comparisons, and any future
  absolute-value consumer — notably the planned **sim-value second target
  stream for the teacher** (roadmap "models" section), which should not be
  wired up until this is fixed or calibrated, on top of its existing
  terminal-configuration constraint.
- Under the **spread objective**, the slope could in principle distort
  within-position comparisons (higher-scoring candidates reach horizons
  with higher cur); measured effect is bounded negligible
  (+0.0004 ± 0.0013 wp), but recheck if spread ever becomes the default
  sim objective.

## Fix plan

Metric first, then structure, then drift. The match harness arbitrates the
stack end to end, as always.

**Phase 1 — make it observable (trainer health metrics).** Add to the
position_eval trainer's eval pass (and the dashboard), computed on a small
held-out set of *fresh* self-play games each eval:

- `pov_coherence`: mean over POV-alternating adjacent post-move row pairs
  of (v_A + v_B) − 1, and the analog pd_A + pd_B for the diff head. Zero
  for a coherent model, outcome-noise-free by construction, a few hundred
  rows suffice.
- `lead_slope`: the within-game regression slope of (pred − realized
  final) on the current-diff input, per the reproduction script below.

These make both components visible per-epoch (when does the offset drift
start?) and gate teacher promotion under
[generational_teacher.md](generational_teacher.md).

**Phase 2 — structural fix: richer score-diff featurization** — **the
encoding change is LANDED; the retrain is not.** The differential's
representation now lives in one component,
[score_diff_features.h](../engine/include/encoding/score_diff_features.h):
the raw normalized scalar (unchanged, still first in its block, so anything
reading the differential back off a row still works) followed by 15 Gaussian
bumps spaced uniformly over the squashed coordinate
`u(d) = d / (|d| + 30)`. Uniform in `u` is dense in points near a close game
and sparse out in the decided tail — centers near 0, ±5, ±12, ±23, ±40, ±75,
±180 — which is the resolution profile the differential actually needs, and
adjacent overlapping bumps let a linear readout build a steep ramp anywhere
in range. Both consumers moved together:
[input_encoder.h](../engine/include/encoding/input_encoder.h)'s `kScoreDiff`
block (`kInputEncodingVersion` → 2) and the move set model's resultant-diff
move feature in
[move_set_encoder.h](../engine/include/training/move_set_encoder.h)
(`kMoveScalars` 3 → 18, `kMoveEncodingVersion` → 2), whose basis sits at the
scalar tail so the three named scalars keep their indices. The in-place
rewrite paths (`overwrite_score_diff`, `encode_input_with_score_diff`, and
the FFI score-diff sweep) rewrite the whole block, which a test pins — a
rewrite that touched only the raw scalar would leave the basis describing
the position's original differential.

What remains: **retrain the teacher on the new encoding**, then regenerate
downstream (student, corpora) on the normal generational cadence — the
encoding-version gates sequence this safely, and every existing checkpoint
is now rejected loudly by them. Nothing about the bias is fixed until that
retrain happens; the change only makes the structure expressible.

Success = `lead_slope` ~0 on the new teacher with no eval_win_mae or
match-strength regression. (Phase 1's metrics are not implemented, so for
now that is measured with the reproduction script below.)

**Phase 3 — attribute and fix the offset drift.**

- **Schedule-free export gap test** (cheap, decisive): score the same
  held-out rows with the trainer's train-mode weights and with the
  exported eval-mode weights; the difference in `pov_coherence` between
  them isolates the export-gap contribution directly.
- **Reuse ablation**: the Phase-1 metrics per-epoch over an existing run's
  checkpoint history (the reproduction script works on any exported
  epoch) show whether the drift tracks reuse-heavy stretches.
- If neither fully closes it, a last-resort mitigation is an export-time
  per-head affine recalibration fit on fresh self-play — kills the offset,
  adds a moving part, does not touch the slope; the structural fix is
  preferred.

**Acceptance criteria** (on ≥400 fresh self-play games, both with and
without random openings, FP32):

- |aggregate dv| < 0.002 and |aggregate dd| < 0.5 pts;
- |`lead_slope`| < 0.05 pts/pt (diff head) and < 0.0002 /pt (wld head);
- `pov_coherence` within ±0.004 of zero;
- no regression in eval_win_mae or the match harness.

## Reproduction

Needs a built engine, the mount lexica, and `onnxruntime` (CPU is fine —
~8k rows in a few minutes). The teacher used here:
`/workspace/mount/tags/position_eval/face-up-official/models/model_epoch_4414.onnx`
(85 spatial planes / 163 scalars: no contingent features, opp-leave input —
adjust `set_*` calls and the 85-plane constant if probing another arm).
Fresh checkpoints will differ in detail; the slope has been stable.

Generate games (~1s):

    ./target/engine/play_game --games=400 --seed=12 --threads=24 \
      --face-up-leaves --binary-log-dir=<dir> \
      --player "--type=hastybot" --player "--type=hastybot"

Measure (aggregate, phase slices, slope):

    PYTHONPATH=py python3 - <<'EOF'
    import glob
    import numpy as np
    import onnxruntime as ort
    from scribblez.ffi import (set_contingent_features, set_opp_leave_input,
                               decode_rows, input_floats)
    from scribblez.sim_evidence import slog_meta

    MODEL = "/workspace/mount/tags/position_eval/face-up-official/models/model_epoch_4414.onnx"
    GEN_DIR = "<dir>"

    set_contingent_features(False)   # must match the model's declared arm
    set_opp_leave_input(True)
    inputs_l, out_l, tgt_l, cur_l, game_l, turn_l = [], [], [], [], [], []
    gbase = 0
    for slog in sorted(glob.glob(f"{GEN_DIR}/*.slog")):
        buf = slog_meta.read_slog_bytes(slog)
        metas = slog_meta.game_metas(buf)
        games, turns = [], []
        for g, m in enumerate(metas):
            for t in range(int(m["eligible_begin"]), int(m["eligible_end"])):
                games.append(g); turns.append(t)
        games, turns = np.array(games), np.array(turns)
        rows = decode_rows(slog, games, turns, post_move=True)
        nf = input_floats()
        inputs_l.append(rows[:, :nf])
        out_l.append(rows[:, nf] + 0.5 * rows[:, nf + 1])  # realized W + D/2
        tgt_l.append(rows[:, nf + 3])                      # realized final diff
        cur_l.append(rows[:, 85 * 225 + 127] * 100.0)      # kScoreDiff, unscaled
        game_l.append(games + gbase); turn_l.append(turns)
        gbase += len(metas)
    inputs = np.concatenate(inputs_l)
    outcome_v, target_diff = np.concatenate(out_l), np.concatenate(tgt_l)
    cur, games, turns = np.concatenate(cur_l), np.concatenate(game_l), np.concatenate(turn_l)

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    sp = inputs[:, :85 * 225].reshape(-1, 85, 15, 15)
    sc = inputs[:, 85 * 225:]
    pv, pd = [], []
    for i in range(0, len(sp), 256):
        o = sess.run(["wld", "score_diff"],
                     {"input_spatial": sp[i:i + 256], "input_scalar": sc[i:i + 256]})
        w = np.exp(o[0] - o[0].max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        pv.append(w[:, 0] + 0.5 * w[:, 1]); pd.append(o[1][:, 0])
    pv, pd = np.concatenate(pv), np.concatenate(pd)
    dv, dd = pv - outcome_v, pd - target_diff

    def cluster(vals, mask):
        gg = np.unique(games[mask])
        gm = np.array([vals[mask & (games == g)].mean() for g in gg])
        return gm.mean(), gm.std(ddof=1) / np.sqrt(len(gm))

    every = np.ones(len(dv), bool)
    print("aggregate dv %+.4f +/- %.4f   dd %+.3f +/- %.3f"
          % (*cluster(dv, every), *cluster(dd, every)))
    for lo, hi in [(0, 6), (6, 12), (12, 30)]:
        m = (turns >= lo) & (turns < hi)
        print("  turns [%2d,%2d): dv %+.4f +/- %.4f   dd %+.2f +/- %.2f"
              % (lo, hi, *cluster(dv, m), *cluster(dd, m)))
    slopes = []
    for g in np.unique(games):
        m = games == g
        x = cur[m] - cur[m].mean()
        if m.sum() < 6 or (x ** 2).sum() < 1e-6:
            continue
        slopes.append((x * (dd[m] - dd[m].mean())).sum() / (x ** 2).sum())
    slopes = np.array(slopes)
    print("lead_slope (dd~cur) %+.4f +/- %.4f pts/pt"
          % (slopes.mean(), slopes.std(ddof=1) / np.sqrt(len(slopes))))
    EOF

The script's `85 * 225 + 127` offset is the raw score-diff scalar, which
Phase 2 left in place as the first float of its block, so this reads the same
on either encoding — but ep4414 itself is a v1 checkpoint, so measuring it
needs an engine from before the version bump.

Expected on ep4414 (400 games, seed 12): aggregate dv ≈ +0.0080 ± 0.0009,
dd ≈ +2.59 ± 0.18, phase dd ≈ +0.9 / +2.3 / +4.8, lead_slope ≈
+0.21 ± 0.04. On fixed5 ep0047: aggregate ≈ 0, same slope — that pair of
facts is the decomposition.

The paired truncated-vs-terminal study (evidence 1) additionally needs
PR #106's `--horizon`/`--leaf-model` flags on `sim_obs_tool`; the full
methodology and numbers are archived in PR #106's comment thread.

Caveats: the measurement games carry no random openings (training
generation uses `random_opening_mean` 2.0) — close to, but not exactly,
the training distribution; truth was verified invariant to greedy-vs-
solver endgame play; evaluate in FP32 (the FP16 NaN overflow on
extreme-advantage states is a separate finding, handled in PR #106 by the
FP32 leaf default).

## Pointers

- [PR #106](https://github.com/eigen7/Scribblez/pull/106) — value-truncated
  rollouts, plus the bias-validation comment thread this doc consolidates.
- [score_diff_features.h](../engine/include/encoding/score_diff_features.h) —
  the differential's representation, which Phase 2 replaced the bare scalar
  with; it carries the rationale the encoder's `ANALYSIS TODO` used to.
- [position_eval/trainer.py](../py/scribblez/position_eval/trainer.py) —
  `post_move=True`: the teacher's post-move-only training regime.
- [generational_teacher.md](generational_teacher.md) — where the Phase-1
  metrics become promotion gates.
- PR #83 (BN recalibration in schedule-free eval mode) and PR #46 (the
  reuse-4 regime) — background for the Phase-3 drift suspects.

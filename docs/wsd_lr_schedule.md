# The learning-rate schedule: cyclic warmup-stable-decay

The `wsd` optimizer arm: AdamW with a cyclic warmup-stable-decay (WSD)
learning rate over the rows-clock. It is the only schedule of the
max_move_per_lane and evidence_trajectories trainers, and one of two arms of
position_eval and move_set_eval, which default to the other arm,
`schedule_free` (AdamWScheduleFree; see `py/scribblez/generational/optim.py`).

The code is `WsdSchedule` and `WsdLrController` in
`py/scribblez/generational/controls.py`. This document records why the
schedule is shaped the way it is, how its defaults were sized, and the design
review behind it. Read it before changing the schedule.

## Why a cyclic schedule

The generational trainers are open-ended: they checkpoint every generation
indefinitely and can be paused and resumed at any point. An annealing
schedule normally assumes a known training horizon, so there is never an
obvious moment to decay. Without a schedule, an operator has to step the rate
down by hand off the loss plots. The unscheduled alternative, a flat rate,
shows its cost in [move_set_eval_results.md](move_set_eval_results.md): a run
trained flat at 1e-3 for 25 passes plateaued, with no way to tell whether a
decay would have helped.

Plain warmup plus cosine-to-zero needs the horizon. WSD solves half the
problem: its stable phase is a flat rate, and its decay is a short tail that
can start without knowing the total run length. The standard formulation
still decays once, at the end of a run, which still requires someone to say
"decay now".

To remove that judgment, the schedule **cycles**: warm up once, then repeat
stable → decay → warm restart. Each cycle leaves a well-annealed checkpoint,
and the sliding data window keeps getting fresh high-rate passes.
Structurally this is SGDR-style warm restarts with WSD's tail. It is this
project's own adaptation to the continual self-play setting, not a published
recipe.

## The schedule

`WsdSchedule(lr, warmup_rows, cycle_rows)` is a pure function of the
rows-clock. With `W = warmup_rows`, `C = cycle_rows`, `R = W // 4`,
`D = LR_DECAY_FRAC` (0.2), `F = LR_FLOOR_FRAC` (0.1), and
`t = (rows - W) mod C`:

| Segment | Rows | Value |
|---|---|---|
| Warmup (once) | `rows < W` | linear `0 → lr` |
| Cycle `k ≥ 0`: | | |
| &nbsp;&nbsp;re-warmup (skipped for `k = 0`) | `t < R` | linear `lr·F → lr` |
| &nbsp;&nbsp;stable | `R ≤ t < (1-D)·C` | `lr` |
| &nbsp;&nbsp;decay | `(1-D)·C ≤ t < C` | cosine `lr → lr·F` |

Shape decisions:

- **Cosine decay**, not linear and not the WSD paper's 1-√. Cosine's slope
  reaches zero at the end of the segment, which composes cleanly with the
  restart that follows; linear leaves a slope discontinuity at the jump. The
  1-√ shape is justified by single-decay analysis at LLM-pretraining scale,
  which does not transfer to small periodic cycles.
- **Floor at `0.1·lr`**, not near zero. A restart follows immediately, so
  decaying to near zero just before jumping back up wastes the end of the
  decay.
- **Re-warmup on restart, from the floor, inside the cycle.** By the end of a
  decay, AdamW's second-moment estimate has adapted to the low rate, so a
  bare jump to the peak risks an oversized effective step on the first
  batches after the restart. The ramp length is derived (`W // 4`) rather
  than a param, to keep the knob count down. It eats the front of the stable
  segment, so the period stays exactly `C`.
- **Decay and floor fractions are module constants**, not per-trainer params:
  no trainer has a reason to differ on the schedule's shape. Promote them to
  params if one does.
- **No validation of degenerate settings** (a re-warmup that swallows the
  stable segment). Every row count still maps to a rate, and a bad setting
  shows up on the loss plot like any other mis-set tunable.

## The controller

`WsdLrController(recorder, schedule, rows_trained)` is what a trainer holds.
It serves `schedule.value` as `run_epoch`'s per-batch `lr_fn` and keeps two
pieces of in-memory state. Both are re-derived from `rows_trained` on
construction; nothing is persisted, so a resume picks up mid-phase with no
extra checkpoint state.

- `.current` is the rate applied to the most recent batch. The trainers'
  end-of-generation log line and `metrics.lr` value read it after `run_epoch`,
  so they report the end-of-generation rate, consistent with the row count
  logged beside it. Reading a start-of-generation value there would shift
  every warmup and decay point on the "Learning rate" plot by one generation.
- The current phase, for boundary detection. A phase crossing is detected in
  the per-batch call and recorded as a control event named `lr` at that
  batch's exact row position. Because the phase is initialized from the
  resume cursor, restarting mid-phase records nothing spurious.

Events fire **only at phase boundaries** (end of warmup, each decay start,
each restart), never per generation. The continuous trajectory is already the
per-generation `metrics.lr` series and its log-scale "Learning rate" plot.
Control events exist to flag structurally notable moments on the loss-plot
overlay and in the Controls tab's history table; per-generation events during
a decay would be near-duplicate clutter.

## Per-trainer sizing

Each trainer's params carry `lr` (the peak), `lr_warmup_rows` and
`lr_cycle_rows`. The last two are in **that trainer's own rows-clock units**,
which differ. The defaults are starting points to retune from the loss plots.

- **position_eval and max_move_per_lane** count positions. With
  `games_per_generation=20000`, `window=4` and `turns_per_game=1`, a
  generation is ~80k rows. `lr_warmup_rows = 200_000` (~2.5 generations);
  `lr_cycle_rows = 2_000_000` (~25 generations per cycle, which gives several
  annealed checkpoints over a long run without squeezing the stable phase
  where most training happens).
- **move_set_eval** counts candidate moves. The reference run in
  [move_set_eval_results.md](move_set_eval_results.md) trained 772M rows over
  25 passes, ~31M rows per pass: two orders of magnitude more than the other
  trainers. Their defaults would end warmup within the first 1% of pass 1 and
  thrash through ~15 cycles per pass. `lr_warmup_rows = 15_000_000` (~0.5
  pass); `lr_cycle_rows = 300_000_000` (~10 passes, so ~2.5 cycles over a
  25-pass run, with decays landing around passes 12–13 and 22–23). These are
  sized from that one run; re-check against a live tag's rows per pass.
- **evidence_trajectories** counts held-out candidate rows.
  `lr_warmup_rows = 800_000` (~half a pass over a 350k-position corpus);
  `lr_cycle_rows = 16_000_000` (~10 passes).

On position_eval and move_set_eval, `lr_warmup_rows` also sets the
schedule-free arm's warmup, and `lr` left at 0 takes each arm's own default
(`generational/optimizer_arms.py`).

## Open questions

- **move_set_eval has a knowable horizon** once its corpus is complete
  (`target_pairs` plus `train_epochs`), which is the classic single-decay WSD
  case. A horizon-aware policy (decay over the last k epochs on the finished
  corpus) would fit it better than periodic cycles. Its default arm is
  schedule-free, so this matters only for `wsd` runs.
- Whether periodic restarts beat a monotone horizon-free schedule (warmup
  plus a floored inverse-sqrt decay, no cycles) on the open-ended trainers is
  unmeasured.

## Design review record (2026-08-17)

The design was reviewed as a plan before implementation by a four-seat panel:
hidden complexity (Claude), rival designer (Codex, with full repo access),
scope (Sonnet), and integration (Sonnet). The table keeps every serious
critique and its resolution, because the rejected ones are decisions someone
may want to revisit.

| # | Panelist | Critique | Resolution |
|---|---|---|---|
| 1 | hidden complexity, integration (independently) | A `.current` computed at epoch start lags a generation on the LR plot; phase-crossing detection needs remembered state and a defined emission point; resume needs a rule against spurious events. | Adopted: per-batch `.current`, in-loop crossing detection at the exact row position, phase seeded from the resume cursor. |
| 2 | scope, hidden complexity, rival (independently) | `move_set_eval` copied the other trainers' defaults despite a ~31M-rows-per-pass clock. | Adopted: per-workload defaults sized from the results doc, with units in the help text. |
| 3 | scope | Nothing verified a real decay before merge. | Adopted: a live position_eval run with a shrunk cycle crossed every boundary (7 `lr` events at the exact expected positions) before merge. |
| 4 | scope | Decay and floor fractions were per-trainer fields with identical values. | Adopted: module constants; two params per trainer. |
| 5 | hidden complexity | Tags started under a hand-set rate had no migration: resumed under the schedule, their cursor lands wherever the clock says, possibly at the peak with no re-warmup. | Accepted as a limitation; a persisted schedule-origin field was rejected to keep the controller stateless. Such tags finish under the code they started with, or restart. |
| 6 | rival | Split policies: single-decay horizon-aware WSD for `move_set_eval`; monotone warmup plus floored inverse-sqrt for the open-ended trainers; no cycles or re-warmups, which the plan offered no evidence for. | Rejected: the open-ended trainers' output is a checkpoint stream evaluated by match readouts under a moving data window, so periodic annealed checkpoints are the point, and an inverse-sqrt floor is no less arbitrary than a cycle length. Neither side has evidence, and two policies double the surface. The `move_set_eval` half is conceded on the merits (see Open questions). |

Also considered by the author and set aside at the time: ScheduleFree+
(Defazio et al., arXiv 2605.19095), whose "anytime" framing fits the
no-horizon constraint even better, but which was validated only on static
LLM-pretraining corpora and trades schedule tuning for a new set of
hyperparameters. The plain schedule-free optimizer (AdamWScheduleFree) is now
the `schedule_free` arm.

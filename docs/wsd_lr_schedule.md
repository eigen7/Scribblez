# The learning-rate schedule: cyclic warmup-stable-decay

How the generational trainers (`position_eval`, `move_set_eval`,
`max_move_per_lane`) set their learning rate, and why. The code is
`WsdSchedule` / `WsdLrController` in `py/scribblez/generational/controls.py`;
this doc carries the rationale, the design decisions, and the review record
behind them (landed in PR #63, replacing the manual `base_lr` dashboard
control). The one-paragraph summary lives in
[generational_training.md](generational_training.md); read this when you want
to know why it is shaped the way it is or want to change it.

## Why a schedule, and why this one

Before PR #63 every trainer ran AdamW at a rate the operator set by hand in the
dashboard's Controls tab, stepping it down manually off the loss plots. That was
a deliberate choice: an annealing schedule normally assumes a known training
horizon, and these trainers are open-ended — they checkpoint every generation
indefinitely and can be paused/resumed at any point — so there was never an
obvious moment to trigger a decay. `move_set_eval_results.md` documents the
cost: a run trained flat at 1e-3 for 25 passes, plateauing, with no way to test
whether a decay would have helped.

Plain warmup+cosine-to-zero does not fit because it needs the horizon.
**Warmup-Stable-Decay (WSD)** solves half of the problem: its stable phase is
exactly the old flat-LR loop, and the decay is a short tail that can be
triggered without knowing the total run length. But the standard formulation
still assumes one decay at the very end of *a* run, i.e. still a human "decay
now" call. To remove that judgment entirely, the schedule **cycles WSD
periodically**: warm up once, then repeat stable→decay→warm-restart. Each cycle
leaves a well-annealed checkpoint behind, and the sliding data window keeps
getting fresh high-LR passes. Structurally this is close to SGDR warm restarts
with WSD's tail; it is this project's own adaptation to the continual/self-play
setting, not a recipe lifted from a paper.

Considered and set aside: Defazio et al., "ScheduleFree+" (arXiv 2605.19095),
whose "anytime" framing (no annealing phase, stop/resume anywhere) matches the
no-horizon constraint even better. Reasons: (a) validated only on static
LLM-pretraining corpora, not non-stationary sliding-window self-play data;
(b) it trades LR-schedule tuning for a different set of new hyperparameters
(much larger weight-decay range, a decoupled-weight-decay optimizer variant,
β-annealing, restart warmup) rather than eliminating tuning; (c) a single very
fresh paper. Worth a one-workload prototype as a follow-up experiment.

## The schedule

`WsdSchedule(lr, warmup_rows, cycle_rows)` is a pure function of the
rows-clock. With `W = warmup_rows`, `C = cycle_rows`, `R = W // 4`,
`D = LR_DECAY_FRAC (0.2)`, `F = LR_FLOOR_FRAC (0.1)`, and `t = (rows - W) mod C`:

| Segment | Rows | Value |
|---|---|---|
| Warmup (once) | `rows < W` | linear `0 -> lr` |
| Cycle `k >= 0`: | | |
| &nbsp;&nbsp;re-warmup (skipped for `k = 0`) | `t < R` | linear `lr*F -> lr` |
| &nbsp;&nbsp;stable | `R <= t < (1-D)*C` | `lr` |
| &nbsp;&nbsp;decay | `(1-D)*C <= t < C` | cosine `lr -> lr*F` |

Shape decisions:
- **Cosine decay, not linear and not the WSD paper's 1-√.** Cosine's slope
  reaches zero at the segment end, which composes cleanly with the restart that
  follows; linear leaves a slope discontinuity at the jump. The 1-√ shape is
  justified by LLM-pretraining-scale single-decay analysis that does not
  transfer to small periodic cycles.
- **Floor at `0.1·lr`, not ~0.** A restart follows immediately; decaying to
  near-zero right before jumping back up wastes the tail of the decay.
- **Re-warmup on restart, from the floor, inside the cycle.** AdamW's
  second-moment estimate has adapted to the low-LR regime by the end of a
  decay, so a bare jump to peak risks an oversized effective step for the first
  post-restart batches. The ramp is `W // 4` long, derived rather than a param
  to keep the knob count down; it eats the front of the stable segment so the
  period stays exactly `C`.
- **Decay/floor fractions are module constants**, not per-trainer params: no
  trainer has a stated reason to diverge on the schedule *shape*. Promote to
  params if one does.
- **No range validation** of degenerate configs (re-warmup swallowing the
  stable segment): `params.py`'s `validate()` has no cross-field-range
  precedent, and every row count still maps to a value. A bad setting shows up
  on the loss plot like any other mis-set tunable.

## The controller

`WsdLrController(conn, schedule, rows_trained)` is what a trainer holds. It
serves `schedule.value` as `run_epoch`'s per-batch `lr_fn` and keeps two
pieces of in-memory bookkeeping, both re-derived from `rows_trained` on
construction — **nothing is persisted**, so a resume picks up mid-phase with no
new checkpoint state:

- `.current` — the rate applied to the most recent batch. The trainers'
  end-of-generation log line and `metrics.lr` row read it *after* `run_epoch`,
  so they report the end-of-generation value, consistent with the end rows
  count they log next to it. (Reading a start-of-generation value there would
  displace every warmup/decay point on the "Learning rate" plot by a
  generation.)
- `_last_phase` — for boundary detection. A crossing is detected inside the
  per-batch call and written as a `control_event` named `"lr"` at that batch's
  exact rows position (a DB write in the training loop, a handful of times per
  run). Because the phase is initialised from the resume cursor, restarting
  mid-phase logs nothing spurious.

Events fire **only at phase boundaries** (warmup end, each decay start, each
restart), never per generation: the continuous trajectory is already the
per-generation `metrics.lr` series and its log-scale "Learning rate" plot;
`control_event` markers exist to flag structurally notable moments on the
loss-plot overlay and the Controls-tab history table, and per-generation rows
during a decay would just be near-duplicate clutter. The event is named `"lr"`
rather than the old `"base_lr"` because it is a derived, logged value, not
something the operator sets. Old tags' `base_lr` events are still rendered
(exponential format) by the history table.

## Per-trainer sizing

Each trainer's params carry `lr` (the peak), `lr_warmup_rows`, and
`lr_cycle_rows`. The last two are **in that trainer's own rows-clock units**,
which differ: `position_eval` and `max_move_per_lane` count positions;
`move_set_eval` counts candidate moves. Defaults, all starting points to retune
from the loss plots:

- `position_eval` / `max_move_per_lane`: with `games_per_generation=20000,
  window=4, turns_per_game=1` a generation is ~80k rows. `lr_warmup_rows =
  200_000` (~2.5 generations), `lr_cycle_rows = 2_000_000` (~25 generations per
  cycle: several annealed checkpoints over any long run without squeezing the
  stable phase where most training happens).
- `move_set_eval`: `move_set_eval_results.md` records 772M rows over 25 passes,
  ~31M rows/pass — two orders of magnitude above the other two. Copying their
  defaults would end warmup inside the first 1% of pass 1 and thrash through ~15
  cycles per pass. `lr_warmup_rows = 15_000_000` (~0.5 pass), `lr_cycle_rows =
  300_000_000` (~10 passes; ~2.5 cycles over the reference 25-pass run, decays
  landing around passes 12-13 and 22-23). Sized from that one run; re-check
  against a live tag's `positions` deltas per pass.

## Known limitations and follow-ups

- **Existing tags were not migrated.** The clock is absolute `rows_trained`. A
  tag hand-stepped down under the old control, resumed under the schedule with
  its frozen `lr` still at the original peak, lands wherever
  `(rows_trained - W) mod C` says — possibly peak with no re-warmup, the exact
  bare-jump case the re-warmup exists to avoid. The fix (a persisted
  schedule-origin field via the `GenerationalState` subclass mechanism) was
  deliberately left out to keep the controller state-free; such tags finish
  under old code or restart from scratch.
- **`move_set_eval` has a knowable horizon** once its corpus settles
  (`target_pairs` + `train_epochs`), which is the classic single-shot WSD case
  and the direct ask in `move_set_eval_results.md`. A horizon-aware policy for
  it (decay over the last k *settled* epochs, rows-per-settled-epoch measured
  from the first settled pass) is the natural follow-up; the periodic sizing
  above is the interim.
- Whether periodic restarts beat a monotone horizon-free schedule (warmup +
  floored inverse-sqrt, no cycles) on the open-ended trainers is unmeasured
  either way; see the review record below.

## Review record (plan-review panel, 2026-08-17)

The design was reviewed as a plan before implementation by a four-seat panel:
hidden-complexity (Claude, session tier), rival-designer (**codex**, full repo
access), scope (sonnet), integration (sonnet). Every serious critique and how
it landed — kept here because the rejections are decisions someone may want to
revisit.

| # | Panelist | Critique | Resolution |
|---|---|---|---|
| 1 | hidden-complexity, integration (independently) | `.current` as "the value computed at epoch start" lags a generation on the LR plot; phase-crossing detection needs remembered state and a defined emission point; resume needs a no-spurious-event rule. | Adopted: per-batch `.current`, in-loop crossing detection at the exact rows position, phase seeded from the resume cursor. |
| 2 | scope, hidden-complexity, codex (independently) | `move_set_eval` copied the other trainers' defaults despite a ~31M-rows/pass clock. | Adopted: per-workload defaults sized from the results doc; units in help text. |
| 3 | scope | Nothing verified a real decay before merge. | Adopted: a shrunk-cycle live `position_eval` run crossed every boundary (7 `lr` events at the exact expected positions) before the PR opened. |
| 4 | scope | Decay/floor fractions were per-trainer fields with identical values. | Adopted: module constants; two params per trainer. |
| 5 | hidden-complexity | No migration for tags started under the manual control. | Accepted as a limitation (above); persisted origin field rejected to keep the controller state-free. Human call, taken. |
| 6 | codex (rival-designer) | Split policies: single-shot horizon-aware WSD for `move_set_eval`; monotone warmup + floored inverse-sqrt for the open-ended trainers; no cycles or re-warmups (which the plan offered no evidence for). | Rejected for v1: the open-ended trainers' output is a checkpoint stream evaluated by match/arms readouts under a moving data window, so periodic annealed checkpoints are the point, and an inverse-sqrt floor is no less arbitrary than a cycle length; neither side has evidence; two policies doubles the surface. The `move_set_eval` half is conceded on the merits and is the follow-up above. |

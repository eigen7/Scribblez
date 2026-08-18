"""The optimizer arm for the generational trainers: which optimizer runs, and
how -- or whether -- its learning rate is scheduled.

Two arms, selected by a frozen task param:

- `wsd`: AdamW driven by the rows-clock warmup-stable-decay schedule
  (controls.WsdSchedule). The rate is a function of rows trained, so a
  checkpoint's quality depends on where in the cycle it was exported: the good
  ones are those at the end of a decay.
- `schedule_free`: AdamWScheduleFree (Defazio et al., 2024), which drops the
  schedule in favour of an averaged iterate. Nothing has to be timed against a
  horizon the open-ended runs do not have, and every generation's export is
  equally deployable.

The averaging is why the arms need a mode: schedule-free training steps at one
point (`y`) and deploys another (`x`), keeping the difference in optimizer
state, so the live weights have to be swapped before anything reads the model
and swapped back afterwards. The trainer brackets its phases with train_mode /
eval_mode rather than knowing which arm needs them; under `wsd` they do
nothing, because there the live weights are the only weights.

Both arms present one surface: the per-step `lr_fn` run_epoch applies (None
when the arm imposes no schedule), the `current` rate for the metrics row, and
the two mode hooks.
"""

from __future__ import annotations

import torch
from schedulefree import AdamWScheduleFree

from .controls import WsdLrController, WsdSchedule

OPTIMIZER_WSD = "wsd"
OPTIMIZER_SCHEDULE_FREE = "schedule_free"
OPTIMIZERS = (OPTIMIZER_WSD, OPTIMIZER_SCHEDULE_FREE)


def build_optimizer(model, params):
    """The run's optimizer, per `params.optimizer`.

    Built before the rolling checkpoint is resumed, so it carries no rows-clock
    state: the schedule-free arm's own warmup is counted in optimizer steps,
    converted here from the same `lr_warmup_rows` the WSD arm ramps over so one
    knob covers both."""
    if params.optimizer == OPTIMIZER_SCHEDULE_FREE:
        return AdamWScheduleFree(
            model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay,
            warmup_steps=params.lr_warmup_rows // params.batch_size,
        )
    if params.optimizer == OPTIMIZER_WSD:
        return torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    raise ValueError(f"unknown optimizer {params.optimizer!r}; expected one of {OPTIMIZERS}")


class WsdArm:
    """AdamW under the rows-clock WSD schedule, with no mode to switch."""

    def __init__(self, conn, params, rows_trained: int):
        self._controller = WsdLrController(conn, WsdSchedule.from_params(params), rows_trained)
        self.lr_fn = self._controller.lr_fn

    @property
    def current(self) -> float:
        return self._controller.current

    def train_mode(self):
        pass

    def eval_mode(self):
        pass


class ScheduleFreeArm:
    """AdamWScheduleFree at a constant rate, plus the averaged-iterate swap.

    `lr_fn` is None so run_epoch leaves the rate alone -- the optimizer's own
    warmup is the only ramp, and after it the rate never moves. `current`
    reports that constant so the dashboard's learning-rate plot still says what
    the generation ran at (a flat line being the honest picture of an arm that
    schedules nothing)."""

    def __init__(self, params, optimizer):
        self._optimizer = optimizer
        self.lr_fn = None
        self.current = params.lr

    def train_mode(self):
        self._optimizer.train()

    def eval_mode(self):
        self._optimizer.eval()


def build_optim_arm(conn, params, optimizer, rows_trained: int):
    """The arm driving `optimizer`, picking up at `rows_trained`."""
    if params.optimizer == OPTIMIZER_SCHEDULE_FREE:
        return ScheduleFreeArm(params, optimizer)
    return WsdArm(conn, params, rows_trained)

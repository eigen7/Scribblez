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
when the arm imposes no schedule), the `current` rate for the metrics row, any
extra `metrics()` the arm wants recorded alongside it, and the two mode hooks.
"""

from __future__ import annotations

import torch
from schedulefree import AdamWScheduleFree

from .controls import WsdLrController, WsdSchedule
from .optimizer_arms import DEFAULT_LR, OPTIMIZER_SCHEDULE_FREE, OPTIMIZERS


def arm_lr(params) -> float:
    """The run's learning rate: the task's, or its arm's default when unset."""
    if params.optimizer not in DEFAULT_LR:
        raise ValueError(f"unknown optimizer {params.optimizer!r}; expected one of {OPTIMIZERS}")
    return params.lr or DEFAULT_LR[params.optimizer]


def build_optimizer(model, params):
    """The run's optimizer, per `params.optimizer`.

    Built before the rolling checkpoint is resumed, so it carries no rows-clock
    state: the schedule-free arm's own warmup is counted in optimizer steps,
    converted here from the same `lr_warmup_rows` the WSD arm ramps over so one
    knob covers both."""
    lr = arm_lr(params)
    if params.optimizer == OPTIMIZER_SCHEDULE_FREE:
        return AdamWScheduleFree(
            model.parameters(),
            lr=lr,
            weight_decay=params.weight_decay,
            warmup_steps=params.lr_warmup_rows // params.batch_size,
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=params.weight_decay)


class WsdArm:
    """AdamW under the rows-clock WSD schedule, with no mode to switch."""

    #: Nothing beyond `current`: the rate is the whole story for this arm.
    metrics = staticmethod(dict)

    def __init__(self, conn, params, rows_trained: int):
        schedule = WsdSchedule(arm_lr(params), params.lr_warmup_rows, params.lr_cycle_rows)
        self._controller = WsdLrController(conn, schedule, rows_trained)
        self.lr_fn = self._controller.lr_fn

    @property
    def current(self) -> float:
        return self._controller.current

    def train_mode(self):
        pass

    def eval_mode(self):
        pass


def _averaging_weight(group) -> float:
    """The weight the last step gave the base iterate in the deployed average.

    This is where a schedule-free run does its annealing: the rate never moves,
    but each new base iterate enters the average with a smaller share of it, so
    the deployed weights settle the way a decaying step size would settle them.
    With the default `r`, the share falls off as 1/k.

    schedulefree computes the weight inside step() and does not keep it, so it
    is recomputed here from the group -- `k` has already been incremented past
    the step, which is exactly the exponent that step used."""
    if not group["weight_sum"]:
        return 1.0  # before the first step the base iterate would be the average
    weight = group["k"] ** group["r"] * group["lr_max"] ** group["weight_lr_power"]
    return weight / group["weight_sum"]


class ScheduleFreeArm:
    """AdamWScheduleFree at a constant rate, plus the averaged-iterate swap.

    `lr_fn` is None so run_epoch leaves the rate alone: this arm anneals by
    averaging rather than by stepping the rate down, so nothing per-batch has
    to move it. `current` reports the rate the optimizer actually applied,
    which is the constant except during its own warmup ramp; the annealing that
    the rate no longer shows is recorded separately by `metrics`."""

    def __init__(self, params, optimizer):
        self._optimizer = optimizer
        self.lr_fn = None
        self._nominal_lr = arm_lr(params)

    @property
    def _group(self) -> dict:
        return self._optimizer.param_groups[0]

    @property
    def current(self) -> float:
        # scheduled_lr is 0 until the first step has ramped it.
        return self._group["scheduled_lr"] or self._nominal_lr

    def metrics(self) -> dict:
        """The averaging weight, so the Training tab plots what actually
        anneals here rather than a flat rate that looks like nothing moved."""
        return {"averaging_weight": _averaging_weight(self._group)}

    def train_mode(self):
        self._optimizer.train()

    def eval_mode(self):
        self._optimizer.eval()


def build_optim_arm(conn, params, optimizer, rows_trained: int):
    """The arm driving `optimizer`, picking up at `rows_trained`."""
    if params.optimizer == OPTIMIZER_SCHEDULE_FREE:
        return ScheduleFreeArm(params, optimizer)
    return WsdArm(conn, params, rows_trained)

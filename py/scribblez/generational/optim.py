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

The averaging is why the arms need modes. Schedule-free training steps at one
point (`y`) and deploys another (`x`), keeping the difference in optimizer
state, so the live weights must be swapped before anything reads the model
and swapped back afterwards. For a model with BatchNorm the swap is not
enough: the running statistics accumulate during training, at `y`, and are
wrong for `x` (the schedulefree README's BatchNorm caveat). On the
position-evaluation large test set, a schedule-free export evaluated that way
had badly under-confident placement heads (calibration slope 1.44 instead of
~1.05) and a 10% worse win MAE. So eval_mode also recomputes every BatchNorm
layer's statistics at `x` over a few training batches, which restores both.
The trainer brackets its phases with train_mode / eval_mode without knowing
which arm needs them; under `wsd` they do nothing, because the live weights
are the only weights and the statistics already match them.

Both arms decay only the tensors weight decay is meant for (decay_groups):
the optimizer gets a decay group and a no-decay group, in that order.

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


# Parameters that hold positional geometry rather than features: the
# transformer tower's learnable RoPE frequencies and register-token positions.
# Decay would pull them toward zero, coarsening every head's positional
# resolution and collapsing the registers onto the origin.
_GEOMETRY_SUFFIXES = ("rope_freqs", "register_pos")


def decays(name: str, param: torch.Tensor) -> bool:
    """Whether weight decay applies to the parameter `name`: to weight
    matrices and convolution kernels, but not to vectors (norm gains, biases)
    or positional geometry (_GEOMETRY_SUFFIXES). The register tokens' embedding
    table is a matrix and decays like any other."""
    return param.ndim >= 2 and not name.endswith(_GEOMETRY_SUFFIXES)


def decay_groups(model, weight_decay: float) -> list[dict]:
    """`model`'s parameters as [decay group, no-decay group], the decay group
    at `weight_decay`, the other at 0 (see decays). The decay group comes
    first: ScheduleFreeArm reads its rate and averaging weight off
    param_groups[0]."""
    named = list(model.named_parameters())
    return [
        {"params": [p for n, p in named if decays(n, p)], "weight_decay": weight_decay},
        {"params": [p for n, p in named if not decays(n, p)], "weight_decay": 0.0},
    ]


def build_optimizer(model, params, rows_per_step: int | None = None):
    """The run's optimizer, per `params.optimizer`.

    Built before the rolling checkpoint is resumed, so it carries no rows-clock
    state: the schedule-free arm's own warmup is counted in optimizer steps,
    converted here from the same `lr_warmup_rows` the WSD arm ramps over so one
    knob covers both.

    `rows_per_step` is the mean training rows per optimizer step, which turns
    that row-count warmup into the step count AdamWScheduleFree wants. It
    defaults to `params.batch_size` -- the rows-per-step of a trainer whose
    rows-clock counts the same unit its batches are sized in (position_eval). A
    trainer whose clock counts a different unit passes its own conversion
    (move_set_eval counts candidate moves but batches by position, so a step is
    many rows)."""
    lr = arm_lr(params)
    groups = decay_groups(model, params.weight_decay)
    betas = (0.9, params.adam_beta2)
    if rows_per_step is None:
        rows_per_step = params.batch_size
    if params.optimizer == OPTIMIZER_SCHEDULE_FREE:
        return AdamWScheduleFree(
            groups,
            lr=lr,
            betas=betas,
            warmup_steps=int(params.lr_warmup_rows // max(rows_per_step, 1)),
        )
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def _model_forward(model, spatial, scalar):
    """Default BatchNorm-exercising forward for recalibration: the whole model
    over one (spatial, scalar) board-input pair. A trainer whose BatchNorm sits
    behind more than the plain board pass passes its own (move_set_eval's heads
    take a candidate set the recalibration has no reason to build)."""
    model(spatial, scalar)


class WsdArm:
    """AdamW under the rows-clock WSD schedule, with no mode to switch."""

    #: No metrics beyond `current`.
    metrics = staticmethod(dict)

    def __init__(self, recorder, params, rows_trained: int):
        schedule = WsdSchedule(arm_lr(params), params.lr_warmup_rows, params.lr_cycle_rows)
        self._controller = WsdLrController(recorder, schedule, rows_trained)
        self.lr_fn = self._controller.lr_fn

    @property
    def current(self) -> float:
        return self._controller.current

    def train_mode(self):
        pass

    def eval_mode(self, model, batches, forward_fn=_model_forward):
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

    def eval_mode(self, model, batches, forward_fn=_model_forward):
        """Swap the averaged iterate in and recompute `model`'s BatchNorm
        statistics for it over `batches` (an iterable of (spatial, scalar)
        input pairs on the model's device; a few dozen batches suffice).
        `forward_fn(model, spatial, scalar)` is the pass that exercises the
        BatchNorm layers -- the whole model by default (see _model_forward)."""
        self._optimizer.eval()
        recalibrate_batchnorm(model, batches, forward_fn)


@torch.no_grad()
def recalibrate_batchnorm(model, batches, forward_fn=_model_forward):
    """Replace every BatchNorm layer's running statistics with the exact
    (cumulative, momentum=None) mean/variance of the live weights' activations
    over `batches`, leaving the model in eval mode. `forward_fn(model, spatial,
    scalar)` runs the pass that reaches those layers. Each layer's momentum is
    restored afterwards so training resumes with the ordinary running update."""
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bn_layers]
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for spatial, scalar in batches:
        forward_fn(model, spatial, scalar)
    model.eval()
    for m, momentum in zip(bn_layers, momenta, strict=True):
        m.momentum = momentum


def build_optim_arm(recorder, params, optimizer, rows_trained: int):
    """The arm driving `optimizer`, picking up at `rows_trained`. `recorder`
    (generational/records.py) takes the WSD schedule's phase-boundary events."""
    if params.optimizer == OPTIMIZER_SCHEDULE_FREE:
        return ScheduleFreeArm(params, optimizer)
    return WsdArm(recorder, params, rows_trained)

"""Unit tests for the optimizer arms (scribblez/generational/optim.py).

The schedule-free arm keeps two sets of weights and swaps the live ones, so
the tests that matter here are about the swap: that a checkpoint written in
eval mode is what a deployed model would see, that resuming from one returns to
exactly the training weights the run left off at, and that BatchNorm statistics
are recomputed for the swapped-in weights. Getting the first two wrong corrupts
a resumed run silently; the third silently degrades every export.
"""

from dataclasses import dataclass

import pytest
import torch
from scribblez.generational.checkpoint import GenerationalState, resume, save
from scribblez.generational.optim import (
    ScheduleFreeArm,
    WsdArm,
    arm_lr,
    build_optim_arm,
    build_optimizer,
)
from scribblez.generational.optimizer_arms import (
    DEFAULT_LR,
    OPTIMIZER_SCHEDULE_FREE,
    OPTIMIZER_WSD,
)
from scribblez.paths import POSITION_EVAL, TagPaths

_CPU = torch.device("cpu")


@dataclass
class _Params:
    """The fields the arms read off a trainer's params dataclass."""

    optimizer: str = OPTIMIZER_SCHEDULE_FREE
    lr: float = 0.0
    weight_decay: float = 1e-4
    batch_size: int = 8
    lr_warmup_rows: int = 80
    lr_cycle_rows: int = 800


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


def _step(model, opt, n: int = 3):
    for i in range(n):
        opt.zero_grad()
        model(torch.randn(3, 4, generator=torch.Generator().manual_seed(i))).sum().backward()
        opt.step()


def test_build_optimizer_picks_the_arm():
    model = _model()
    sf = build_optimizer(model, _Params(optimizer=OPTIMIZER_SCHEDULE_FREE))
    assert type(sf).__name__ == "AdamWScheduleFree"
    # Warmup is expressed in rows by the params and in steps by the optimizer.
    assert sf.param_groups[0]["warmup_steps"] == 10
    assert isinstance(build_optimizer(model, _Params(optimizer=OPTIMIZER_WSD)), torch.optim.AdamW)


def test_an_unknown_optimizer_is_rejected():
    try:
        build_optimizer(_model(), _Params(optimizer="cosine"))
    except ValueError as e:
        assert "cosine" in str(e)
    else:
        raise AssertionError("expected a ValueError")


def test_the_arms_report_their_schedule():
    """The schedule-free arm imposes no per-step rate; the WSD arm does."""
    model = _model()
    params = _Params(optimizer=OPTIMIZER_SCHEDULE_FREE)
    arm = build_optim_arm(None, params, build_optimizer(model, params), 0)
    assert isinstance(arm, ScheduleFreeArm)
    assert arm.lr_fn is None
    assert arm.current == DEFAULT_LR[OPTIMIZER_SCHEDULE_FREE]


def test_the_wsd_arm_still_drives_the_rows_clock_schedule():
    model = _model()
    params = _Params(optimizer=OPTIMIZER_WSD)
    arm = build_optim_arm(None, params, build_optimizer(model, params), 0)
    assert isinstance(arm, WsdArm)
    # Mid-warmup, the schedule is below the peak and rising off the rows clock.
    assert arm.lr_fn(40) < arm.lr_fn(60) <= DEFAULT_LR[OPTIMIZER_WSD]
    arm.train_mode()  # no-op under this arm, but the trainer calls it either way
    arm.eval_mode(model, [])


def test_schedule_free_eval_mode_swaps_to_different_weights():
    """The deployed weights are the averaged iterate, not the training one --
    if these matched, the mode switch would be doing nothing."""
    model = _model()
    params = _Params()
    opt = build_optimizer(model, params)
    arm = ScheduleFreeArm(params, opt)
    arm.train_mode()
    _step(model, opt)
    training = [p.detach().clone() for p in model.parameters()]
    arm.eval_mode(model, [])
    assert not all(torch.equal(a, b) for a, b in zip(training, model.parameters(), strict=True))


def test_schedule_free_survives_an_eval_mode_checkpoint_roundtrip(tmp_path):
    """The trainer checkpoints inside its eval-mode bracket, so a resume has to
    recover the training weights from averaged ones plus optimizer state."""
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    params = _Params()

    model = _model()
    opt = build_optimizer(model, params)
    arm = ScheduleFreeArm(params, opt)
    arm.train_mode()
    _step(model, opt)
    expected = [p.detach().clone() for p in model.parameters()]

    arm.eval_mode(model, [])  # what _checkpoint_and_eval runs inside
    save(paths, model, opt, GenerationalState(24, 3), {})

    model2 = _model()
    opt2 = build_optimizer(model2, params)
    arm2 = ScheduleFreeArm(params, opt2)
    assert resume(paths, model2, opt2, _CPU) == GenerationalState(24, 3)
    arm2.train_mode()  # what the next generation runs

    for a, b in zip(expected, model2.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6), "resumed run diverged from the training weights"


def _bn_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))


def _bn_step(model, opt, n: int = 3):
    for i in range(n):
        opt.zero_grad()
        model(torch.randn(8, 4, generator=torch.Generator().manual_seed(i))).sum().backward()
        opt.step()


def test_schedule_free_eval_mode_recomputes_batchnorm_for_the_deployed_weights():
    """Running statistics accumulate at the training weights; eval_mode must
    replace them with the deployed weights' own, exactly (a cumulative average
    over the given batches), and hand the momentum back for the next epoch."""
    model = _bn_model()
    params = _Params()
    opt = build_optimizer(model, params)
    arm = ScheduleFreeArm(params, opt)
    arm.train_mode()
    _bn_step(model, opt)
    bn = model[1]
    momentum = bn.momentum
    x = torch.randn(32, 4, generator=torch.Generator().manual_seed(99))
    batches = [(x[:16], None), (x[16:], None)]

    # A scalar-free stand-in for the trainer's (spatial, scalar) pairs.
    class _Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, spatial, scalar):
            return self.inner(spatial)

    arm.eval_mode(_Wrapped(model), batches)
    assert not model.training
    assert bn.momentum == momentum
    with torch.no_grad():  # the deployed (averaged) weights' pre-BN activations
        pre = [model[0](b) for b, _ in batches]
    # A cumulative average (momentum=None) is the mean over batches of each
    # batch's statistics, the variance unbiased as BatchNorm tracks it.
    assert torch.allclose(bn.running_mean, torch.stack([a.mean(0) for a in pre]).mean(0), atol=1e-6)
    assert torch.allclose(
        bn.running_var, torch.stack([a.var(0, unbiased=True) for a in pre]).mean(0), atol=1e-5
    )


def test_an_unset_rate_falls_back_to_the_arms_default():
    """`lr` means different things to the two arms, so one default cannot serve
    both: 0 asks for whichever the chosen arm wants."""
    assert arm_lr(_Params(optimizer=OPTIMIZER_WSD)) == DEFAULT_LR[OPTIMIZER_WSD]
    assert arm_lr(_Params(optimizer=OPTIMIZER_SCHEDULE_FREE)) == DEFAULT_LR[OPTIMIZER_SCHEDULE_FREE]
    assert DEFAULT_LR[OPTIMIZER_WSD] != DEFAULT_LR[OPTIMIZER_SCHEDULE_FREE]


def test_a_named_rate_wins_over_the_arms_default():
    assert arm_lr(_Params(optimizer=OPTIMIZER_SCHEDULE_FREE, lr=3e-4)) == 3e-4


def test_the_resolved_rate_reaches_both_arms():
    """Whatever arm_lr returns is what the optimizer and the schedule run at --
    not the raw 0 the task left behind."""
    model = _model()
    sf = _Params(optimizer=OPTIMIZER_SCHEDULE_FREE)
    assert build_optimizer(model, sf).param_groups[0]["lr"] == DEFAULT_LR[OPTIMIZER_SCHEDULE_FREE]
    wsd = _Params(optimizer=OPTIMIZER_WSD)
    arm = build_optim_arm(None, wsd, build_optimizer(model, wsd), wsd.lr_warmup_rows)
    assert arm.current == DEFAULT_LR[OPTIMIZER_WSD]


def test_the_wsd_arm_reports_no_extra_metrics():
    """`lr` already tells the whole story there, so the averaging-weight panel
    stays absent on a WSD run rather than showing a meaningless flat line."""
    assert WsdArm(None, _Params(optimizer=OPTIMIZER_WSD), 0).metrics() == {}


def test_the_averaging_weight_falls_off_as_training_proceeds():
    """The schedule-free arm's annealing lives here: the rate is constant, but
    each new base iterate enters the deployed average with a smaller share."""
    model = _model()
    params = _Params(lr_warmup_rows=0)
    opt = build_optimizer(model, params)
    arm = ScheduleFreeArm(params, opt)
    arm.train_mode()

    _step(model, opt, n=1)
    first = arm.metrics()["averaging_weight"]
    _step(model, opt, n=9)
    tenth = arm.metrics()["averaging_weight"]

    assert first == 1.0  # the first base iterate is the whole average
    assert tenth == pytest.approx(0.1)  # ... and the tenth is a tenth of it
    assert tenth < first


def test_the_reported_rate_is_the_one_the_optimizer_applied():
    """During its own warmup the schedule-free arm is below its nominal rate,
    so `current` reports the ramp rather than the constant it is heading for."""
    model = _model()
    params = _Params(lr_warmup_rows=80, batch_size=8)  # 10 steps of warmup
    opt = build_optimizer(model, params)
    arm = ScheduleFreeArm(params, opt)
    nominal = DEFAULT_LR[OPTIMIZER_SCHEDULE_FREE]

    assert arm.current == nominal  # before any step, the nominal rate
    arm.train_mode()
    _step(model, opt, n=1)
    assert arm.current == pytest.approx(nominal / 10)  # one step into the ramp
    _step(model, opt, n=19)
    assert arm.current == nominal  # past it

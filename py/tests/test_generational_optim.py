"""Unit tests for the optimizer arms (scribblez/generational/optim.py).

The schedule-free arm keeps two sets of weights and swaps the live ones, so
the tests that matter here are about the swap: that a checkpoint written in
eval mode is what a deployed model would see, and that resuming from one
returns to exactly the training weights the run left off at. Getting that
wrong corrupts a resumed run silently.
"""

from dataclasses import dataclass

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
    arm.eval_mode()


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
    arm.eval_mode()
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

    arm.eval_mode()  # what _checkpoint_and_eval runs inside
    save(paths, model, opt, GenerationalState(24, 3), {})

    model2 = _model()
    opt2 = build_optimizer(model2, params)
    arm2 = ScheduleFreeArm(params, opt2)
    assert resume(paths, model2, opt2, _CPU) == GenerationalState(24, 3)
    arm2.train_mode()  # what the next generation runs

    for a, b in zip(expected, model2.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6), "resumed run diverged from the training weights"


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

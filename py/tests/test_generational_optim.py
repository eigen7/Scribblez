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
from schedulefree import AdamWScheduleFree
from scribblez.generational.checkpoint import (
    GenerationalState,
    regroup_optimizer_state,
    resume,
    save,
)
from scribblez.generational.optim import (
    ScheduleFreeArm,
    WsdArm,
    arm_lr,
    build_optim_arm,
    build_optimizer,
    decay_groups,
    decays,
    recalibrate_batchnorm,
)
from scribblez.generational.optimizer_arms import (
    DEFAULT_LR,
    OPTIMIZER_SCHEDULE_FREE,
    OPTIMIZER_WSD,
)
from scribblez.paths import POSITION_EVAL, TagPaths
from scribblez.position_eval.model import PositionEvalModel
from scribblez.supply_registers import SCALAR_SIZE_OPEN_LEAVES
from scribblez.transformer_tower import TransformerConfig

_CPU = torch.device("cpu")


@dataclass
class _Params:
    """The fields the arms read off a trainer's params dataclass."""

    optimizer: str = OPTIMIZER_SCHEDULE_FREE
    lr: float = 0.0
    weight_decay: float = 1e-4
    adam_beta2: float = 0.999
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


def test_build_optimizer_sizes_warmup_by_rows_per_step():
    """A trainer whose rows-clock counts a unit other than its batch size passes
    rows_per_step, and the schedule-free warmup is counted off it rather than
    off params.batch_size (which such a trainer need not even have)."""
    model = _model()
    params = _Params(optimizer=OPTIMIZER_SCHEDULE_FREE, lr_warmup_rows=800, batch_size=8)
    # batch_size alone would give 100 warmup steps; a wider row-per-step, fewer.
    assert build_optimizer(model, params, rows_per_step=200).param_groups[0]["warmup_steps"] == 4
    assert build_optimizer(model, params).param_groups[0]["warmup_steps"] == 100  # the default


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


class _BoardOnly(torch.nn.Module):
    """A model whose BatchNorm is reached only through a named sub-forward, not
    its plain forward -- a stand-in for the mset model, whose BatchNorm lives in
    the board trunk and whose forward wants a candidate set recalibration has no
    reason to build."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.bn = torch.nn.BatchNorm1d(3)

    def encode(self, spatial, scalar):
        return self.bn(self.lin(spatial))

    def forward(self, spatial, scalar):
        raise AssertionError("the plain forward must not drive recalibration here")


def _encode_forward(model, spatial, scalar):
    model.encode(spatial, scalar)


def test_recalibrate_batchnorm_runs_the_given_forward():
    """The BatchNorm-exercising pass is the caller's forward_fn: the default is
    the whole model, but a trainer whose statistics sit behind a sub-forward
    supplies its own, and only that pass runs."""
    torch.manual_seed(0)
    model = _BoardOnly()
    x = torch.randn(16, 3, generator=torch.Generator().manual_seed(1))
    batches = [(x[:8], None), (x[8:], None)]

    recalibrate_batchnorm(model, batches, _encode_forward)
    assert not model.training
    with torch.no_grad():
        pre = [model.lin(b) for b, _ in batches]
    assert torch.allclose(
        model.bn.running_mean, torch.stack([a.mean(0) for a in pre]).mean(0), atol=1e-6
    )
    # Omitting it falls back to the plain forward, which here is the wrong pass.
    with pytest.raises(AssertionError):
        recalibrate_batchnorm(model, batches)


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


def _transformer_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return PositionEvalModel(
        87,
        SCALAR_SIZE_OPEN_LEAVES,
        trunk_channels=16,
        num_blocks=2,
        transformer=TransformerConfig(mid_channels=8, num_heads=2, ffn_channels=16),
    )


def _conv_model() -> torch.nn.Module:
    return PositionEvalModel(87, SCALAR_SIZE_OPEN_LEAVES, trunk_channels=16, num_blocks=2)


@pytest.mark.parametrize("make_model", [_transformer_model, _conv_model])
def test_decay_groups_partition_the_model(make_model):
    """Every parameter lands in exactly one group, the decay group first, and
    only matrices and kernels decay."""
    model = make_model()
    decay, no_decay = decay_groups(list(model.named_parameters()), 0.05)
    assert (decay["weight_decay"], no_decay["weight_decay"]) == (0.05, 0.0)
    ids = [id(p) for p in decay["params"] + no_decay["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())
    assert all(p.ndim >= 2 for p in decay["params"])
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert all(id(p) in no_decay_ids for p in model.parameters() if p.ndim < 2)


def test_decay_groups_spare_the_transformers_positional_geometry():
    """RoPE frequencies and register positions are matrices by shape but
    geometry by meaning; the register embedding table is an ordinary matrix."""
    names = {n: p for n, p in _transformer_model().named_parameters()}
    geometry = [n for n in names if n.endswith(("rope_freqs", "register_pos"))]
    assert any(n.endswith("rope_freqs") for n in geometry)
    assert any(n.endswith("register_pos") for n in geometry)
    for n in geometry:
        assert names[n].ndim >= 2 and not decays(n, names[n])
    embeds = [n for n in names if "registers" in n and names[n].ndim == 2]
    assert embeds and all(decays(n, names[n]) for n in embeds)


@pytest.mark.parametrize("optimizer", [OPTIMIZER_SCHEDULE_FREE, OPTIMIZER_WSD])
def test_build_optimizer_uses_the_decay_groups_and_beta2(optimizer):
    params = _Params(optimizer=optimizer, weight_decay=0.05, adam_beta2=0.95)
    opt = build_optimizer(_ln_model(), params)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.05, 0.0]
    assert all(g["betas"] == (0.9, 0.95) for g in opt.param_groups)


def _ln_model() -> torch.nn.Module:
    """Matrices and vectors interleaved in parameter order (weight, bias, gain,
    bias, weight, bias), so a state regrouping that confused positions would
    hand a tensor another's state."""
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3), torch.nn.Linear(3, 2))


def test_a_single_group_checkpoint_resumes_into_the_split_groups(tmp_path):
    """A rolling checkpoint written by a one-group optimizer (every run before
    the no-decay split) resumes into the two groups and continues exactly as
    the one-group optimizer would have. Weight decay is 0 so the two agree."""
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    legacy_model = _ln_model()
    legacy = AdamWScheduleFree(legacy_model.parameters(), lr=1e-2, weight_decay=0.0)
    legacy.train()
    _step(legacy_model, legacy)
    save(paths, legacy_model, legacy, GenerationalState(3, 1), {})

    model = _ln_model()
    opt = build_optimizer(model, _Params(lr=1e-2, weight_decay=0.0, lr_warmup_rows=0))
    assert len(opt.param_groups) == 2
    resume(paths, model, opt, _CPU)
    opt.train()
    assert [g["k"] for g in opt.param_groups] == [3, 3]  # the saved step count, per group

    _step(legacy_model, legacy, n=3)
    _step(model, opt, n=3)
    for a, b in zip(legacy_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(a, b), "a regrouped tensor lost or swapped its optimizer state"


def test_regrouping_keeps_each_groups_own_weight_decay(tmp_path):
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    legacy_model = _ln_model()
    legacy = AdamWScheduleFree(legacy_model.parameters(), weight_decay=1e-4)
    save(paths, legacy_model, legacy, GenerationalState(), {})

    model = _ln_model()
    opt = build_optimizer(model, _Params(weight_decay=0.1))
    resume(paths, model, opt, _CPU)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]


def test_regrouping_passes_a_matching_state_dict_through():
    model = _ln_model()
    opt = build_optimizer(model, _Params())
    saved = opt.state_dict()
    assert regroup_optimizer_state(saved, model, opt) is saved


def test_schedule_free_mode_swaps_cover_both_groups():
    """eval()/train() must move the no-decay tensors between the training and
    averaged iterates too, or the deployed model would mix the two."""
    model = _ln_model()
    opt = build_optimizer(model, _Params(lr_warmup_rows=0))
    opt.train()
    _step(model, opt)
    training = [p.detach().clone() for p in model.parameters()]
    opt.eval()
    for group in opt.param_groups:
        assert any(
            not torch.equal(p, training[i])
            for i, q in enumerate(model.parameters())
            for p in group["params"]
            if p is q
        )
    opt.train()
    for a, b in zip(training, model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)

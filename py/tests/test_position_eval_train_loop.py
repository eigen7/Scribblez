"""Unit tests for the shared position evaluation training-epoch loop.

A tiny real PositionEvalModel (small trunk, few blocks) over small inputs
exercises run_epoch's forward/backward/accumulate and its learning-rate
handling, and drives the per-head loss registry through compute_loss.
"""

import torch
import torch.nn.functional as F
from scribblez.position_eval.model import (
    FOOTPRINT_CLASSES,
    FOOTPRINT_EXTRA_CLASS,
    MAD_TO_STD,
    PLACEMENT_HEAD_NAMES,
    PLACEMENT_MASK_NAMES,
    PositionEvalModel,
    _head_legal_mask,
    _placement_ce,
)
from scribblez.position_eval.train_loop import EpochResult, LossConfig, run_epoch

_LOSS_CFG = LossConfig(
    lambda_wld=1.0,
    lambda_sd=0.004,
    lambda_next_placement=0.5,
    lambda_win_placement=0.5,
    huber_delta_mean=10.0,
    huber_delta_std=10.0,
    mask_placement=True,
)
_CPU = torch.device("cpu")
_SCALAR_SIZE = 8
_SPATIAL_PLANES = 2

# The label keys compute_loss reads: a class index per head plus the two per-side
# legality masks.
_TARGET_KEYS = ["wld", "score_diff", *PLACEMENT_HEAD_NAMES, *PLACEMENT_MASK_NAMES]


def _model() -> PositionEvalModel:
    """A small real model on the _batch() input shapes -- a genuine head
    registry, cheap enough (tiny trunk, 2 blocks) for a unit test."""
    return PositionEvalModel(
        spatial_planes=_SPATIAL_PLANES,
        scalar_size=_SCALAR_SIZE,
        trunk_channels=8,
        num_blocks=2,
        board_size=15,
    )


def _batch(bs: int = 4) -> dict:
    wld = torch.zeros(bs, 3)
    wld[torch.arange(bs), torch.randint(0, 3, (bs,))] = 1.0
    batch = {
        "input_spatial": torch.randn(bs, _SPATIAL_PLANES, 15, 15),
        "input_scalar": torch.randn(bs, _SCALAR_SIZE),
        "wld": wld,
        "score_diff": torch.randn(bs, 1) * 20,
    }
    for name in PLACEMENT_HEAD_NAMES:  # a footprint class index per head
        batch[name] = torch.randint(0, FOOTPRINT_CLASSES, (bs, 1)).float()
    for name in PLACEMENT_MASK_NAMES:  # a non-trivial legality mask per side
        batch[name] = (torch.rand(bs, FOOTPRINT_CLASSES) > 0.5).float()
    return batch


def test_run_epoch_accumulates_and_counts():
    torch.manual_seed(0)
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    result = run_epoch(model, opt, [_batch() for _ in range(3)], _CPU, _LOSS_CFG, rows_trained=100)
    assert isinstance(result, EpochResult)
    assert result.n_batches == 3
    assert result.samples == 12
    assert result.rows_trained == 112  # 100 + 3 * 4
    assert 0.0 <= result.wld_acc <= 1.0
    assert set(result.losses) >= {"total", "wld", "score_diff", *PLACEMENT_HEAD_NAMES}
    assert result.losses["total"] > 0.0


def test_run_epoch_takes_a_gradient_step():
    torch.manual_seed(0)
    model = _model()
    before = model.heads["wld"].fc[0].weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    run_epoch(model, opt, [_batch()], _CPU, _LOSS_CFG)
    assert not torch.equal(before, model.heads["wld"].fc[0].weight)


def test_run_epoch_applies_lr_fn_per_step():
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=999.0)  # overwritten by lr_fn
    seen = []

    def lr_fn(rows: int) -> float:
        lr = 0.01 + rows  # varies with the running row count
        seen.append(lr)
        return lr

    run_epoch(model, opt, [_batch(), _batch()], _CPU, _LOSS_CFG, lr_fn=lr_fn, rows_trained=0)
    # Called once per step, with the row count as it stood before that step.
    assert seen == [0.01, 4.01]
    assert opt.param_groups[0]["lr"] == seen[-1]


def test_run_epoch_leaves_lr_untouched_without_lr_fn():
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.123)
    run_epoch(model, opt, [_batch()], _CPU, _LOSS_CFG)
    assert opt.param_groups[0]["lr"] == 0.123


def test_masked_placement_guards_the_target_class():
    """The NaN guard force-keeps the target class, so the masked softmax-CE stays
    finite even with an all-illegal side mask -- a data-dependent gap must degrade
    to an unmasked target, never to -log(0). With every class but the target
    illegal, a plays head has only the target legal so its CE is ~0; a leaky mask
    (a large finite fill instead of -inf) would leave mass elsewhere and fail
    that. A win head keeps its not-win (extra) class legal too, so its effective
    mask holds exactly {target, extra} and its CE is finite but > 0 -- checked
    both directly (the head's mask opens the extra slot) and through the loss (a
    regression that dropped the win-head extra-opening would leave a target-only
    mask and a 0 loss)."""
    torch.manual_seed(0)
    model = _model()
    batch = _batch()
    out = model(batch["input_spatial"], batch["input_scalar"])
    targets = {k: batch[k] for k in _TARGET_KEYS}
    for name in PLACEMENT_MASK_NAMES:  # the adversarial case: nothing legal
        targets[name] = torch.zeros_like(targets[name])
    losses = model.compute_loss(out, targets, mask_placement=True)
    for name in PLACEMENT_HEAD_NAMES:
        assert torch.isfinite(losses[name]), f"{name} loss is not finite"
        extra_legal = _head_legal_mask(name, targets)[:, FOOTPRINT_EXTRA_CLASS]
        if "win" in name:
            assert extra_legal.all(), f"{name} did not open the not-win class"
            assert losses[name] > 1e-3, f"{name} loss {losses[name]} collapsed to a plays head"
        else:
            assert not extra_legal.any(), f"{name} wrongly opened the not-win class"
            assert losses[name] < 1e-5, f"{name} loss {losses[name]} is not ~0"


def test_mask_placement_changes_the_loss():
    """Masking illegal footprints reshapes the softmax normalizer, so the masked
    and unmasked placement losses differ for every head -- the masked-vs-unmasked
    arm that keeps masking from confounding the loss-geometry result is a real,
    per-head toggle, not one wired for a single head."""
    torch.manual_seed(0)
    model = _model()
    batch = _batch()
    out = model(batch["input_spatial"], batch["input_scalar"])
    targets = {k: batch[k] for k in _TARGET_KEYS}
    masked = model.compute_loss(out, targets, mask_placement=True)
    unmasked = model.compute_loss(out, targets, mask_placement=False)
    for name in PLACEMENT_HEAD_NAMES:
        assert not torch.isclose(masked[name], unmasked[name]), f"{name} masking is inert"


def test_compute_loss_matches_reference_math():
    """Every head's loss and the weighted total must equal a hand-written
    reference computed straight from the loss formulas, so a change to one head's
    loss -- a swapped Huber delta, a flipped win/plays weight, a dropped detach --
    can only move that head's number, never silently the whole objective. The
    weights and Huber deltas below are all DISTINCT so that a mean-vs-std delta or
    win-vs-plays weight mix-up actually changes the result the reference checks."""
    torch.manual_seed(0)
    model = _model()
    batch = _batch()
    out = model(batch["input_spatial"], batch["input_scalar"])
    targets = {k: batch[k] for k in _TARGET_KEYS}
    lambda_wld, lambda_sd = 1.0, 0.004
    lambda_next, lambda_win = 0.3, 0.7
    delta_mean, delta_std = 8.0, 12.0
    losses = model.compute_loss(
        out,
        targets,
        lambda_wld=lambda_wld,
        lambda_sd=lambda_sd,
        lambda_next_placement=lambda_next,
        lambda_win_placement=lambda_win,
        huber_delta_mean=delta_mean,
        huber_delta_std=delta_std,
        mask_placement=True,
    )

    # The reference: the loss written out by hand, head by head.
    ref = {"wld": F.cross_entropy(out["wld"], targets["wld"].argmax(dim=1))}
    sd_mean, sd_std = out["score_diff"][:, 0], out["score_diff"][:, 1]
    sd_target = targets["score_diff"].squeeze(1)
    loss_mean = F.huber_loss(sd_mean, sd_target, delta=delta_mean)
    loss_std = F.huber_loss(
        sd_std, (sd_mean.detach() - sd_target).abs() * MAD_TO_STD, delta=delta_std
    )
    ref["score_diff_mean"], ref["score_diff_std"] = loss_mean, loss_std
    ref["score_diff"] = loss_mean + loss_std
    for name in PLACEMENT_HEAD_NAMES:
        ref[name] = _placement_ce(
            out[name], targets[name].squeeze(1).long(), _head_legal_mask(name, targets)
        )
    total = (
        lambda_wld * ref["wld"]
        + lambda_sd * ref["score_diff"]
        + lambda_next * (ref["opp_next_placement"] + ref["self_next_placement"])
        + lambda_win * (ref["opp_win_placement"] + ref["self_win_placement"])
    )

    for key, value in ref.items():
        assert torch.allclose(losses[key], value), f"{key} diverged from the reference"
    assert torch.allclose(losses["total"], total)


def test_lambda_wld_scales_the_wld_term_out_of_the_total():
    """lambda_wld weights the WLD loss in the total (the diagnostic that isolates
    the other heads sets it to 0): total(lambda_wld=0) == total(lambda_wld=1)
    minus exactly the wld term, and the per-head wld loss itself is unweighted."""
    torch.manual_seed(0)
    model = _model()
    batch = _batch()
    out = model(batch["input_spatial"], batch["input_scalar"])
    targets = {k: batch[k] for k in _TARGET_KEYS}
    on = model.compute_loss(out, targets, lambda_wld=1.0)
    off = model.compute_loss(out, targets, lambda_wld=0.0)
    assert torch.isclose(off["total"], on["total"] - on["wld"])
    assert torch.equal(off["wld"], on["wld"])  # the reported head loss is unweighted

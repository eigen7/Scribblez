"""Unit tests for the shared position evaluation training-epoch loop.

A tiny stub model producing every position evaluation head over small inputs
exercises run_epoch's forward/backward/accumulate and its learning-rate
handling without the real trunk or the C++ data layout.
"""

import torch
from scribblez.position_eval.model import MASK_HEAD_NAMES, compute_loss
from scribblez.position_eval.train_loop import EpochResult, LossConfig, run_epoch

_LOSS_CFG = LossConfig(
    lambda_wld=1.0,
    lambda_sd=0.004,
    lambda_next_placement=0.5,
    lambda_win_placement=0.5,
    huber_delta_mean=10.0,
    huber_delta_std=10.0,
)
_CPU = torch.device("cpu")


class _StubModel(torch.nn.Module):
    """Minimal model producing every position evaluation head from the scalar input."""

    def __init__(self, scalar_size: int = 8):
        super().__init__()
        self.wld = torch.nn.Linear(scalar_size, 3)
        self.score = torch.nn.Linear(scalar_size, 2)
        self.masks = torch.nn.ModuleDict(
            {name: torch.nn.Linear(scalar_size, 15 * 15) for name in MASK_HEAD_NAMES}
        )

    def forward(self, input_spatial, input_scalar):
        out = {
            "wld": self.wld(input_scalar),
            "score_diff": self.score(input_scalar),
        }
        for name, fc in self.masks.items():
            out[name] = fc(input_scalar).view(-1, 15, 15)
        return out


def _batch(bs: int = 4, scalar_size: int = 8) -> dict:
    wld = torch.zeros(bs, 3)
    wld[torch.arange(bs), torch.randint(0, 3, (bs,))] = 1.0
    return {
        "input_spatial": torch.randn(bs, 2, 15, 15),
        "input_scalar": torch.randn(bs, scalar_size),
        "wld": wld,
        "score_diff": torch.randn(bs, 1) * 20,
        **{name: (torch.rand(bs, 15, 15) > 0.85).float() for name in MASK_HEAD_NAMES},
    }


def test_run_epoch_accumulates_and_counts():
    torch.manual_seed(0)
    model = _StubModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    result = run_epoch(model, opt, [_batch() for _ in range(3)], _CPU, _LOSS_CFG, rows_trained=100)
    assert isinstance(result, EpochResult)
    assert result.n_batches == 3
    assert result.samples == 12
    assert result.rows_trained == 112  # 100 + 3 * 4
    assert 0.0 <= result.wld_acc <= 1.0
    assert set(result.losses) >= {"total", "wld", "score_diff", *MASK_HEAD_NAMES}
    assert result.losses["total"] > 0.0


def test_run_epoch_takes_a_gradient_step():
    torch.manual_seed(0)
    model = _StubModel()
    before = model.wld.weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    run_epoch(model, opt, [_batch()], _CPU, _LOSS_CFG)
    assert not torch.equal(before, model.wld.weight)


def test_run_epoch_applies_lr_fn_per_step():
    model = _StubModel()
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
    model = _StubModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.123)
    run_epoch(model, opt, [_batch()], _CPU, _LOSS_CFG)
    assert opt.param_groups[0]["lr"] == 0.123


def test_lambda_wld_scales_the_wld_term_out_of_the_total():
    """lambda_wld weights the WLD loss in the total (the diagnostic that isolates
    the other heads sets it to 0): total(lambda_wld=0) == total(lambda_wld=1)
    minus exactly the wld term, and the per-head wld loss itself is unweighted."""
    torch.manual_seed(0)
    model = _StubModel()
    batch = _batch()
    out = model(batch["input_spatial"], batch["input_scalar"])
    targets = {k: batch[k] for k in ["wld", "score_diff", *MASK_HEAD_NAMES]}
    on = compute_loss(out, targets, lambda_wld=1.0)
    off = compute_loss(out, targets, lambda_wld=0.0)
    assert torch.isclose(off["total"], on["total"] - on["wld"])
    assert torch.equal(off["wld"], on["wld"])  # the reported head loss is unweighted

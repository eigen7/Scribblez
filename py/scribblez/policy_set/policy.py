"""Form 3 of the policy-set model, and the search network's empty-store pass
(docs/plans/deep_bot.md, "The search network"): a policy over the position's
candidates and a value per candidate.

Each candidate's vector (model.PolicySetEncoder) gets one logit, and a softmax
over the position's candidates, however many it has, makes the policy. The
loss is its cross-entropy against the pool's q, P(the move is the sim's
best), and its set of N is its top N. The events "a is the best move" are
disjoint, so q summed over a set is the chance it holds the best move: a
policy that matched q would make the best set of N for recall and cover at
every N.

The value head predicts, per candidate, the evaluator's win probability
after it as the sims measure it: the mean, and the variance of one rollout's
value about it, the spread a sim's residual is read against. Both fit by
moments (deep_bot_leaf_eval.model.moment_losses): the mean against the pool's
win probability, the variance against the rollouts' squared deviations from
the predicted mean. In VALUE_SCALE units of win probability, so the losses sit
near 1 rather than 1e-4.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from scribblez.policy_set.model import PolicySetEncoder, mean_spread, to_grid

# The loss's unit: a tenth of a win probability.
VALUE_SCALE = 10.0


class PolicyOutputs(NamedTuple):
    logits: torch.Tensor  # (P, maxK) FP32, -inf at padding
    pad: torch.Tensor  # (P, maxK) True where there is no candidate
    value: torch.Tensor  # (M, 2) FP32: [mean, variance] in win-probability units


class PolicyModel(nn.Module):
    """Form 3: the encoder, a linear logit and a linear value per candidate."""

    def __init__(self, encoder: PolicySetEncoder):
        super().__init__()
        self.encoder = encoder
        channels = encoder.board_pos_emb.shape[1]
        self.logit = nn.Linear(channels, 1)
        self.value = nn.Linear(channels, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> PolicyOutputs:
        h, _ = self.encoder(batch)
        num_positions = batch["input_spatial"].shape[0]
        logits, pad = to_grid(
            self.logit(h).squeeze(1).float(), batch["move_pos_id"], num_positions, -torch.inf
        )
        return PolicyOutputs(logits, pad, value_moments(self.value(h).float()))


def value_moments(raw: torch.Tensor) -> torch.Tensor:
    """(M, 2) raw head output -> [mean, variance] in win-probability units:
    the mean a sigmoid, the variance a softplus in VALUE_SCALE units (so that
    the head starts near the rollouts' spread rather than far above it),
    scaled back. The export (onnx_export.py) applies the same."""
    var = F.softplus(raw[:, 1]) / VALUE_SCALE**2
    return torch.stack([torch.sigmoid(raw[:, 0]), var], dim=1)


def policy_forward(model: PolicyModel, batch: dict[str, torch.Tensor]) -> PolicyOutputs:
    """The model's outputs, under autocast."""
    with torch.autocast(device_type=batch["input_spatial"].device.type, dtype=torch.bfloat16):
        return model(batch)


def cross_entropies(out: PolicyOutputs, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Each position's -sum_a q(a) log pi(a): (P,)."""
    q, _ = to_grid(batch["q"].float(), batch["move_pos_id"], out.logits.shape[0])
    return -torch.where(out.pad, 0.0, q * F.log_softmax(out.logits, dim=1)).sum(dim=1)


def rollout_deviations(mean: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """(M,): each candidate's mean squared deviation of its rollouts' values
    (dataset.rollout_grid) from `mean` (M,), in win-probability units."""
    values, n = batch["rollout_values"].float(), batch["rollouts"]
    ran = torch.arange(values.shape[1], device=values.device)[None, :] < n[:, None]
    dev2 = torch.where(ran, (values - mean[:, None]) ** 2, 0.0).sum(dim=1)
    return dev2 / n.clamp(min=1).to(dev2.dtype)


def value_losses(out: PolicyOutputs, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The value head's moment losses over the batch's candidates, keyed
    "value_mean" and "value_var": the mean against the pool's win probability,
    the variance against the rollouts' squared deviations from the predicted
    mean (detached), both in VALUE_SCALE units."""
    mean, var = out.value.unbind(dim=1)
    dev2 = rollout_deviations(mean.detach(), batch)
    return {
        "value_mean": (((mean - batch["win_prob"].float()) * VALUE_SCALE) ** 2).mean(),
        "value_var": (((var - dev2) * VALUE_SCALE**2) ** 2).mean(),
    }


def policy_loss(
    model: PolicyModel, batch: dict[str, torch.Tensor], weighted: bool = False
) -> torch.Tensor:
    """The mean over positions of the cross-entropy against q; `weighted`,
    each position's times its stakes (pools.stakes), so a decided game, where
    the move barely matters, barely trains. Not renormalized: Adam's updates
    are nearly invariant to the loss's scale, and a per-batch renormalization
    would undo the weighting for a batch of decided games."""
    return weighted_cross_entropy(
        cross_entropies(policy_forward(model, batch), batch), batch, weighted
    )


def weighted_cross_entropy(ce: torch.Tensor, batch: dict[str, torch.Tensor], weighted: bool):
    return (batch["stakes"] * ce).mean() if weighted else ce.mean()


def form_losses(
    model: PolicyModel, batch: dict[str, torch.Tensor], weighted: bool, lambda_value: float
) -> dict[str, torch.Tensor]:
    """One forward: the policy loss ("ce") and the value head's losses,
    weighted by `lambda_value`."""
    out = policy_forward(model, batch)
    losses = {"ce": weighted_cross_entropy(cross_entropies(out, batch), batch, weighted)}
    losses |= {key: lambda_value * v for key, v in value_losses(out, batch).items()}
    return losses


@torch.no_grad()
def policy_picks(model: PolicyModel, batch: dict[str, torch.Tensor], n: int) -> list[list[int]]:
    """Each position's top `n` candidates by logit, best first, as candidate
    indices within the position."""
    out = policy_forward(model, batch)
    sizes = (~out.pad).sum(dim=1).tolist()
    order = torch.argsort(out.logits, dim=1, descending=True).tolist()
    return [o[: min(n, k)] for o, k in zip(order, sizes, strict=True)]


@torch.no_grad()
def policy_diagnostics(
    model: PolicyModel, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The logits' spread over each position's candidates, the policy's mean
    entropy (nats), to read against the labels' (metrics.pool_stats), the
    cross-entropy against q, plain and in the stakes' weighted mean (sum s*ce
    / sum s), both reading the same with the loss weighted or not; and the
    value head's: its mean absolute error against the pools' win
    probabilities, and its calibration, the predicted variance over the
    rollouts' realized squared deviation about the pools' win probability,
    1 when calibrated."""
    out = policy_forward(model, batch)
    logp = F.log_softmax(out.logits, dim=1)
    entropy = -torch.where(out.pad, 0.0, logp.exp() * logp).sum(dim=1).mean()
    s, ce = batch["stakes"], cross_entropies(out, batch)
    mean, var = out.value.unbind(dim=1)
    win = batch["win_prob"].float()
    return {
        "spread": mean_spread(out.logits, out.pad),
        "entropy": entropy,
        "ce": ce.mean(),
        "stakes_ce": (s * ce).sum() / s.sum().clamp(min=1e-9),
        "value_mae": (mean - win).abs().mean(),
        "value_var_ratio": var.mean() / rollout_deviations(win, batch).mean().clamp(min=1e-9),
    }

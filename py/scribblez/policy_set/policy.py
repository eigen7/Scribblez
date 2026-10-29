"""Form 3 of the policy-set model: a policy over the position's candidates.
Each candidate's vector (model.PolicySetEncoder) gets one logit, and a softmax
over the position's candidates, however many it has, makes the policy. The
loss is its cross-entropy against the pool's q, P(the move is the sim's best),
and its set of N is its top N. The events "a is the best move" are disjoint,
so q summed over a set is the chance it holds the best move: a policy that
matched q would make the best set of N for recall and cover at every N.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from scribblez.policy_set.model import PolicySetEncoder, mean_spread, to_grid


class PolicyModel(nn.Module):
    """Form 3: the encoder and a linear logit per candidate."""

    def __init__(self, encoder: PolicySetEncoder):
        super().__init__()
        self.encoder = encoder
        self.logit = nn.Linear(encoder.board_pos_emb.shape[1], 1)

    def forward(self, batch: dict[str, torch.Tensor]):
        """Logits (P, maxK) in FP32, -inf at padding, and the padding mask
        (P, maxK)."""
        h, _ = self.encoder(batch)
        num_positions = batch["input_spatial"].shape[0]
        return to_grid(
            self.logit(h).squeeze(1).float(), batch["move_pos_id"], num_positions, -torch.inf
        )


def policy_forward(model: PolicyModel, batch: dict[str, torch.Tensor]):
    """The policy's logits and padding mask, under autocast."""
    with torch.autocast(device_type=batch["input_spatial"].device.type, dtype=torch.bfloat16):
        return model(batch)


def policy_loss(model: PolicyModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The mean over positions of -sum_a q(a) log pi(a)."""
    logits, pad = policy_forward(model, batch)
    q, _ = to_grid(batch["q"].float(), batch["move_pos_id"], logits.shape[0])
    return -torch.where(pad, 0.0, q * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@torch.no_grad()
def policy_picks(model: PolicyModel, batch: dict[str, torch.Tensor], n: int) -> list[list[int]]:
    """Each position's top `n` candidates by logit, best first, as candidate
    indices within the position."""
    logits, pad = policy_forward(model, batch)
    sizes = (~pad).sum(dim=1).tolist()
    order = torch.argsort(logits, dim=1, descending=True).tolist()
    return [o[: min(n, k)] for o, k in zip(order, sizes, strict=True)]


@torch.no_grad()
def policy_diagnostics(
    model: PolicyModel, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The logits' spread over each position's candidates, and the policy's
    mean entropy (nats), to read against the labels' (metrics.pool_stats)."""
    logits, pad = policy_forward(model, batch)
    logp = F.log_softmax(logits, dim=1)
    entropy = -torch.where(pad, 0.0, logp.exp() * logp).sum(dim=1).mean()
    return {"spread": mean_spread(logits, pad), "entropy": entropy}

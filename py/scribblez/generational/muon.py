"""Muon (Jordan et al., 2024) for hidden weight matrices, with AdamW for
every other parameter, as one torch optimizer.

Muon replaces a matrix's update with the semi-orthogonal matrix nearest its
Nesterov momentum, computed by a quintic Newton-Schulz iteration. Every
direction in the matrix's row and column spaces then moves at the same rate,
instead of the few dominant directions a raw gradient step favours. It is meant
for hidden matrices only; embeddings, output heads, norms and biases stay on
AdamW.

This is a translation of the single-device MuonWithAuxAdam in the reference
implementation (github.com/KellerJordan/Muon), with one change: the Muon update
is scaled by 0.2 * sqrt(max(rows, cols)) instead of sqrt(max(1, rows / cols)).
That scale (Liu et al., 2025, "Muon is Scalable for LLM Training") matches the
update's RMS to AdamW's, so the Muon and AdamW groups share one learning rate
and one weight decay -- the rate a trainer's schedule writes to every group.

The state dict is torch's ordinary one, so the rolling checkpoint saves and
resumes it like any other optimizer's.
"""

from __future__ import annotations

import torch

# The reference implementation's quintic coefficients and iteration count.
# They trade exactness for speed: the singular values land around [0.7, 1.2]
# rather than at 1, which training does not notice.
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
# The RMS-matching scale factor from Liu et al.
RMS_MATCH = 0.2


def orthogonalize(g: torch.Tensor) -> torch.Tensor:
    """Approximately U V^T for the SVD g = U S V^T of the matrix `g` -- or of
    each matrix in a (..., rows, cols) stack -- in bf16. The iteration runs on
    the wide orientation, where X X^T is the smaller product."""
    a, b, c = NS_COEFFICIENTS
    tall = g.size(-2) > g.size(-1)
    x = g.bfloat16()
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # spectral <= Frobenius <= 1
    for _ in range(NS_STEPS):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if tall else x


def _by_shape(params: list[torch.Tensor]) -> list[list[torch.Tensor]]:
    """`params` partitioned by shape, each part in its original order."""
    parts: dict[torch.Size, list[torch.Tensor]] = {}
    for p in params:
        parts.setdefault(p.shape, []).append(p)
    return list(parts.values())


class MuonAdamW(torch.optim.Optimizer):
    """Muon on the groups flagged `"muon": True`, AdamW on the rest.

    Every group takes `lr` and its own `weight_decay` (decoupled, as in
    AdamW). Muon groups also take `momentum` (Nesterov); AdamW groups take
    `betas` and `eps`."""

    def __init__(
        self,
        param_groups: list[dict],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.95,
        eps: float = 1e-8,
    ):
        for group in param_groups:
            if group["muon"] and any(p.ndim != 2 for p in group["params"]):
                raise ValueError("Muon groups may hold only 2-D weight matrices")
        defaults = {"lr": lr, "betas": betas, "momentum": momentum, "eps": eps, "weight_decay": 0}
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            if group["muon"]:
                for same_shape in _by_shape(params):
                    self._muon_update(same_shape, group)
            else:
                for p in params:
                    self._adamw_update(p, group)

    def _muon_update(self, params: list[torch.Tensor], group: dict):
        """One Muon step for same-shaped matrices, orthogonalized as one stack:
        a model has only a few distinct matrix shapes, so this launches a few
        batched iterations instead of one per matrix."""
        nesterov = []
        for p in params:
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)
            momentum = state["momentum_buffer"]
            momentum.lerp_(p.grad, 1 - group["momentum"])
            nesterov.append(p.grad.lerp(momentum, group["momentum"]))
        updates = orthogonalize(torch.stack(nesterov))
        scale = RMS_MATCH * max(params[0].shape) ** 0.5
        for p, update in zip(params, updates, strict=True):
            p.add_(update, alpha=-group["lr"] * scale)

    def _adamw_update(self, p: torch.Tensor, group: dict):
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        state["step"] += 1
        beta1, beta2 = group["betas"]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.lerp_(p.grad, 1 - beta1)
        exp_avg_sq.lerp_(p.grad.square(), 1 - beta2)
        denom = (exp_avg_sq / (1 - beta2 ** state["step"])).sqrt_().add_(group["eps"])
        p.addcdiv_(exp_avg, denom, value=-group["lr"] / (1 - beta1 ** state["step"]))

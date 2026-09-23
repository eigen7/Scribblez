"""Training-epoch loop for the move set evaluation model, the counterpart of
position_eval/train_loop.

Batches hold varying numbers of candidate moves and each batch loss is a mean
over its moves, so the epoch averages are weighted by candidate count.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from .model import INPUT_KEYS, MOVE_KEYS, compute_loss

# compute_loss keys averaged each epoch; "total" is the optimized objective.
LOSS_KEYS = (
    "total",
    "wld",
    "score_diff",
    "score_diff_mean",
    "score_diff_std",
    "planes",
)
# target_planes is present only on a plane-carrying corpus (dataset.has_planes).
TARGET_KEYS = ("target_wld", "target_score_diff", "target_planes")


@dataclass
class LossConfig:
    """compute_loss weights and Huber transition points."""

    lambda_sd: float
    huber_delta_mean: float
    huber_delta_std: float
    lambda_planes: float

    @classmethod
    def from_args(cls, args) -> LossConfig:
        return cls(
            args.lambda_sd,
            args.huber_delta_mean,
            args.huber_delta_std,
            args.lambda_planes,
        )

    def loss(self, outputs: dict, targets: dict) -> dict:
        return compute_loss(
            outputs,
            targets,
            lambda_sd=self.lambda_sd,
            huber_delta_mean=self.huber_delta_mean,
            huber_delta_std=self.huber_delta_std,
            lambda_planes=self.lambda_planes,
        )


@dataclass
class EpochResult:
    """Candidate-weighted averages over one epoch, plus the advanced counters."""

    losses: dict[str, float]
    n_batches: int
    candidates: int  # candidate moves seen this epoch
    rows_trained: int  # cumulative candidate moves across the run


def _forward_args(batch: dict, device):
    inputs = tuple(batch[k].to(device) for k in INPUT_KEYS)
    move_args = tuple(batch[k].to(device) for k in MOVE_KEYS)
    targets = {k: batch[k].to(device) for k in TARGET_KEYS if k in batch}
    return inputs, move_args, targets


def batch_loss(model, batch: dict, device, loss_cfg: LossConfig) -> dict:
    """Plain (evidence-free) forward and distillation loss for one batch,
    as compute_loss' dict."""
    inputs, move_args, targets = _forward_args(batch, device)
    return loss_cfg.loss(model(*inputs, *move_args), targets)


def run_epoch(
    model,
    optimizer,
    batches: Iterable[dict],
    device,
    loss_cfg: LossConfig,
    *,
    lr_fn: Callable[[int], float] | None = None,
    rows_trained: int = 0,
    on_batch: Callable[[int, int, float, int], None] | None = None,
    grad_clip: float = 0.0,
) -> EpochResult:
    """Run one training pass over `batches`, in the order given.

    rows_trained: the run's cumulative candidate count at the start; the
        result carries it forward. lr_fn, if given, maps it to the learning
        rate before each step.
    on_batch: progress callback (done_batches, candidates, elapsed_s,
        rows_trained), called at most about once per second.
    grad_clip: global gradient-norm clip; 0 disables.
    """
    model.train()
    sums = {k: 0.0 for k in LOSS_KEYS}
    weight_sum = 0
    n_batches = 0
    candidates = 0
    t0 = time.time()
    last_progress = 0.0

    for batch in batches:
        if lr_fn is not None:
            lr = lr_fn(rows_trained)
            for group in optimizer.param_groups:
                group["lr"] = lr

        losses = batch_loss(model, batch, device, loss_cfg)
        optimizer.zero_grad()
        losses["total"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        m = batch["target_wld"].shape[0]
        n_batches += 1
        candidates += m
        weight_sum += m
        rows_trained += m
        for k in sums:
            sums[k] += losses[k].item() * m

        if on_batch is not None and time.time() - last_progress > 1.0:
            on_batch(n_batches, candidates, time.time() - t0, rows_trained)
            last_progress = time.time()

    return EpochResult(
        losses={k: v / max(weight_sum, 1) for k, v in sums.items()},
        n_batches=n_batches,
        candidates=candidates,
        rows_trained=rows_trained,
    )

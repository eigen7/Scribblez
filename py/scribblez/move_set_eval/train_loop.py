"""Training-epoch loop for the move set evaluation model.

A sibling to position_eval/train_loop: one pass over the flattened candidate
batches, moving each to the device, forward, combined-loss backward, optimizer
step, and accumulating the per-head losses. The per-batch loss is a mean over
that batch's candidate moves, so the epoch averages are weighted by candidate
count (batches hold variable move totals).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .model import compute_loss

LOSS_KEYS = ("total", "wld", "score_diff", "score_diff_mean", "score_diff_std", "planes")

# Tensors in the batch dict, split into board inputs, move inputs, and targets.
_INPUT_KEYS = ("input_spatial", "input_scalar")
_MOVE_KEYS = (
    "move_letters",
    "move_blanks",
    "move_squares",
    "move_tile_mask",
    "move_scalars",
    "move_pos_id",
)
# target_planes is present only on a plane-carrying corpus (dataset.has_planes).
TARGET_KEYS = ("target_wld", "target_score_diff", "target_planes")


@dataclass
class LossConfig:
    """Weight and Huber transition points for the combined pre-move loss."""

    lambda_sd: float
    huber_delta_mean: float
    huber_delta_std: float
    lambda_planes: float

    @classmethod
    def from_args(cls, args) -> LossConfig:
        return cls(args.lambda_sd, args.huber_delta_mean, args.huber_delta_std, args.lambda_planes)

    def loss(self, outputs: dict, targets: dict) -> dict:
        """compute_loss under this config."""
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
    inputs = tuple(batch[k].to(device) for k in _INPUT_KEYS)
    move_args = tuple(batch[k].to(device) for k in _MOVE_KEYS)
    targets = {k: batch[k].to(device) for k in TARGET_KEYS if k in batch}
    return inputs, move_args, targets


def batch_loss(model, batch: dict, device, loss_cfg: LossConfig) -> dict:
    """The plain forward over one batch and its distillation loss (compute_loss'
    dict): the step of this loop, and the distillation half of the evidence
    trainer's joint (unfrozen-backbone) step."""
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
) -> EpochResult:
    """Run one training pass over `batches` (already ordered by the caller).

    rows_trained: starting cumulative candidate count; carried forward in the
        result. lr_fn, if given, sets every param group's LR from that count
        before each step (the rows-clock learning rate).
    on_batch: optional progress callback (done_batches, candidates, elapsed_s,
        rows_trained), invoked at most ~once per second.
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

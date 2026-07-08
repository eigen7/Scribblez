"""Shared training-epoch loop for the max-move-per-lane model.

The generational orchestrator drives per-minibatch training through run_epoch:
move a batch to the device, forward, combined-lane-loss backward, optimizer step,
and accumulate the per-component losses plus the per-lane train accuracies.
Keeping the step here (a sibling to the position-evaluation trainer's train_loop)
means the orchestrator owns only its data lifecycle, learning-rate policy,
evaluation, and checkpointing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from .model import compute_loss

# Per-component loss keys accumulated each epoch. compute_loss returns all of
# these; "total" is the optimized objective.
LOSS_KEYS = ("total", "score_pdf", "score_cdf", "move", "has_move")

# The three lane targets the model is trained against.
TARGET_KEYS = ("lane_occupancy", "lane_score", "lane_mask")


@dataclass
class LossConfig:
    """Weights for the combined max-move-per-lane loss (the score-PDF term has
    weight 1)."""

    lambda_cdf: float
    lambda_occ: float
    lambda_has_move: float

    @classmethod
    def from_args(cls, args) -> LossConfig:
        return cls(args.lambda_cdf, args.lambda_occ, args.lambda_has_move)


@dataclass
class EpochResult:
    """Averages over one epoch's minibatches, plus the advanced rows counter."""

    losses: dict[str, float]  # per-component means, including "total"
    accs: dict[str, float]  # score_acc / move_acc / has_move_acc means
    n_batches: int
    samples: int
    rows_trained: int


def lane_accuracy(outputs: dict, targets: dict) -> dict:
    """Per-legal-lane train accuracy: does the model get each lane's best move and
    score right? Averaged over the lanes that actually have a legal move (plus a
    has-move accuracy over all 30 lanes)."""
    mask = targets["lane_mask"]  # (B, 30)
    legal = mask.sum().clamp_min(1.0)

    pred_bin = outputs["lane_score_logits"].argmax(-1)  # (B, 30)
    score_ok = (((pred_bin == targets["lane_score"].long()).float() * mask).sum() / legal).item()

    # "Move right" == the thresholded occupancy union matches the target exactly.
    pred_occ = (outputs["lane_occupancy_logits"] > 0).float()  # (B, 30, 15, 27)
    lane_match = (pred_occ == targets["lane_occupancy"]).all(dim=-1).all(dim=-1).float()  # (B, 30)
    move_ok = ((lane_match * mask).sum() / legal).item()

    has_pred = (outputs["lane_has_move_logits"] > 0).float()
    has_move_ok = (has_pred == mask).float().mean().item()
    return {"score_acc": score_ok, "move_acc": move_ok, "has_move_acc": has_move_ok}


def _to_device(batch: dict, device):
    """Split a batch dict into (spatial, scalar) inputs and the three lane target
    tensors, each moved to `device`."""
    inputs = (batch["input_spatial"].to(device), batch["input_scalar"].to(device))
    targets = {k: batch[k].to(device) for k in TARGET_KEYS}
    return inputs, targets


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
    """Run one training pass over `batches` (already seeded/ordered by the caller).

    lr_fn: if given, called per step with the running rows count; its result is
        written to every optimizer param group before the step (the generational
        rows-clock learning rate). When None the caller owns the learning rate.
    rows_trained: starting cumulative row (position) count; the return value
        carries it forward across epochs and generations.
    on_batch: optional progress callback (done_batches, samples, elapsed_s,
        rows_trained), invoked at most ~once per second.
    """
    model.train()
    loss_sums = {k: 0.0 for k in LOSS_KEYS}
    acc_sums = {"score_acc": 0.0, "move_acc": 0.0, "has_move_acc": 0.0}
    n_batches = 0
    samples = 0
    t0 = time.time()
    last_progress = 0.0

    for batch in batches:
        (input_spatial, input_scalar), targets = _to_device(batch, device)
        if lr_fn is not None:
            lr = lr_fn(rows_trained)
            for group in optimizer.param_groups:
                group["lr"] = lr

        outputs = model(input_spatial, input_scalar)
        losses = compute_loss(
            outputs,
            targets,
            lambda_cdf=loss_cfg.lambda_cdf,
            lambda_occ=loss_cfg.lambda_occ,
            lambda_has_move=loss_cfg.lambda_has_move,
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        bs = input_spatial.shape[0]
        n_batches += 1
        samples += bs
        rows_trained += bs
        for k in loss_sums:
            loss_sums[k] += losses[k].item()
        with torch.no_grad():
            accs = lane_accuracy(outputs, targets)
        for k in acc_sums:
            acc_sums[k] += accs[k]

        if on_batch is not None and time.time() - last_progress > 1.0:
            on_batch(n_batches, samples, time.time() - t0, rows_trained)
            last_progress = time.time()

    nb = max(n_batches, 1)
    return EpochResult(
        losses={k: v / nb for k, v in loss_sums.items()},
        accs={k: v / nb for k, v in acc_sums.items()},
        n_batches=n_batches,
        samples=samples,
        rows_trained=rows_trained,
    )

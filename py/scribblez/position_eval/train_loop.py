"""Training-epoch loop for the position evaluation model.

The generational trainer (scripts/position_eval/train.py) drives its
per-minibatch step through run_epoch: move a batch to the device, forward,
combined-loss backward, optimizer step, and accumulate the per-head losses plus
WLD accuracy. Keeping the step here (a sibling to the max-move-per-lane
train_loop) isolates the gradient update from the orchestrator, which owns only
the data lifecycle, learning-rate policy, evaluation, and checkpointing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .model import compute_loss

# Per-head loss keys accumulated each epoch. compute_loss returns all of these;
# "total" is the optimized objective.
LOSS_KEYS = (
    "total",
    "wld",
    "score_diff",
    "score_diff_mean",
    "score_diff_std",
    "opp_next_placement",
    "self_next_placement",
    "opp_win_placement",
    "self_win_placement",
)


@dataclass
class LossConfig:
    """Weights and Huber transition points for the combined post-move loss."""

    lambda_wld: float
    lambda_sd: float
    lambda_next_placement: float
    lambda_win_placement: float
    huber_delta_mean: float
    huber_delta_std: float
    placement_pos_weight: float

    @classmethod
    def from_args(cls, args) -> LossConfig:
        return cls(
            args.lambda_wld,
            args.lambda_sd,
            args.lambda_next_placement,
            args.lambda_win_placement,
            args.huber_delta_mean,
            args.huber_delta_std,
            args.placement_pos_weight,
        )


@dataclass
class EpochResult:
    """Averages over one epoch's minibatches, plus the advanced rows counter."""

    losses: dict[str, float]  # per-head means, including "total"
    wld_acc: float
    n_batches: int
    samples: int
    rows_trained: int


# Target tensors compute_loss consumes, pulled from the batch dict by name.
TARGET_KEYS = (
    "wld",
    "score_diff",
    "opp_next_placement",
    "self_next_placement",
    "opp_win_placement",
    "self_win_placement",
)


def _to_device(batch: dict, device):
    """Split a batch dict into (spatial, scalar) inputs and the target
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
        rows-clock learning rate). When None the caller owns the learning rate
        (e.g. a per-epoch scheduler) and this loop leaves it untouched.
    rows_trained: starting cumulative row (position) count; the return value
        carries it forward across epochs and generations.
    on_batch: optional progress callback (done_batches, samples, elapsed_s,
        rows_trained), invoked at most ~once per second.
    """
    model.train()
    sums = {k: 0.0 for k in LOSS_KEYS}
    n_batches = 0
    correct = 0
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
            lambda_wld=loss_cfg.lambda_wld,
            lambda_sd=loss_cfg.lambda_sd,
            lambda_next_placement=loss_cfg.lambda_next_placement,
            lambda_win_placement=loss_cfg.lambda_win_placement,
            huber_delta_mean=loss_cfg.huber_delta_mean,
            huber_delta_std=loss_cfg.huber_delta_std,
            placement_pos_weight=loss_cfg.placement_pos_weight,
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        bs = input_spatial.shape[0]
        n_batches += 1
        samples += bs
        rows_trained += bs
        for k in sums:
            sums[k] += losses[k].item()
        correct += (outputs["wld"].argmax(1) == targets["wld"].argmax(1)).sum().item()

        if on_batch is not None and time.time() - last_progress > 1.0:
            on_batch(n_batches, samples, time.time() - t0, rows_trained)
            last_progress = time.time()

    return EpochResult(
        losses={k: v / max(n_batches, 1) for k, v in sums.items()},
        wld_acc=correct / max(samples, 1),
        n_batches=n_batches,
        samples=samples,
        rows_trained=rows_trained,
    )

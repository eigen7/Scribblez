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

# The loss config and the loss itself live with the head registry in model.py:
# the model owns compute_loss(), and the per-head loss keys and batch target keys
# are derived from its heads (model.loss_keys() / model.target_keys()), so a new
# head extends them without touching this loop. LossConfig is re-exported for
# callers that import it from the train loop.
from .model import LossConfig

__all__ = ["LossConfig", "EpochResult", "run_epoch"]


@dataclass
class EpochResult:
    """Averages over one epoch's minibatches, plus the advanced rows counter."""

    losses: dict[str, float]  # per-head means, including "total"
    wld_acc: float
    n_batches: int
    samples: int
    rows_trained: int


def _to_device(batch: dict, device, target_keys: tuple[str, ...]):
    """Split a batch dict into (spatial, scalar) inputs and the `target_keys`
    target tensors, each moved to `device`."""
    inputs = (batch["input_spatial"].to(device), batch["input_scalar"].to(device))
    targets = {k: batch[k].to(device) for k in target_keys}
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
    target_keys = model.target_keys()
    sums = {k: 0.0 for k in model.loss_keys()}
    n_batches = 0
    correct = 0
    samples = 0
    t0 = time.time()
    last_progress = 0.0

    for batch in batches:
        (input_spatial, input_scalar), targets = _to_device(batch, device, target_keys)
        if lr_fn is not None:
            lr = lr_fn(rows_trained)
            for group in optimizer.param_groups:
                group["lr"] = lr

        outputs = model(input_spatial, input_scalar)
        losses = model.compute_loss(
            outputs,
            targets,
            lambda_wld=loss_cfg.lambda_wld,
            lambda_sd=loss_cfg.lambda_sd,
            lambda_next_placement=loss_cfg.lambda_next_placement,
            lambda_win_placement=loss_cfg.lambda_win_placement,
            huber_delta_mean=loss_cfg.huber_delta_mean,
            huber_delta_std=loss_cfg.huber_delta_std,
            mask_placement=loss_cfg.mask_placement,
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

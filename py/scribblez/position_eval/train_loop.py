"""One training epoch of the position evaluation model.

The gradient step lives here, apart from the trainer (trainer.py), which owns
the data lifecycle, learning-rate policy, evaluation and checkpointing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

# Re-exported: callers import LossConfig from here. The loss itself, and the
# loss and target keys this loop iterates, come from the model's head registry.
from .model import LossConfig

__all__ = ["LossConfig", "EpochResult", "run_epoch"]


@dataclass
class EpochResult:
    """Per-epoch averages, plus the updated cumulative rows counter."""

    losses: dict[str, float]  # per-head means, including "total"
    wld_acc: float
    n_batches: int
    samples: int
    rows_trained: int


def _to_device(batch: dict, device, target_keys: tuple[str, ...]):
    """((spatial, scalar) inputs, targets dict), moved to `device`."""
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
    grad_clip: float = 0.0,
) -> EpochResult:
    """Run one training pass over `batches`.

    lr_fn: if given, maps the cumulative rows count to the learning rate,
        applied to every param group before each step. Otherwise the caller
        owns the learning rate.
    rows_trained: cumulative rows before this epoch; the result carries the
        updated count.
    on_batch: progress callback (done_batches, samples, elapsed_s,
        rows_trained), called at most about once per second.
    grad_clip: if > 0, the global gradient-norm clip.
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

        # bf16 for the network only; the loss runs in fp32 on upcast outputs.
        # Rationale in trainer.py's module docstring.
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs = model(input_spatial, input_scalar)
        outputs = {k: v.float() for k, v in outputs.items()}
        losses = model.compute_loss(outputs, targets, loss_cfg)
        optimizer.zero_grad()
        losses["total"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

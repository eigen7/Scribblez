"""One training epoch of the position evaluation model.

The gradient step lives here, apart from the trainer (trainer.py), which owns
the data lifecycle, learning-rate policy, evaluation and checkpointing.

The loop never waits on the GPU: per-step statistics accumulate in device
tensors and are read back once, when the epoch ends. A per-step read (`.item()`)
blocks the host until that step's kernels finish, so the next batch, which the
data loader fills on this same thread, could not be prepared while the GPU works.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch.nn.utils import clip_grads_with_norm_, get_total_norm

# Re-exported: callers import LossConfig from here. The loss itself, and the
# loss and target keys this loop iterates, come from the model's head registry.
from .model import LossConfig

__all__ = ["LossConfig", "EpochResult", "GradNormTracker", "run_epoch"]


@dataclass
class EpochResult:
    """Per-epoch averages, plus the updated cumulative rows counter."""

    losses: dict[str, float]  # per-head means, including "total"
    wld_acc: float
    n_batches: int
    samples: int
    rows_trained: int
    grad_norm: dict[str, float]  # GradNormTracker.summary(), keyed by metric name


class GradNormTracker:
    """Clips each step's gradient to a global norm and keeps the epoch's
    statistics of the norm as measured before clipping, so a run shows whether
    its clip is a rare spike guard or rescales most steps (in which case it
    changes the effective learning rate). The statistics stay on the device
    until summary() (see the module docstring)."""

    def __init__(self, device, clip: float):
        self._clip = clip
        self._sum = torch.zeros((), device=device)
        self._max = torch.zeros((), device=device)
        self._clipped = torch.zeros((), device=device)
        self._steps = 0

    def clip_and_record(self, params: list[torch.Tensor]):
        """Measure the global gradient norm over `params`, record it, and clip
        the gradients to the tracker's norm when that is > 0."""
        norm = get_total_norm([p.grad for p in params if p.grad is not None])
        if self._clip > 0:
            clip_grads_with_norm_(params, self._clip, norm)
            self._clipped += norm > self._clip
        self._sum += norm
        self._max = torch.maximum(self._max, norm)
        self._steps += 1

    def summary(self) -> dict[str, float]:
        """The epoch's mean and max pre-clip norm, and the fraction of steps
        clipped (0 when the tracker does not clip)."""
        steps = max(self._steps, 1)
        return {
            "grad_norm_mean": self._sum.item() / steps,
            "grad_norm_max": self._max.item(),
            "clip_frac": self._clipped.item() / steps,
        }


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
    grad_clip: if > 0, the global gradient-norm clip. The norm is recorded
        either way (EpochResult.grad_norm).
    """
    model.train()
    params = list(model.parameters())
    grad_norms = GradNormTracker(device, grad_clip)
    target_keys = model.target_keys()
    sums = {k: torch.zeros((), device=device) for k in model.loss_keys()}
    n_batches = 0
    correct = torch.zeros((), dtype=torch.long, device=device)
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
        grad_norms.clip_and_record(params)
        optimizer.step()

        bs = input_spatial.shape[0]
        n_batches += 1
        samples += bs
        rows_trained += bs
        for k in sums:
            sums[k] += losses[k].detach()
        correct += (outputs["wld"].argmax(1) == targets["wld"].argmax(1)).sum()

        if on_batch is not None and time.time() - last_progress > 1.0:
            on_batch(n_batches, samples, time.time() - t0, rows_trained)
            last_progress = time.time()

    return EpochResult(
        losses={k: v.item() / max(n_batches, 1) for k, v in sums.items()},
        wld_acc=correct.item() / max(samples, 1),
        n_batches=n_batches,
        samples=samples,
        rows_trained=rows_trained,
        grad_norm=grad_norms.summary(),
    )

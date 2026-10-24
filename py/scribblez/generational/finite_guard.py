"""Stopping a trainer whose training went non-finite, before it checkpoints.

Under bf16 autocast (no loss scaler, fp32's exponent range) a non-finite loss
or gradient is never a transient overflow to step past: the model diverged, a
batch carries a bad value, or an op is numerically broken. A skipped step
would hide each of these, and a diverged model only limps on. So the trainer
stops (WorkerFatal), the dashboard pauses its slot as failed rather than
restarting it into the same failure, and the last checkpoint, written before
the bad step, stays the one to resume from once the cause is fixed.

The check costs no per-step host sync: each step's finiteness stays on the
device, and the host reads the steps since its last look when it already
syncs (the progress line, the epoch's end). The weights a bad step corrupted
are discarded with the process, so a late look loses nothing.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from scribblez.workloads.worker import WorkerFatal


class FiniteGuard:
    """Watches each training step's named values (losses, gradient norms) for a
    non-finite one; check() raises WorkerFatal naming the first bad step."""

    def __init__(self):
        # Per step since the last check: its description, and whether each of
        # its values was finite (device booleans) under their names.
        self._steps: list[tuple[str, list[str], torch.Tensor]] = []

    def observe(self, description: str, values: dict[str, torch.Tensor]):
        """Record a step: `description` names it in the error (its index and
        its batch), `values` are the scalars that must be finite."""
        names = list(values)
        finite = torch.stack([torch.isfinite(values[n].detach()).all() for n in names])
        self._steps.append((description, names, finite))

    def check(self):
        """Raise WorkerFatal for the first step since the last check with a
        non-finite value; one host sync."""
        if not self._steps:
            return
        steps, self._steps = self._steps, []
        flags = torch.stack([finite for _, _, finite in steps]).cpu()
        for (description, names, _), finite in zip(steps, flags, strict=True):
            if not bool(finite.all()):
                bad = ", ".join(n for n, ok in zip(names, finite.tolist(), strict=True) if not ok)
                raise WorkerFatal(f"non-finite {bad} at {description}")


def check_parameters(models: Iterable[tuple[str, nn.Module]], when: str):
    """Raise WorkerFatal if any parameter or buffer of the named `models` is
    non-finite: the last look before a checkpoint makes the weights the one
    to resume from."""
    for name, model in models:
        tensors = list(model.parameters()) + list(model.buffers())
        floats = [t for t in tensors if t.is_floating_point()]
        if floats and not bool(torch.stack([torch.isfinite(t).all() for t in floats]).all()):
            raise WorkerFatal(f"{name}'s weights are non-finite {when}")

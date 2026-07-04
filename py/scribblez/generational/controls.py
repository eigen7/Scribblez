"""Live operator controls shared by the generational trainers.

Every generational trainer (post-move value, max-move-per-lane) exposes the same
dashboard-tunable knobs -- a base learning rate and the three CPU-thread pools --
persisted in the per-tag dashboard DB's control table and restored on restart.
The controllers here read those controls at their natural cadence (the LR once
per epoch, the CPU pools once per generation), apply them, and log each change as
a rows-clock control event so the metric plots can annotate where a knob moved.
The task-specific orchestrators own only their model, loss, and evaluation.
"""

from __future__ import annotations

import sys

import torch

from ..dashboard import db
from ..train_common import timed_print
from .lr import make_lr_fn

# Names of the live controls (dashboard Controls tab / DB). The base learning rate
# plus the three CPU-thread pools the operator can retune between generations.
CONTROL_BASE_LR = "base_lr"
CONTROL_GEN_THREADS = "gen_threads"
CONTROL_DATALOADER_WORKERS = "dataloader_workers"
CONTROL_TORCH_THREADS = "torch_threads"


class LrController:
    """Serves the per-step learning rate from the live base rate in the control
    table, so the operator can step it down mid-run from the dashboard. The base
    is read once per epoch (cheap; the rate only needs to change at epoch
    granularity) and scaled by the rows-clock warmup. When the operator has moved
    the base since the last epoch, a rows-clock control event is recorded so the
    metric plots can annotate where it changed."""

    def __init__(self, conn, args):
        self._conn = conn
        self._warmup_rows = args.warmup_rows
        self._default = args.lr
        self.base = db.read_control(conn, CONTROL_BASE_LR, default=args.lr)

    def epoch_lr_fn(self, rows_trained: int):
        """The per-step lr_fn for the upcoming epoch, from the current base rate."""
        base = db.read_control(self._conn, CONTROL_BASE_LR, default=self._default)
        if base != self.base:
            db.write_control_event(self._conn, rows_trained, CONTROL_BASE_LR, base)
            timed_print(f"base LR {self.base:.2e} -> {base:.2e} at {rows_trained} rows")
            self.base = base
        return make_lr_fn(base, self._warmup_rows)


class CpuController:
    """Serves the live CPU-thread controls -- game-generation threads, C++
    DataLoader workers, and PyTorch intra-op threads -- from the control table,
    refreshed once per generation (the natural point to retune, since the
    generation subprocess and the dataset are rebuilt there). torch's thread count
    is applied here; the other two are read by the generation and dataset builders
    via the properties. Changes are recorded as rows-clock control events."""

    def __init__(self, conn, args):
        self._conn = conn
        self._defaults = {
            CONTROL_GEN_THREADS: args.gen_threads,
            CONTROL_DATALOADER_WORKERS: args.dataloader_workers,
            CONTROL_TORCH_THREADS: args.torch_threads or torch.get_num_threads(),
        }
        self._vals: dict = {}
        self.refresh(0)

    def refresh(self, rows_trained: int):
        """Re-read the thread controls, apply torch's count, and log any changes.
        Call once per generation."""
        vals = {
            name: max(1, int(db.read_control(self._conn, name, default=default)))
            for name, default in self._defaults.items()
        }
        for name, v in vals.items():
            if self._vals and self._vals.get(name) != v:
                db.write_control_event(self._conn, rows_trained, name, v)
                timed_print(f"{name} {self._vals[name]} -> {v} at {rows_trained} rows")
        self._vals = vals
        torch.set_num_threads(vals[CONTROL_TORCH_THREADS])

    @property
    def gen_threads(self) -> int:
        return self._vals[CONTROL_GEN_THREADS]

    @property
    def dataloader_workers(self) -> int:
        return self._vals[CONTROL_DATALOADER_WORKERS]


def init_controls(conn, args):
    """Seed the four live controls with the CLI-supplied initial values (kept
    across restarts; retuned from the Controls tab). The --lr / --*-threads flags
    only set the starting point."""
    db.init_control(
        conn,
        {
            CONTROL_BASE_LR: args.lr,
            CONTROL_GEN_THREADS: args.gen_threads,
            CONTROL_DATALOADER_WORKERS: args.dataloader_workers,
            CONTROL_TORCH_THREADS: args.torch_threads or torch.get_num_threads(),
        },
    )


def progress_line(generation_index, epoch, done_batches, samples, elapsed, rows):
    """run_epoch on_batch callback: an in-place per-epoch throughput line."""
    rate = samples / elapsed if elapsed > 0 else 0.0
    sys.stdout.write(
        f"\r  gen {generation_index} epoch {epoch}: {done_batches} batches | "
        f"{rate / 1000:.1f}k rows/s | {rows} rows total   "
    )
    sys.stdout.flush()

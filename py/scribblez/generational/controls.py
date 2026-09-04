"""Learning-rate schedule and live operator controls shared by the generational
trainers.

Every generational trainer (position evaluation, move-set evaluation,
max-move-per-lane) drives its learning rate from the same rows-clock schedule
(WsdLrController) and exposes the same dashboard-tunable CPU knobs -- C++
DataLoader workers and torch intra-op threads -- read from the tag's controls
file (generational/records.py), which the dashboard keeps across restarts.
Game-generation capacity is deliberately not a control here: generation
belongs to the generator worker fleet, sized per worker slot from the master
dashboard.

The CPU controller reads its controls once per generation, applies them, and
logs each change as a rows-clock control event so the metric plots can annotate
where a knob moved; the LR schedule logs its phase boundaries the same way.
Events go to the trainer's recorder, which delivers them with the next
generation's record. The task-specific trainers own only their model, loss,
and evaluation.
"""

from __future__ import annotations

import math
import sys

import torch

from ..train_common import timed_print

# Names of the live controls (dashboard Controls tab / DB).
CONTROL_DATALOADER_WORKERS = "dataloader_workers"
CONTROL_TORCH_THREADS = "torch_threads"

# Event name under which the LR schedule logs its phase boundaries. Not a
# control (nothing reads it back): a derived value the plots annotate.
LR_EVENT = "lr"

# Starting points for the CPU-thread controls when a run first creates them.
DEFAULT_DATALOADER_WORKERS = 4

# Shape of the schedule's decay tail, shared by every trainer: the last
# LR_DECAY_FRAC of each cycle decays (cosine) from the peak to peak *
# LR_FLOOR_FRAC. The floor is well above zero because a restart follows
# immediately -- decaying to ~0 right before jumping back up wastes the tail.
LR_DECAY_FRAC = 0.2
LR_FLOOR_FRAC = 0.1

# Schedule phases, in cycle order. Warmup happens once; the other three repeat.
PHASE_WARMUP = "warmup"
PHASE_REWARMUP = "rewarmup"
PHASE_STABLE = "stable"
PHASE_DECAY = "decay"


class WsdSchedule:
    """Warmup-stable-decay learning rate with periodic restarts, as a pure
    function of the rows-clock.

    An open-ended self-play run has no known horizon, so a single end-of-run
    decay has no trigger point; instead the stable/decay pair repeats every
    `cycle_rows`, giving a well-annealed checkpoint per cycle and then a warm
    restart back to the peak (this project's own adaptation of WSD to the
    continual setting, structurally like SGDR warm restarts with WSD's tail).

    With W = warmup_rows, C = cycle_rows, R = W // 4 and t = (rows - W) mod C:
      rows < W                 warmup    linear 0 -> lr
      t < R (not first cycle)  rewarmup  linear lr*floor -> lr
      R <= t < (1-D)*C         stable    lr
      (1-D)*C <= t < C         decay     cosine lr -> lr*floor
    The re-warmup sits inside the cycle (period stays C) and ramps from the
    floor rather than 0: AdamW's second-moment estimate has adapted to the
    low-LR regime by the end of a decay, so a bare jump to the peak risks an
    oversized effective step for the first post-restart batches. Degenerate
    settings (re-warmup swallowing the stable segment) are not rejected; every
    row count still maps to a value.
    """

    def __init__(self, lr: float, warmup_rows: int, cycle_rows: int):
        self.lr = lr
        self.warmup_rows = warmup_rows
        self.cycle_rows = cycle_rows
        self.rewarmup_rows = warmup_rows // 4
        self.decay_start = round((1.0 - LR_DECAY_FRAC) * cycle_rows)
        self.floor = lr * LR_FLOOR_FRAC

    @classmethod
    def from_params(cls, params) -> WsdSchedule:
        """From a trainer's params dataclass (`lr`, `lr_warmup_rows`, `lr_cycle_rows`)."""
        return cls(params.lr, params.lr_warmup_rows, params.lr_cycle_rows)

    def phase(self, rows: int) -> str:
        """Which segment of the schedule `rows` falls in."""
        if rows < self.warmup_rows:
            return PHASE_WARMUP
        since_warmup = rows - self.warmup_rows
        t = since_warmup % self.cycle_rows
        if t >= self.decay_start:
            return PHASE_DECAY
        if t < self.rewarmup_rows and since_warmup >= self.cycle_rows:
            return PHASE_REWARMUP
        return PHASE_STABLE

    def value(self, rows: int) -> float:
        """The learning rate at `rows`."""
        phase = self.phase(rows)
        if phase == PHASE_WARMUP:
            return self.lr * rows / self.warmup_rows
        t = (rows - self.warmup_rows) % self.cycle_rows
        if phase == PHASE_REWARMUP:
            return self.floor + (self.lr - self.floor) * t / self.rewarmup_rows
        if phase == PHASE_DECAY:
            frac = (t - self.decay_start) / (self.cycle_rows - self.decay_start)
            return self.floor + (self.lr - self.floor) * 0.5 * (1.0 + math.cos(math.pi * frac))
        return self.lr


class WsdLrController:
    """Serves the WsdSchedule as a trainer's per-batch lr_fn and records its
    phase boundaries as rows-clock control events.

    Nothing here is persisted: the schedule is a function of `rows_trained`,
    which the generational checkpoint already carries, so a resume re-derives
    everything from the cursor. `.current` is the rate applied to the most
    recent batch, which is what the trainers' end-of-generation log line and
    metrics row report. Phase crossings are detected per batch and logged at
    the exact rows position; the phase is initialised from the resume cursor so
    a restart mid-phase logs nothing spurious."""

    def __init__(self, recorder, schedule: WsdSchedule, rows_trained: int):
        self._recorder = recorder
        self.schedule = schedule
        self._phase = schedule.phase(rows_trained)
        self.current = schedule.value(rows_trained)

    def lr_fn(self, rows_trained: int) -> float:
        """run_epoch's per-step lr_fn: the rate for the batch starting at
        `rows_trained`."""
        phase = self.schedule.phase(rows_trained)
        self.current = self.schedule.value(rows_trained)
        if phase != self._phase:
            self._recorder.control_event(rows_trained, LR_EVENT, self.current)
            timed_print(
                f"LR schedule {self._phase} -> {phase} ({self.current:.2e}) at {rows_trained} rows"
            )
            self._phase = phase
        return self.current


def default_controls() -> dict[str, int]:
    """The CPU-thread controls' starting values on this machine: what a run
    publishes for the Controls tab to show until the operator moves them."""
    return {
        CONTROL_DATALOADER_WORKERS: DEFAULT_DATALOADER_WORKERS,
        CONTROL_TORCH_THREADS: torch.get_num_threads(),
    }


class CpuController:
    """Serves the live CPU-thread controls -- C++ DataLoader workers and PyTorch
    intra-op threads -- refreshed once per generation (the natural point to
    retune, since the dataset is rebuilt there). `read_controls()` returns the
    operator's current values (records.read_controls over the trainer's sink);
    a control it lacks is at its default. torch's thread count is applied
    here; the DataLoader count is read by the dataset builder via the
    property. Changes are recorded as rows-clock control events."""

    def __init__(self, recorder, read_controls):
        self._recorder = recorder
        self._read_controls = read_controls
        self._defaults = default_controls()
        self._vals: dict = {}
        self.refresh(0)

    def refresh(self, rows_trained: int):
        """Re-read the thread controls, apply torch's count, and log any changes.
        Call once per generation."""
        current = self._read_controls()
        vals = {
            name: max(1, int(current.get(name, default)))
            for name, default in self._defaults.items()
        }
        for name, v in vals.items():
            if self._vals and self._vals.get(name) != v:
                self._recorder.control_event(rows_trained, name, v)
                timed_print(f"{name} {self._vals[name]} -> {v} at {rows_trained} rows")
        self._vals = vals
        torch.set_num_threads(vals[CONTROL_TORCH_THREADS])

    @property
    def dataloader_workers(self) -> int:
        return self._vals[CONTROL_DATALOADER_WORKERS]


def progress_line(generation_index, done_batches, samples, elapsed, rows):
    """run_epoch on_batch callback: an in-place per-generation throughput line."""
    rate = samples / elapsed if elapsed > 0 else 0.0
    sys.stdout.write(
        f"\r  gen {generation_index}: {done_batches} batches | "
        f"{rate / 1000:.1f}k rows/s | {rows} rows total   "
    )
    sys.stdout.flush()

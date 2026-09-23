"""The warmup-stable-decay LR schedule and the live CPU controls shared by the
trainers.

The "rows-clock" is the count of training rows seen so far. The WSD schedule
is a pure function of it, which is what lets a resumed trainer pick up the
schedule from its checkpoint cursor alone. The evidence and max_move_per_lane
trainers run it directly (WsdLrController); position_eval and move_set_eval
reach it through the `wsd` optimizer arm (optim.py).

The CPU controls (C++ DataLoader workers, torch intra-op threads) are knobs
the operator can move mid-run from the dashboard's Controls tab. They reach
the trainer through the tag's controls file (records.py), read once per
generation. Generation capacity is not a control: it belongs to the generator
fleet, sized per worker slot.

Both the CPU controller and the LR schedule log changes as rows-clock control
events, so the metric plots can mark where a knob moved or a phase began. The
events travel with the next generation's record.
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

    An open-ended self-play run has no known horizon to time a single final
    decay against. Instead the stable/decay pair repeats every `cycle_rows`,
    giving a well-annealed checkpoint at the end of each cycle followed by a
    warm restart to the peak: SGDR-style restarts with WSD's decay tail.

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
    phase boundaries as control events.

    Nothing here is persisted: a resume re-derives everything from the
    checkpoint's `rows_trained`. `current` is the rate applied to the most
    recent batch, which the metrics row reports. The phase starts from the
    resume cursor, so a restart mid-phase logs no spurious crossing."""

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
    """The CPU-thread controls' starting values on this machine, shown on the
    Controls tab until the operator moves them."""
    return {
        CONTROL_DATALOADER_WORKERS: DEFAULT_DATALOADER_WORKERS,
        CONTROL_TORCH_THREADS: torch.get_num_threads(),
    }


class CpuController:
    """Serves the live CPU-thread controls, refreshed once per generation: the
    natural point to retune, since the dataset is rebuilt there.

    `read_controls()` returns the operator's current values (typically
    records.read_controls over the trainer's sink); a missing control keeps its
    default. The torch thread count is applied here; the dataset builder reads
    `dataloader_workers`."""

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

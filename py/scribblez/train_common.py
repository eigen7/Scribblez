"""Shared scaffolding for the streaming trainers.

Both streaming trainers (post-move value, max-move-per-lane) drive the same
in-process self-play ring buffer and write to the same per-tag dashboard DB, so
the task-independent plumbing -- checkpoint save/resume, run-artifact reset,
throughput/backpressure accounting, and interval averaging -- lives here. Each
trainer keeps only its task-specific model, loss, and eval.
"""

import sys
from datetime import datetime
from pathlib import Path

import torch

from .dashboard import db
from .paths import TagPaths


def timed_print(msg: str):
    """Print `msg` with a millisecond-resolution local-time prefix, e.g.
    '2026-06-29 11:45:09.815 <msg>'. Used for the trainers' progress lines."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} {msg}")


def reset_tag(paths: TagPaths):
    """Wipe prior run artifacts (checkpoints, onnx, dashboard DB). Keeps any val set."""
    import shutil

    print(f"--restart: clearing prior run artifacts under {paths.root}", file=sys.stderr)
    shutil.rmtree(paths.checkpoints_dir, ignore_errors=True)
    shutil.rmtree(paths.onnx_dir, ignore_errors=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(paths.dashboard_db) + suffix).unlink(missing_ok=True)


def save_rolling_checkpoint(
    path: Path, model, optimizer, ckpt_idx: int, positions: int, step: int, args
):
    """Persist the single rolling checkpoint that holds resume state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ckpt_idx": ckpt_idx,
            "positions": positions,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def maybe_resume(paths: TagPaths, model, optimizer, device) -> tuple[int, int, int]:
    """Load the rolling checkpoint if present; return (ckpt_idx, positions, step)."""
    p = paths.rolling_checkpoint
    if not p.exists():
        return 0, 0, 0
    ckpt = torch.load(p, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    ci, pos, step = int(ckpt["ckpt_idx"]), int(ckpt["positions"]), int(ckpt.get("step", 0))
    print(f"Resuming from {p.name}: checkpoint {ci}, {pos} positions trained")
    return ci, pos, step


class ThroughputMeter:
    """Tracks streaming throughput and producer/consumer backpressure between samples.

    Each sample() turns the streamer's cumulative counters into a dashboard
    throughput row (rate over the last interval + which side -- CPU producers or
    GPU consumer -- is the bottleneck, from the wait-time deltas)."""

    def __init__(self, positions: int, t0: float):
        self.last_t = t0
        self.last_positions = positions
        self.last_prod_ns = 0
        self.last_cons_ns = 0

    def sample(self, now: float, positions: int, stats: dict) -> dict:
        d_prod = stats["producer_blocked_ns"] - self.last_prod_ns
        d_cons = stats["consumer_blocked_ns"] - self.last_cons_ns
        dt = max(now - self.last_t, 1e-9)
        pos_per_s = (positions - self.last_positions) / dt
        bottleneck = "cpu" if d_cons > d_prod else "gpu"
        self.last_t, self.last_positions = now, positions
        self.last_prod_ns, self.last_cons_ns = (
            stats["producer_blocked_ns"],
            stats["consumer_blocked_ns"],
        )
        return {
            "t": now,
            "positions": positions,
            "games": stats["games_played"],
            "positions_per_s": pos_per_s,
            "producer_blocked_ns": stats["producer_blocked_ns"],
            "consumer_blocked_ns": stats["consumer_blocked_ns"],
            "bottleneck": bottleneck,
        }


class IntervalStats:
    """Running mean of arbitrary named scalars over a checkpoint interval."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums: dict[str, float] = {}
        self.n_batches = 0

    def update(self, values: dict):
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.n_batches += 1

    def record(self) -> dict:
        nb = max(self.n_batches, 1)
        return {k: v / nb for k, v in self.sums.items()}


def add_train_log_args(parser):
    """Register the knob for per-minibatch dashboard resolution.
    Shared by the trainers so the default and help text live in one place."""
    parser.add_argument(
        "--max-log-points", type=int, default=1024,
        help="Target number of points the loss/accuracy plots show. New minibatches "
             "are stored aggregated at ~total/this resolution (append-only; older "
             "data stays finer), and the read rolls everything up to ~this many "
             "uniform points. Storage grows only logarithmically with the run.",
    )


class TrainStepWriter:
    """Appends per-minibatch training metrics to the dashboard at a self-coarsening
    *gradient* of resolutions, while the read (db.read_train_steps) rolls them up
    to one uniform resolution for display.

    A point aggregates `r` consecutive minibatches (mean + weight `n`=r), where the
    current `r` is a power of two chosen so the newest data sits at ~max_points
    points: r = the smallest power of two with total_minibatches <= max_points * r.
    As the run grows `r` doubles, so recent segments are coarser than old ones --
    but nothing is ever rewritten (append-only), and the older finer points let the
    read reconstruct a uniform view at any resolution. Storage therefore grows only
    logarithmically with the run (~max_points rows per doubling).

    Lifecycle: record() each minibatch; commit() appends completed points on a
    cadence; close() flushes the open partial bucket. Resume needs no state -- `r`
    is derived from the cumulative minibatch index each time, and the DB already
    holds the history.
    """

    def __init__(self, conn, max_points: int):
        self._conn = conn
        self._max_points = max(1, max_points)
        self._pending: list[dict] = []
        self._reset_open()

    def _reset_open(self):
        self._sums: dict[str, float] = {}
        self._count = 0
        self._last_step = 0
        self._last_positions = 0

    def _resolution(self, total_minibatches: int) -> int:
        """Minibatches per point for the current run length: the smallest power of
        two `r` with total_minibatches <= max_points * r (~max_points newest points)."""
        r = 1
        while total_minibatches > self._max_points * r:
            r *= 2
        return r

    def _emit(self):
        """Turn the full (or, on close, partial) open bucket into a point."""
        if self._count == 0:
            return
        self._pending.append({
            "step": self._last_step, "positions": self._last_positions, "n": self._count,
            **{name: total / self._count for name, total in self._sums.items()},
        })
        self._reset_open()

    def record(self, step: int, positions: int, metrics: dict):
        """Fold one minibatch into the open bucket, emitting a point when it fills
        the current resolution."""
        for name, value in metrics.items():
            self._sums[name] = self._sums.get(name, 0.0) + float(value)
        self._count += 1
        self._last_step, self._last_positions = step, positions
        if self._count >= self._resolution(step):
            self._emit()

    def commit(self):
        """Append the completed points (no rewrite); keep the open bucket open."""
        if self._pending:
            db.write_train_steps(self._conn, self._pending)
            self._pending.clear()

    def close(self):
        """Emit the open partial bucket as a final point, then commit."""
        self._emit()
        self.commit()

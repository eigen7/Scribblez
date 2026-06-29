"""Shared scaffolding for the streaming trainers.

Both streaming trainers (post-move value, max-move-per-lane) drive the same
in-process self-play ring buffer and write to the same per-tag dashboard DB, so
the task-independent plumbing -- checkpoint save/resume, run-artifact reset,
throughput/backpressure accounting, and interval averaging -- lives here. Each
trainer keeps only its task-specific model, loss, and eval.
"""

import math
import statistics
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
    """Register the knob that bounds per-minibatch dashboard resolution.
    Shared by the trainers so the default and help text live in one place."""
    parser.add_argument(
        "--max-log-points", type=int, default=1024,
        help="Max per-minibatch dashboard points kept. When the series reaches "
             "this, the WHOLE series re-aggregates to the next coarser power-of-two "
             "resolution -- the plot stays one uniform, smoothly-coarsening series.",
    )


class TrainStepWriter:
    """Writes per-minibatch training metrics to the dashboard at a uniform, self-
    coarsening resolution.

    Every point in the series represents the same number of minibatches, `bucket`
    (a power of two). New minibatches accumulate into the open bucket; a full
    bucket becomes a point. When the series reaches `max_points`, the WHOLE series
    re-aggregates: adjacent points are merged pairwise (equal-weight means
    compose), which halves the count and doubles `bucket`. So the plot is always
    ONE uniform resolution that doubles over the run -- no dense-early/sparse-late
    seam -- and the DB stays bounded to ~max_points rows.

    The writer owns the entire (downsampled) series in memory and rewrites it to
    the DB on commit() (one transaction, so the dashboard never reads it empty);
    close() also flushes the open partial bucket. On construction it reloads any
    existing series (resume) and infers `bucket` from the point spacing.
    """

    def __init__(self, conn, max_points: int, batch_size: int):
        self._conn = conn
        self._max_points = max(2, max_points - max_points % 2)  # even: halving splits cleanly
        self._batch = max(batch_size, 1)
        self._points: list[dict] = []
        self._bucket = 1  # minibatches per point (a power of two)
        self._dirty = False
        self._reset_open()
        self._load_existing()

    def _reset_open(self):
        self._sums: dict[str, float] = {}
        self._count = 0
        self._last_step = 0
        self._last_positions = 0

    def _load_existing(self):
        """Reload the series from the DB (resume) and infer the current `bucket`."""
        existing = db.read_train_steps(self._conn)
        pos = existing.get("positions", [])
        if len(pos) == 0:
            return
        metrics = [k for k in existing if k not in ("step", "positions")]
        self._points = [
            {"step": int(existing["step"][i]), "positions": int(pos[i]),
             **{m: float(existing[m][i]) for m in metrics}}
            for i in range(len(pos))
        ]
        if len(pos) >= 2:
            spacing = statistics.median(pos[i + 1] - pos[i] for i in range(len(pos) - 1))
            self._bucket = 2 ** round(math.log2(max(spacing / self._batch, 1.0)))
        while len(self._points) > self._max_points:
            self._halve()

    @staticmethod
    def _merge(a: dict, b: dict) -> dict:
        """Mean of two equal-weight points; the later one's step/positions span it."""
        out = {"step": b["step"], "positions": b["positions"]}
        for k in a:
            if k not in ("step", "positions"):
                out[k] = (a[k] + b[k]) / 2.0
        return out

    def _halve(self):
        pts = self._points
        merged = [self._merge(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
        if len(pts) % 2:
            merged.append(pts[-1])  # odd tail (a resume edge); keep as-is
        self._points = merged
        self._bucket *= 2

    def _emit_point(self):
        """Turn the full (or, on close, partial) open bucket into a point."""
        if self._count == 0:
            return
        self._points.append({
            "step": self._last_step, "positions": self._last_positions,
            **{name: total / self._count for name, total in self._sums.items()},
        })
        self._reset_open()
        self._dirty = True
        if len(self._points) >= self._max_points:
            self._halve()

    def record(self, step: int, positions: int, metrics: dict):
        """Fold one minibatch into the open bucket, emitting a point when it fills."""
        for name, value in metrics.items():
            self._sums[name] = self._sums.get(name, 0.0) + float(value)
        self._count += 1
        self._last_step, self._last_positions = step, positions
        if self._count >= self._bucket:
            self._emit_point()

    def commit(self):
        """Rewrite the (bounded) series to the DB if it changed since last commit."""
        if self._dirty:
            db.replace_train_steps(self._conn, self._points)
            self._dirty = False

    def close(self):
        """Emit the open partial bucket as a final point, then commit."""
        self._emit_point()
        self.commit()

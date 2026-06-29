"""Shared scaffolding for the streaming trainers.

Both streaming trainers (post-move value, max-move-per-lane) drive the same
in-process self-play ring buffer and write to the same per-tag dashboard DB, so
the task-independent plumbing -- checkpoint save/resume, run-artifact reset,
throughput/backpressure accounting, and interval averaging -- lives here. Each
trainer keeps only its task-specific model, loss, and eval.
"""

import sys
from pathlib import Path

import torch

from .dashboard import db
from .paths import TagPaths


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
    """Register the two knobs that control per-minibatch dashboard resolution.
    Shared by the trainers so the defaults and help text live in one place."""
    parser.add_argument(
        "--fine-log-positions", type=int, default=102400,
        help="Log every minibatch until this many positions are trained (catch early anomalies).",
    )
    parser.add_argument(
        "--coarse-log-window", type=int, default=25600,
        help="After the fine phase, log one mean-aggregated point per this many positions.",
    )


class TrainStepWriter:
    """Writes per-minibatch training metrics to the dashboard DB at an adaptive
    resolution, so a long run does not bloat the DB while early detail is kept.

    Resolution schedule (positions = cumulative training rows):
      * fine phase   -- one row per minibatch, until `fine_positions` positions;
      * coarse phase -- one row per `window` positions, each the MEAN of the
                        minibatches in that span.
    Aggregation is the only state; the DB stays a plain long-format point store
    and the dashboard downsamples further at render time.

    Lifecycle: record() each minibatch; commit() on a cadence to batch completed
    rows to the DB; close() at the end to flush the final partial bucket. commit()
    deliberately leaves the in-progress bucket alone so a coarse bucket keeps
    accumulating across commits (no partial coarse rows reach the DB until full).
    """

    def __init__(self, conn, fine_positions: int, window: int, start_positions: int = 0):
        self._conn = conn
        self._fine_positions = fine_positions
        self._window = max(window, 1)
        self._bucket_start = start_positions  # positions at the last flush boundary
        self._pending: list[dict] = []
        self._reset_bucket()

    def _reset_bucket(self):
        self._sums: dict[str, float] = {}
        self._count = 0
        self._last_step = 0
        self._last_positions = 0

    def _bucket_full(self, positions: int) -> bool:
        if positions <= self._fine_positions:
            return True  # fine phase: every minibatch is its own bucket
        return positions - self._bucket_start >= self._window

    def _close_bucket(self):
        """Reduce the open bucket to one mean row in `_pending` (no-op if empty)."""
        if self._count == 0:
            return
        row = {"step": self._last_step, "positions": self._last_positions}
        row.update({name: total / self._count for name, total in self._sums.items()})
        self._pending.append(row)
        self._bucket_start = self._last_positions
        self._reset_bucket()

    def record(self, step: int, positions: int, metrics: dict):
        """Fold one minibatch's metrics into the open bucket, closing it when full."""
        for name, value in metrics.items():
            self._sums[name] = self._sums.get(name, 0.0) + float(value)
        self._count += 1
        self._last_step, self._last_positions = step, positions
        if self._bucket_full(positions):
            self._close_bucket()

    def commit(self):
        """Batch-write the completed bucket rows; keep any in-progress bucket open."""
        if self._pending:
            db.write_train_steps(self._conn, self._pending)
            self._pending.clear()

    def close(self):
        """Flush the final partial bucket, then commit."""
        self._close_bucket()
        self.commit()

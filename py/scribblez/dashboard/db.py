"""SQLite store for training metrics and eval artifacts.

A single per-tag database (``tags/<tag>/dashboard.db``) holds everything the
dashboard renders, so training writes data (never PNGs) and the Bokeh app
renders on the fly. WAL mode lets the dashboard read while training writes.

Tables:
  meta            one row of run config (args, model size, timestamps)
  metrics         long-format scalar series: (epoch, name) -> value
  monotonicity    per-epoch win-rate curves: score_diffs, win_rate, per-curve scores
  score_belief    per-epoch score-diff percentile bands
  calibration     per-epoch reliability binning (WLD + score-diff)
  throughput      streaming throughput/backpressure time series (one row per sample)
  train_step      per-minibatch loss/accuracy time series (streaming pipeline)

NumPy arrays are stored as ``np.save`` BLOBs (shape + dtype preserved). The
frozen test-subset board images are NOT in the DB -- they are static PNGs
rendered offline to ``tags/<tag>/test-subset/pos-NN.png`` and displayed directly.
"""

import io
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Array (de)serialization
# --------------------------------------------------------------------------


def to_blob(arr: np.ndarray) -> bytes:
    """Serialize a NumPy array to a .npy-format BLOB (shape + dtype preserved)."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr))
    return buf.getvalue()


def from_blob(blob: bytes) -> np.ndarray:
    """Inverse of to_blob."""
    return np.load(io.BytesIO(blob), allow_pickle=False)


# --------------------------------------------------------------------------
# Connection + schema
# --------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  id INTEGER PRIMARY KEY CHECK (id = 0),
  tag TEXT, args_json TEXT, model_params INTEGER,
  created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS metrics (
  epoch INTEGER, name TEXT, value REAL,
  PRIMARY KEY (epoch, name)
);
CREATE TABLE IF NOT EXISTS monotonicity (
  epoch INTEGER PRIMARY KEY,
  score_diffs BLOB, win_rate BLOB, curve_scores BLOB
);
CREATE TABLE IF NOT EXISTS score_belief (
  epoch INTEGER PRIMARY KEY,
  score_diffs BLOB, quantiles BLOB, bands BLOB
);
CREATE TABLE IF NOT EXISTS calibration (
  epoch INTEGER PRIMARY KEY,
  rel_edges BLOB, rel_pred BLOB, rel_actual BLOB, rel_count BLOB,
  sd_edges BLOB, sd_pred BLOB, sd_actual BLOB, sd_count BLOB
);
CREATE TABLE IF NOT EXISTS throughput (
  t REAL,                       -- wall-clock sample time (epoch seconds)
  positions INTEGER,            -- cumulative positions trained
  games INTEGER,                -- cumulative games produced (== positions; 1 sample/game)
  positions_per_s REAL,         -- throughput over the last interval
  producer_blocked_ns INTEGER,  -- cumulative producer wait (GPU-bound when rising)
  consumer_blocked_ns INTEGER,  -- cumulative consumer wait (CPU-bound when rising)
  bottleneck TEXT               -- 'cpu' or 'gpu', from the interval's wait deltas
);
CREATE TABLE IF NOT EXISTS train_step (
  step INTEGER,                 -- cumulative minibatch index (x-axis)
  positions INTEGER,            -- cumulative positions trained
  name TEXT,                    -- metric name: 'loss', 'loss_<head>', '<x>_acc'
  value REAL
);
CREATE TABLE IF NOT EXISTS loss_weights (
  name TEXT PRIMARY KEY,        -- a per-component loss series ('loss_<head>')
  weight REAL                   -- its coefficient in the optimized total loss
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the dashboard DB in WAL mode with the schema applied."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------


def write_meta(conn: sqlite3.Connection, tag: str, args: dict, model_params: int):
    now = time.time()
    conn.execute(
        "INSERT INTO meta (id, tag, args_json, model_params, created_at, updated_at) "
        "VALUES (0, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET tag=excluded.tag, args_json=excluded.args_json, "
        "model_params=excluded.model_params, updated_at=excluded.updated_at",
        (tag, json.dumps(args), int(model_params), now, now),
    )
    conn.commit()


def write_metrics(conn: sqlite3.Connection, epoch: int, record: dict):
    """Upsert every numeric value in `record` (besides 'epoch') as a scalar series point."""
    rows = [
        (epoch, name, float(value))
        for name, value in record.items()
        if name != "epoch" and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    conn.executemany(
        "INSERT INTO metrics (epoch, name, value) VALUES (?, ?, ?) "
        "ON CONFLICT(epoch, name) DO UPDATE SET value=excluded.value",
        rows,
    )
    conn.commit()


def write_loss_weights(conn: sqlite3.Connection, weights: dict):
    """Record each per-component loss series' coefficient in the optimized total
    (e.g. {'loss_score_cdf': lambda_cdf, ...}). The dashboard stacks the WEIGHTED
    components, so a band's height is how much that term actually drives the loss.
    Idempotent; insertion order is preserved (it sets the stacking order)."""
    conn.executemany(
        "INSERT INTO loss_weights (name, weight) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET weight=excluded.weight",
        [(name, float(w)) for name, w in weights.items()],
    )
    conn.commit()


def read_loss_weights(conn: sqlite3.Connection) -> dict:
    """The per-component loss weights in insertion order (empty if none recorded,
    e.g. a DB written before this table existed -> the plot falls back to lines)."""
    return {
        r["name"]: r["weight"]
        for r in conn.execute("SELECT name, weight FROM loss_weights ORDER BY rowid")
    }


def write_monotonicity(
    conn: sqlite3.Connection, epoch: int, score_diffs, win_rate, curve_scores
):
    """Store the win-rate curves and per-curve scores (N x 3: structural, r2, violations)."""
    conn.execute(
        "INSERT INTO monotonicity (epoch, score_diffs, win_rate, curve_scores) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(epoch) DO UPDATE SET score_diffs=excluded.score_diffs, "
        "win_rate=excluded.win_rate, curve_scores=excluded.curve_scores",
        (epoch, to_blob(score_diffs), to_blob(win_rate), to_blob(curve_scores)),
    )
    conn.commit()


def write_score_belief(conn: sqlite3.Connection, epoch: int, score_diffs, quantiles, bands):
    conn.execute(
        "INSERT INTO score_belief (epoch, score_diffs, quantiles, bands) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(epoch) DO UPDATE SET score_diffs=excluded.score_diffs, "
        "quantiles=excluded.quantiles, bands=excluded.bands",
        (epoch, to_blob(score_diffs), to_blob(quantiles), to_blob(bands)),
    )
    conn.commit()


def write_calibration(conn: sqlite3.Connection, epoch: int, report):
    """Store the calibration reliability binning (scalars go through write_metrics)."""
    conn.execute(
        "INSERT INTO calibration "
        "(epoch, rel_edges, rel_pred, rel_actual, rel_count, sd_edges, sd_pred, sd_actual, sd_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(epoch) DO UPDATE SET "
        "rel_edges=excluded.rel_edges, rel_pred=excluded.rel_pred, rel_actual=excluded.rel_actual, "
        "rel_count=excluded.rel_count, sd_edges=excluded.sd_edges, sd_pred=excluded.sd_pred, "
        "sd_actual=excluded.sd_actual, sd_count=excluded.sd_count",
        (
            epoch,
            to_blob(report.rel_bin_edges), to_blob(report.rel_bin_pred),
            to_blob(report.rel_bin_actual), to_blob(report.rel_bin_count),
            to_blob(report.sd_bin_edges), to_blob(report.sd_bin_pred),
            to_blob(report.sd_bin_actual), to_blob(report.sd_bin_count),
        ),
    )
    conn.commit()


_THROUGHPUT_COLS = (
    "t",
    "positions",
    "games",
    "positions_per_s",
    "producer_blocked_ns",
    "consumer_blocked_ns",
    "bottleneck",
)


def write_throughput(conn: sqlite3.Connection, sample: dict):
    """Append one throughput/backpressure time-series sample."""
    conn.execute(
        "INSERT INTO throughput "
        "(t, positions, games, positions_per_s, producer_blocked_ns, "
        "consumer_blocked_ns, bottleneck) VALUES (?, ?, ?, ?, ?, ?, ?)",
        tuple(sample[c] for c in _THROUGHPUT_COLS),
    )
    conn.commit()


def write_train_steps(conn: sqlite3.Connection, rows: list[dict]):
    """Append per-minibatch training stats (batched insert; no-op if empty).

    Long-format: each row is `{step, positions, <metric>: value, ...}` and every
    metric becomes its own `(step, positions, name, value)` record. New loss or
    accuracy series therefore need no schema change -- a metric appears simply by
    being present in the row dicts."""
    if not rows:
        return
    long_rows = [
        (r["step"], r["positions"], name, float(value))
        for r in rows
        for name, value in r.items()
        if name not in ("step", "positions")
    ]
    conn.executemany(
        "INSERT INTO train_step (step, positions, name, value) VALUES (?, ?, ?, ?)",
        long_rows,
    )
    conn.commit()


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


def list_tags(mount_root: str | Path = "/workspace/mount") -> list[str]:
    """Tags that have a dashboard DB, sorted."""
    tags_dir = Path(mount_root) / "tags"
    if not tags_dir.exists():
        return []
    return sorted(p.parent.name for p in tags_dir.glob("*/dashboard.db"))


def read_meta(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute("SELECT * FROM meta WHERE id = 0").fetchone()
    return dict(row) if row else None


def read_metric_names(conn: sqlite3.Connection) -> list[str]:
    return [r["name"] for r in conn.execute("SELECT DISTINCT name FROM metrics ORDER BY name")]


def read_metric_series(conn: sqlite3.Connection, name: str):
    """(epochs, values) arrays for one metric, ordered by epoch."""
    rows = conn.execute(
        "SELECT epoch, value FROM metrics WHERE name = ? ORDER BY epoch", (name,)
    ).fetchall()
    return np.array([r["epoch"] for r in rows]), np.array([r["value"] for r in rows])


def _epochs_in(conn: sqlite3.Connection, table: str) -> list[int]:
    return [r["epoch"] for r in conn.execute(f"SELECT epoch FROM {table} ORDER BY epoch")]


def read_monotonicity_epochs(conn: sqlite3.Connection) -> list[int]:
    return _epochs_in(conn, "monotonicity")


def read_all_monotonicity(conn: sqlite3.Connection):
    """All epochs stacked: (epochs, score_diffs (R,), win_rate (G,N,R), curve_scores (G,N,3))."""
    rows = conn.execute(
        "SELECT epoch, score_diffs, win_rate, curve_scores FROM monotonicity ORDER BY epoch"
    ).fetchall()
    if not rows:
        return [], None, None, None
    epochs = [r["epoch"] for r in rows]
    score_diffs = from_blob(rows[0]["score_diffs"])
    win_rate = np.stack([from_blob(r["win_rate"]) for r in rows])
    curve_scores = np.stack([from_blob(r["curve_scores"]) for r in rows])
    return epochs, score_diffs, win_rate, curve_scores


def read_all_score_belief(conn: sqlite3.Connection):
    """All epochs stacked: (epochs, score_diffs (R,), quantiles (Q,), bands (G,N,R,Q))."""
    rows = conn.execute(
        "SELECT epoch, score_diffs, quantiles, bands FROM score_belief ORDER BY epoch"
    ).fetchall()
    if not rows:
        return [], None, None, None
    epochs = [r["epoch"] for r in rows]
    score_diffs = from_blob(rows[0]["score_diffs"])
    quantiles = from_blob(rows[0]["quantiles"])
    bands = np.stack([from_blob(r["bands"]) for r in rows])
    return epochs, score_diffs, quantiles, bands


def read_throughput(conn: sqlite3.Connection) -> list[dict]:
    """All throughput samples in insertion order, each as a column->value dict."""
    rows = conn.execute(
        "SELECT t, positions, games, positions_per_s, producer_blocked_ns, "
        "consumer_blocked_ns, bottleneck FROM throughput ORDER BY rowid"
    ).fetchall()
    return [dict(r) for r in rows]


def read_train_steps(conn: sqlite3.Connection) -> dict:
    """All per-minibatch training stats as a dict of aligned arrays (pivoted from
    the long format). Always includes 'step' and 'positions'; every other key is
    a metric series aligned to 'step' order (missing values are NaN). Empty dict
    arrays when there is no data."""
    try:
        rows = conn.execute(
            "SELECT step, positions, name, value FROM train_step ORDER BY rowid"
        ).fetchall()
    except sqlite3.OperationalError:
        # A dashboard.db written before the long-format train_step schema. Treat
        # it as having no per-minibatch data (the view falls back to per-epoch).
        return {"step": np.array([]), "positions": np.array([])}
    steps: list[int] = []
    pos_by_step: dict[int, int] = {}
    series: dict[str, dict[int, float]] = {}
    for r in rows:
        s = r["step"]
        if s not in pos_by_step:
            pos_by_step[s] = r["positions"]
            steps.append(s)
        series.setdefault(r["name"], {})[s] = r["value"]
    out = {
        "step": np.array(steps, dtype=np.float64),
        "positions": np.array([pos_by_step[s] for s in steps], dtype=np.float64),
    }
    for name, by_step in series.items():
        out[name] = np.array([by_step.get(s, np.nan) for s in steps], dtype=np.float64)
    return out


def read_all_calibration(conn: sqlite3.Connection):
    """All epochs stacked into a dict of (G, ...) arrays, plus the epoch list.

    Keys: rel_edges (K+1,), rel_pred/rel_actual/rel_count (G,K), sd_edges (M+1,),
    sd_pred/sd_actual/sd_count (G,M).
    """
    rows = conn.execute("SELECT * FROM calibration ORDER BY epoch").fetchall()
    if not rows:
        return [], {}
    epochs = [r["epoch"] for r in rows]
    out = {"rel_edges": from_blob(rows[0]["rel_edges"]), "sd_edges": from_blob(rows[0]["sd_edges"])}
    for key in ("rel_pred", "rel_actual", "rel_count", "sd_pred", "sd_actual", "sd_count"):
        out[key] = np.stack([from_blob(r[key]) for r in rows])
    return epochs, out

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
  step INTEGER,                 -- cumulative minibatch index at this point's end
  positions INTEGER,            -- cumulative positions trained at this point's end
  name TEXT,                    -- metric name: 'loss', 'loss_<head>', '<x>_acc'
  value REAL,                   -- mean of the metric over the point's `n` minibatches
  n INTEGER                     -- minibatches this point aggregates (its weight/resolution)
);
CREATE TABLE IF NOT EXISTS loss_weights (
  name TEXT PRIMARY KEY,        -- a per-component loss series ('loss_<head>')
  weight REAL                   -- its coefficient in the optimized total loss
);
CREATE TABLE IF NOT EXISTS lane_pred (
  generation INTEGER,           -- checkpoint index this prediction was made at
  positions  INTEGER,           -- positions trained at that checkpoint (display label)
  position   INTEGER,           -- lane-analysis dataset position index
  occ        BLOB,              -- (30,15,27) uint8: thresholded predicted occupancy union
  score_pmf  BLOB,              -- (30,100) float32: predicted per-lane score distribution
  has_move   BLOB,              -- (30,) float32: predicted per-lane has-move probability
  PRIMARY KEY (generation, position)
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
    # The train_step read index references the current columns; create it
    # separately so a pre-schema train_step table (missing them) doesn't abort the
    # whole connect -- such a stale table is simply not read (read_train_steps
    # returns empty for it).
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS train_step_step ON train_step(name, step)")
    except sqlite3.OperationalError:
        pass
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


def write_lane_preds(conn: sqlite3.Connection, generation: int, positions: int, preds: dict):
    """Store one model generation's lane-analysis predictions over the dataset.

    `preds` holds per-position-stacked arrays: occ (N,30,15,27) uint8, score_pmf
    (N,30,100) float32, has_move (N,30) float32. One row per dataset position;
    re-recording a generation replaces it (idempotent on resume)."""
    occ, pmf, has = preds["occ"], preds["score_pmf"], preds["has_move"]
    rows = [
        (generation, positions, i, to_blob(occ[i]), to_blob(pmf[i]), to_blob(has[i]))
        for i in range(occ.shape[0])
    ]
    conn.executemany(
        "INSERT INTO lane_pred (generation, positions, position, occ, score_pmf, has_move) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(generation, position) DO UPDATE SET positions=excluded.positions, "
        "occ=excluded.occ, score_pmf=excluded.score_pmf, has_move=excluded.has_move",
        rows,
    )
    conn.commit()


def read_lane_generations(conn: sqlite3.Connection) -> list[dict]:
    """The recorded generations (checkpoints), each {generation, positions}, oldest
    first -- the dashboard's model slider scrubs over these."""
    return [
        {"generation": r["generation"], "positions": r["positions"]}
        for r in conn.execute(
            "SELECT generation, MAX(positions) AS positions FROM lane_pred "
            "GROUP BY generation ORDER BY generation"
        )
    ]


def read_lane_pred(conn: sqlite3.Connection, generation: int, position: int) -> dict | None:
    """One generation's prediction for one dataset position (occ / score_pmf /
    has_move arrays), or None if absent."""
    r = conn.execute(
        "SELECT occ, score_pmf, has_move FROM lane_pred WHERE generation=? AND position=?",
        (generation, position),
    ).fetchone()
    if r is None:
        return None
    return {"occ": from_blob(r["occ"]), "score_pmf": from_blob(r["score_pmf"]),
            "has_move": from_blob(r["has_move"])}


def write_monotonicity(conn: sqlite3.Connection, epoch: int, score_diffs, win_rate, curve_scores):
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
        "(epoch, rel_edges, rel_pred, rel_actual, rel_count, "
        "sd_edges, sd_pred, sd_actual, sd_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(epoch) DO UPDATE SET "
        "rel_edges=excluded.rel_edges, rel_pred=excluded.rel_pred, rel_actual=excluded.rel_actual, "
        "rel_count=excluded.rel_count, sd_edges=excluded.sd_edges, sd_pred=excluded.sd_pred, "
        "sd_actual=excluded.sd_actual, sd_count=excluded.sd_count",
        (
            epoch,
            to_blob(report.rel_bin_edges),
            to_blob(report.rel_bin_pred),
            to_blob(report.rel_bin_actual),
            to_blob(report.rel_bin_count),
            to_blob(report.sd_bin_edges),
            to_blob(report.sd_bin_pred),
            to_blob(report.sd_bin_actual),
            to_blob(report.sd_bin_count),
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


def _train_step_long_rows(rows: list[dict]) -> list[tuple]:
    """Expand `{step, positions, n, <metric>: value, ...}` point dicts into
    long-format `(step, positions, name, value, n)` tuples (one per metric). `n`
    is the point's weight (minibatches it aggregates); it defaults to 1. New
    loss/accuracy series need no schema change -- a metric is a present key."""
    return [
        (r["step"], r["positions"], name, float(value), int(r.get("n", 1)))
        for r in rows
        for name, value in r.items()
        if name not in ("step", "positions", "n")
    ]


def write_train_steps(conn: sqlite3.Connection, rows: list[dict]):
    """Append training-stat points (batched insert; no-op if empty). Append-only:
    points are never rewritten; the read aggregates them to a uniform resolution."""
    if not rows:
        return
    conn.executemany(
        "INSERT INTO train_step (step, positions, name, value, n) VALUES (?, ?, ?, ?, ?)",
        _train_step_long_rows(rows),
    )
    conn.commit()


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


def list_tags(mount_root: str | Path, task: str) -> list[str]:
    """Tags of one task that have a dashboard DB, sorted (tags/<task>/<tag>/)."""
    tags_dir = Path(mount_root) / "tags" / task
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


def read_train_steps(conn: sqlite3.Connection, max_points: int = 1024) -> dict:
    """Training stats rolled up to a single UNIFORM resolution for plotting.

    Points are stored append-only at a *gradient* of resolutions (older data finer,
    newer coarser; see TrainStepWriter). This rolls them all up to one uniform
    display resolution -- the smallest power-of-two block of `R` minibatches per
    point that yields <= max_points points (and never finer than the coarsest
    stored point) -- via a weight (`n`)-aware average, so the plot looks as if the
    data were stored uniformly. Returns a dict of aligned arrays: always 'step' and
    'positions', plus one series per metric (missing values NaN); empty when there
    is no data."""
    try:
        agg = conn.execute("SELECT MAX(step) AS hi, MAX(n) AS coarsest FROM train_step").fetchone()
    except sqlite3.OperationalError:
        # A dashboard.db written before this train_step schema; treat as no data.
        return {"step": np.array([]), "positions": np.array([])}
    if agg["hi"] is None:
        return {"step": np.array([]), "positions": np.array([])}

    # Display block size R: a power of two, large enough to fit in max_points and
    # at least the coarsest stored point (we can roll fine data up, not refine
    # coarse data down). Blocks align to multiples of R, and R is a multiple of
    # every (power-of-two) stored resolution, so no point straddles a block.
    need = max(agg["hi"] / max(max_points, 1), agg["coarsest"] or 1)
    r = 1
    while r < need:
        r *= 2

    rows = conn.execute(
        "SELECT name, (step - 1) / :r AS block, "
        "       SUM(value * n) / SUM(n) AS value, "  # weight-aware mean == true R-mean
        "       MAX(positions) AS positions, MAX(step) AS step "
        "FROM train_step GROUP BY name, block ORDER BY block",
        {"r": int(r)},
    ).fetchall()

    blocks: list[int] = []
    pos_by: dict[int, int] = {}
    step_by: dict[int, int] = {}
    series: dict[str, dict[int, float]] = {}
    for row in rows:
        b = row["block"]
        if b not in pos_by:
            blocks.append(b)
            pos_by[b], step_by[b] = row["positions"], row["step"]
        else:
            pos_by[b] = max(pos_by[b], row["positions"])
            step_by[b] = max(step_by[b], row["step"])
        series.setdefault(row["name"], {})[b] = row["value"]

    out = {
        "step": np.array([step_by[b] for b in blocks], dtype=np.float64),
        "positions": np.array([pos_by[b] for b in blocks], dtype=np.float64),
    }
    for name, by_block in series.items():
        out[name] = np.array([by_block.get(b, np.nan) for b in blocks], dtype=np.float64)
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

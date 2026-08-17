"""SQLite store for training metrics and eval artifacts.

A single per-tag database (``tags/<tag>/dashboard.db``) holds everything the
dashboard renders, so training writes data (never PNGs) and the Bokeh app
renders on the fly. WAL mode lets the dashboard read while training writes.

Tables:
  meta            one row of run config (args, model size, timestamps)
  metrics         long-format scalar series: (epoch, name) -> value
  match_eval      per-generation match-play result vs a fixed opponent

NumPy arrays are stored as ``np.save`` BLOBs (shape + dtype preserved).
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
CREATE TABLE IF NOT EXISTS position_eval_pred (
  generation INTEGER,           -- checkpoint index this prediction was made at
  positions  INTEGER,           -- positions trained at that checkpoint (display label)
  position   INTEGER,           -- position evaluation dataset position index
  wld        BLOB,              -- (3,) float32: model win/draw/loss probabilities
  sd_mean    REAL,              -- predicted final-score-delta mean (points)
  sd_std     REAL,              -- predicted final-score-delta std (points, Gaussian)
  PRIMARY KEY (generation, position)
);
CREATE TABLE IF NOT EXISTS match_eval (
  epoch INTEGER PRIMARY KEY,    -- generation index of the model under test
  positions INTEGER,            -- rows trained at that checkpoint (display label)
  opponent TEXT,                -- the opponent's --player spec
  games INTEGER, wins INTEGER, draws INTEGER, losses INTEGER,
  pair_counts TEXT,             -- JSON [5]: pentanomial pair-score counts
  score REAL,                   -- mean pair score (win rate, draws at 0.5)
  ci_half_width REAL,           -- pointwise CI half-width around score
  llr REAL, llr_lower REAL, llr_upper REAL,
  decision TEXT,                -- 'H0' | 'H1' | 'continue' (= budget-capped)
  elapsed_s REAL
);
CREATE TABLE IF NOT EXISTS match_arm (
  arm TEXT PRIMARY KEY,         -- arm name from the task's frozen arms param
  player_spec TEXT,             -- the arm's --player spec (player 0)
  opponent TEXT,                -- the fixed opponent's --player spec
  games INTEGER, wins INTEGER, draws INTEGER, losses INTEGER,
  pair_counts TEXT,             -- JSON [5]: pentanomial pair-score counts
  score REAL,                   -- mean pair score (win rate, draws at 0.5)
  ci_half_width REAL,           -- pointwise CI half-width around score
  elapsed_s REAL
);
CREATE TABLE IF NOT EXISTS control (
  name       TEXT PRIMARY KEY,  -- live operator knob (e.g. 'dataloader_workers')
  value      REAL,              -- its current value
  updated_at REAL               -- wall-clock of the last write
);
CREATE TABLE IF NOT EXISTS control_event (
  positions INTEGER,            -- rows-clock at which the trainer adopted the change
  name      TEXT,               -- which control changed
  value     REAL,               -- the new value
  t         REAL                -- wall-clock of the change
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


def init_control(conn: sqlite3.Connection, defaults: dict):
    """Seed control values that are not already present, so a restart keeps the
    operator's last-set values while a fresh run gets the configured defaults.
    Idempotent."""
    now = time.time()
    conn.executemany(
        "INSERT OR IGNORE INTO control (name, value, updated_at) VALUES (?, ?, ?)",
        [(name, float(v), now) for name, v in defaults.items()],
    )
    conn.commit()


def write_control(conn: sqlite3.Connection, name: str, value: float):
    """Upsert a live control value (the operator/dashboard write path)."""
    conn.execute(
        "INSERT INTO control (name, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (name, float(value), time.time()),
    )
    conn.commit()


def read_control(conn: sqlite3.Connection, name: str, default: float | None = None):
    """The current value of one control, or `default` if it has never been set."""
    row = conn.execute("SELECT value FROM control WHERE name = ?", (name,)).fetchone()
    return row["value"] if row is not None else default


def read_controls(conn: sqlite3.Connection) -> dict:
    """Every control's current value, name -> value."""
    return {r["name"]: r["value"] for r in conn.execute("SELECT name, value FROM control")}


def write_control_event(conn: sqlite3.Connection, positions: int, name: str, value: float):
    """Record that a control changed to `value` at `positions` rows trained, so the
    dashboard can annotate the metric curves where the operator intervened."""
    conn.execute(
        "INSERT INTO control_event (positions, name, value, t) VALUES (?, ?, ?, ?)",
        (int(positions), name, float(value), time.time()),
    )
    conn.commit()


def read_control_events(conn: sqlite3.Connection, name: str | None = None) -> list[dict]:
    """Control-change events (all, or for one control), oldest first."""
    sql = "SELECT positions, name, value, t FROM control_event"
    params: tuple = ()
    if name is not None:
        sql += " WHERE name = ?"
        params = (name,)
    sql += " ORDER BY positions"
    return [dict(r) for r in conn.execute(sql, params)]


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
    return {
        "occ": from_blob(r["occ"]),
        "score_pmf": from_blob(r["score_pmf"]),
        "has_move": from_blob(r["has_move"]),
    }


def write_position_eval_preds(
    conn: sqlite3.Connection, generation: int, positions: int, preds: dict
):
    """Store one model generation's position evaluation predictions over the dataset.

    `preds` holds per-position-stacked arrays: wld (N,3) float32, sd_mean (N,)
    float32, sd_std (N,) float32. One row per dataset position; re-recording a
    generation replaces it (idempotent on resume)."""
    wld, sd_mean, sd_std = preds["wld"], preds["sd_mean"], preds["sd_std"]
    rows = [
        (generation, positions, i, to_blob(wld[i]), float(sd_mean[i]), float(sd_std[i]))
        for i in range(wld.shape[0])
    ]
    conn.executemany(
        "INSERT INTO position_eval_pred (generation, positions, position, wld, sd_mean, sd_std) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(generation, position) DO UPDATE SET positions=excluded.positions, "
        "wld=excluded.wld, sd_mean=excluded.sd_mean, sd_std=excluded.sd_std",
        rows,
    )
    conn.commit()


def read_position_eval_generations(conn: sqlite3.Connection) -> list[dict]:
    """The recorded generations (checkpoints), each {generation, positions}, oldest
    first -- the dashboard's model slider scrubs over these."""
    return [
        {"generation": r["generation"], "positions": r["positions"]}
        for r in conn.execute(
            "SELECT generation, MAX(positions) AS positions FROM position_eval_pred "
            "GROUP BY generation ORDER BY generation"
        )
    ]


def read_position_eval_pred(
    conn: sqlite3.Connection, generation: int, position: int
) -> dict | None:
    """One generation's prediction for one dataset position (wld / sd_mean / sd_std),
    or None if absent."""
    r = conn.execute(
        "SELECT wld, sd_mean, sd_std FROM position_eval_pred WHERE generation=? AND position=?",
        (generation, position),
    ).fetchone()
    if r is None:
        return None
    return {
        "wld": from_blob(r["wld"]),
        "sd_mean": r["sd_mean"],
        "sd_std": r["sd_std"],
    }


def write_match_eval(conn: sqlite3.Connection, epoch: int, record: dict):
    """Store one generation's match-eval result (the match_eval runner's write
    path). Idempotent: replaying a generation's match replaces its row."""
    conn.execute(
        "INSERT INTO match_eval "
        "(epoch, positions, opponent, games, wins, draws, losses, pair_counts, "
        "score, ci_half_width, llr, llr_lower, llr_upper, decision, elapsed_s) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(epoch) DO UPDATE SET positions=excluded.positions, "
        "opponent=excluded.opponent, games=excluded.games, wins=excluded.wins, "
        "draws=excluded.draws, losses=excluded.losses, pair_counts=excluded.pair_counts, "
        "score=excluded.score, ci_half_width=excluded.ci_half_width, llr=excluded.llr, "
        "llr_lower=excluded.llr_lower, llr_upper=excluded.llr_upper, "
        "decision=excluded.decision, elapsed_s=excluded.elapsed_s",
        (
            epoch,
            int(record["positions"]),
            record["opponent"],
            int(record["games"]),
            int(record["wins"]),
            int(record["draws"]),
            int(record["losses"]),
            json.dumps(record["pair_counts"]),
            float(record["score"]),
            float(record["ci_half_width"]),
            float(record["llr"]),
            float(record["llr_lower"]),
            float(record["llr_upper"]),
            record["decision"],
            float(record["elapsed_s"]),
        ),
    )
    conn.commit()


def read_all_match_eval(conn: sqlite3.Connection) -> list[dict]:
    """Every match-eval row, oldest generation first, pair_counts decoded."""
    rows = [dict(r) for r in conn.execute("SELECT * FROM match_eval ORDER BY epoch")]
    for r in rows:
        r["pair_counts"] = json.loads(r["pair_counts"])
    return rows


def write_match_arm(conn: sqlite3.Connection, arm: str, record: dict):
    """Store one arm's finished match (the match_arms runner's write path).
    Idempotent: replaying an arm replaces its row."""
    conn.execute(
        "INSERT INTO match_arm "
        "(arm, player_spec, opponent, games, wins, draws, losses, pair_counts, "
        "score, ci_half_width, elapsed_s) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(arm) DO UPDATE SET player_spec=excluded.player_spec, "
        "opponent=excluded.opponent, games=excluded.games, wins=excluded.wins, "
        "draws=excluded.draws, losses=excluded.losses, pair_counts=excluded.pair_counts, "
        "score=excluded.score, ci_half_width=excluded.ci_half_width, "
        "elapsed_s=excluded.elapsed_s",
        (
            arm,
            record["player_spec"],
            record["opponent"],
            int(record["games"]),
            int(record["wins"]),
            int(record["draws"]),
            int(record["losses"]),
            json.dumps(record["pair_counts"]),
            float(record["score"]),
            float(record["ci_half_width"]),
            float(record["elapsed_s"]),
        ),
    )
    conn.commit()


def read_all_match_arms(conn: sqlite3.Connection) -> list[dict]:
    """Every arm row in insertion order (the experiment's declared arm order,
    since the runner measures arms in that order), pair_counts decoded."""
    rows = [dict(r) for r in conn.execute("SELECT * FROM match_arm ORDER BY rowid")]
    for r in rows:
        r["pair_counts"] = json.loads(r["pair_counts"])
    return rows


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

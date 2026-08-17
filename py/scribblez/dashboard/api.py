"""Tornado data API for the React dashboard.

Serves what the React dashboard renders: the tag list, a cheap per-tag change
token for polling, and each metrics figure as a Bokeh ``json_item`` (built by the
existing ``plots.py`` builders and embedded client-side with BokehJS). The React
app proxies ``/api`` to this server (see web/vite.config.ts), so the browser sees
one origin.

One server serves any run: the task and tag are request parameters, and each
request opens the matching per-tag ``dashboard.db``. Tornado is used because it is
already present (a Bokeh dependency), so the API needs no extra dependency.

See docs/react_dashboard.md for the architecture.
"""

import argparse
import fcntl
import json
import os
import signal
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tornado.ioloop
import tornado.web
from bokeh.embed import json_item

from scribblez import lane_analysis
from scribblez.dashboard import db, master_api, plots
from scribblez.dashboard.workers import WorkerManager
from scribblez.ffi import (
    analyze_gcg,
    analyze_position_eval_gcg,
    analyze_position_eval_gcg_leave,
    get_input_shapes,
    position_eval_board_json,
)
from scribblez.paths import TagPaths
from scribblez.position_eval import analysis as position_eval_analysis
from scribblez.position_eval.model import MASK_HEAD_NAMES

# The lane-union tile kinds in order: 26 letters then the collapsed blank.
_LANE_KINDS = [chr(ord("A") + k) for k in range(26)] + ["?"]

# The tables whose row counts form the per-tag change token the React shell polls
# (a change in any count means that tab's data advanced). Mirrors the Bokeh shell's
# per-tab ``watch()``.
# How often the WorkerManager closes desired-vs-actual worker-slot gaps.
RECONCILE_SECONDS = 30

VERSION_TABLES = (
    "metrics",
    "control_event",
    "match_eval",
    "match_arm",
)


def _loss(conn, params, mount_root):
    """The Loss tab's top panel: the loss/accuracy curves from the per-epoch
    metrics table, with control-change markers. The value-quality curves are a
    separate figure (`eval_quality`), so the client can place its own controls
    between the two."""
    return plots.metrics_loss_grid(conn, normalized=_truthy(params.get("normalized")))


def _eval_quality(conn, params, mount_root):
    """The Loss tab's aggregate model-vs-Monte-Carlo value-quality curves. `smooth`
    overlays an EMA trend; `secondary`, a second tag, overlays that tag's curves
    dashed for comparison."""
    secondary_tag = params.get("secondary") or None
    sec_conn = _open(mount_root, params.get("task"), secondary_tag) if secondary_tag else None
    try:
        secondary = (sec_conn, secondary_tag) if sec_conn is not None else None
        return plots.eval_quality_grid(
            conn, params.get("tag"), smooth=_truthy(params.get("smooth")), secondary=secondary
        )
    finally:
        if sec_conn is not None:
            sec_conn.close()


def _training_metrics(conn, params, mount_root):
    return plots.series_grid(conn, plots.TRAINING) if _row_count(conn, "metrics") else None


def _mset_metrics(conn, params, mount_root):
    """move_set_eval's Training tab: the generic training curves plus the
    teacher-value-regret quality figure."""
    groups = plots.TRAINING + plots.MSET_QUALITY
    return plots.series_grid(conn, groups) if _row_count(conn, "metrics") else None


def _evidence_metrics(conn, params, mount_root):
    """evidence_trajectories' Training tab: the generic training curves plus
    the conditioned-vs-plain quality figures."""
    groups = plots.TRAINING + plots.EVIDENCE_QUALITY
    return plots.series_grid(conn, groups) if _row_count(conn, "metrics") else None


def _match_eval(conn, params, mount_root):
    return plots.match_eval_grid(conn)


def _match_arms(conn, params, mount_root):
    return plots.match_arms_grid(conn)


# Figure name -> builder(conn, params, mount_root) -> Bokeh model | None.
# Reuses the plots.py builders; the model is serialized with json_item for client-
# side embedding. `mount_root` lets a builder open a second tag's DB (eval_quality's
# secondary-tag overlay).
FIGURES = {
    "loss": _loss,
    "eval_quality": _eval_quality,
    "training_metrics": _training_metrics,
    "mset_metrics": _mset_metrics,
    "evidence_metrics": _evidence_metrics,
    "match_eval": _match_eval,
    "match_arms": _match_arms,
}


def _truthy(value) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def version_token(conn: sqlite3.Connection) -> dict:
    """Per-tag row counts the client polls to decide when to re-fetch figures."""
    return {table: _row_count(conn, table) for table in VERSION_TABLES}


def build_figure_item(conn: sqlite3.Connection, name: str, params: dict, mount_root: str):
    """The Bokeh ``json_item`` dict for figure `name`, or None when there's no data
    (or no such figure). The handler turns None into ``{"item": null}``."""
    builder = FIGURES.get(name)
    if builder is None:
        return None
    model = builder(conn, params, mount_root)
    return json_item(model) if model is not None else None


def _open(mount_root: str, task: str, tag: str) -> sqlite3.Connection | None:
    """Open a tag's dashboard DB, or None if it doesn't exist yet (no spurious
    empty DB is created for an unknown tag)."""
    path = Path(TagPaths(tag, task, mount_root).dashboard_db)
    return db.connect(path) if path.exists() else None


# --- Lane-analysis endpoints -----------------------------------------------
# The position dataset and its ground truth (board + per-lane best moves) are fixed,
# so they are recomputed from the engine once and cached; only the per-generation
# model predictions come from the tag's DB. The score head has 100 bins, with the
# top bin a catch-all for scores >= 99 (matching the C++ label encoding).
_TOP_SCORE_BIN = 99


@lru_cache(maxsize=1)
def _dataset_files() -> tuple:
    return tuple(lane_analysis.dataset_gcgs(lane_analysis.DEFAULT_DATASET))


@lru_cache(maxsize=64)
def _ground_truth(position: int) -> tuple:
    """(name, bundle) for a dataset position -- the engine analysis bundle (board +
    per-lane ground truth). Cached: it is fixed for the dataset."""
    gcg = _dataset_files()[position]
    bundle, _input = analyze_gcg(gcg.read_text())
    return gcg.stem, bundle


def _pred_placed(occ_lane: np.ndarray) -> list:
    """A predicted lane union (15, 27) -> per-cell letter lists, matching the
    ground-truth `placed` shape so the UI can diff them cell by cell."""
    return [[_LANE_KINDS[k] for k in range(27) if occ_lane[c, k]] for c in range(15)]


def _merge_lane(gt: dict, occ_lane, pmf_lane, has_lane) -> dict:
    """A lane's ground truth, plus (when a prediction exists) the model's predicted
    union / score distribution / has-move and the server-computed correctness."""
    o = dict(gt)
    if occ_lane is None:
        return o
    pred_placed = _pred_placed(occ_lane)
    pred_bin = int(np.argmax(pmf_lane))
    o["pred_placed"] = pred_placed
    o["pred_score_bin"] = pred_bin
    o["pred_score_pmf"] = [float(x) for x in pmf_lane]
    o["pred_has_move"] = float(has_lane)
    if gt["has_move"]:
        o["move_correct"] = all(
            set(a) == set(b) for a, b in zip(gt["placed"], pred_placed, strict=True)
        )
        o["score_correct"] = pred_bin == min(gt["max_score"], _TOP_SCORE_BIN)
    return o


def _merge_axis(gt_axis, occ, pmf, has, base: int) -> list:
    return [
        _merge_lane(
            gt_axis[i],
            None if occ is None else occ[base + i],
            None if pmf is None else pmf[base + i],
            None if has is None else has[base + i],
        )
        for i in range(15)
    ]


def lane_position_payload(conn, position: int, generation) -> dict:
    """The full per-position view: board + rack for rendering, and each lane's
    ground truth merged with the selected generation's prediction."""
    name, bundle = _ground_truth(position)
    pred = (
        db.read_lane_pred(conn, generation, position)
        if (conn is not None and generation is not None)
        else None
    )
    occ = pred["occ"] if pred else None
    pmf = pred["score_pmf"] if pred else None
    has = pred["has_move"] if pred else None
    la = bundle["lane_analysis"]
    return {
        "name": name,
        "on_move": bundle["on_move"],
        "board": bundle["board"],
        "bonuses": bundle["bonuses"],
        "rack": bundle["rack"],
        "tile_scores": bundle["tile_scores"],
        "generation": generation,
        "has_prediction": pred is not None,
        "lanes": {
            "rows": _merge_axis(la["rows"], occ, pmf, has, 0),
            "cols": _merge_axis(la["cols"], occ, pmf, has, 15),
        },
    }


@lru_cache(maxsize=1)
def _position_eval_dataset_files() -> tuple:
    return tuple(position_eval_analysis.dataset_gcgs(position_eval_analysis.DEFAULT_DATASET))


@lru_cache(maxsize=64)
def _position_eval_board(position: int) -> tuple:
    """(name, board_bundle) for a position evaluation dataset position -- board / bonuses / leave
    rack for rendering. Cached: it is fixed for the dataset."""
    gcg = _position_eval_dataset_files()[position]
    return gcg.stem, position_eval_board_json(gcg.read_text())


@lru_cache(maxsize=1)
def _mc_ground_truth() -> dict:
    """The committed Monte-Carlo ground truth, keyed by position name (pos-1, ...)."""
    path = position_eval_analysis.DEFAULT_DATASET / position_eval_analysis.GROUND_TRUTH_FILENAME
    return json.loads(path.read_text()) if path.exists() else {}


def _mc_payload(name: str) -> dict:
    """A position's Monte-Carlo ground truth shaped for the UI: W/L/D as fractions, the
    exact score-delta histogram as sorted [delta, count] pairs, and the mean delta (the
    UI derives the std from the histogram)."""
    gt = _mc_ground_truth().get(name, {})
    n = gt.get("n", 0)
    wld = gt.get("wld", {})
    hist = sorted((int(d), c) for d, c in gt.get("score_delta_hist", {}).items())
    total = sum(c for _d, c in hist) or 1
    return {
        "n": n,
        "wld": {k: (wld.get(k, 0) / n if n else 0.0) for k in ("win", "loss", "draw")},
        "score_delta_hist": hist,
        "score_delta_mean": sum(d * c for d, c in hist) / total,
    }


def _placement_block(name: str, pred) -> dict | None:
    """The Positions tab's residual-heat-map payload: for each of the four placement
    heads, the Monte-Carlo truth (per-square rollout fraction count/n, board frame)
    paired with the model's on-demand sigmoid prediction (`pred`, or None).

    None when the ground-truth file carries no per-square planes (an older
    monte-carlo-sim-results.json), so the frontend can hide the overlay."""
    gt = _mc_ground_truth().get(name, {})
    planes = gt.get("placement")
    if planes is None:
        return None
    n = gt.get("n", 0)
    denom = n or 1
    heads = {}
    for head in MASK_HEAD_NAMES:
        truth = (np.asarray(planes[head], dtype=np.float64) / denom).tolist()
        heads[head] = {"truth": truth, "pred": None if pred is None else pred[head].tolist()}
    return {"n": n, "heads": heads}


def position_eval_position_payload(conn, position: int, generation, tag, task, mount_root) -> dict:
    """The full per-position view: board + leave for rendering, the Monte-Carlo ground
    truth, and the selected generation's model prediction (WLD + score-delta Gaussian)
    -- or None when no prediction exists for this generation/position. The `placement`
    block pairs the per-square Monte-Carlo truth with the model's on-demand
    placement-head predictions for the residual heat-map overlay."""
    name, bundle = _position_eval_board(position)
    pred = (
        db.read_position_eval_pred(conn, generation, position)
        if (conn is not None and generation is not None)
        else None
    )
    model = None
    if pred is not None:
        w = pred["wld"]  # (3,) in [win, draw, loss] order
        model = {
            "wld": {"win": float(w[0]), "draw": float(w[1]), "loss": float(w[2])},
            "sd_mean": float(pred["sd_mean"]),
            "sd_std": float(pred["sd_std"]),
        }
    placement_pred = _position_eval_placement_pred(tag, task, mount_root, generation, position)
    return {
        "name": name,
        "start_player": bundle["start_player"],
        "last_move": bundle["last_move"],
        "board": bundle["board"],
        "bonuses": bundle["bonuses"],
        "rack": bundle["rack"],
        "tile_scores": bundle["tile_scores"],
        "scores": bundle["scores"],
        "bag_count": bundle["bag_count"],
        "opponent_rack_count": bundle["opponent_rack_count"],
        "generation": generation,
        "has_prediction": pred is not None,
        "mc": _mc_payload(name),
        "model": model,
        "placement": _placement_block(name, placement_pred),
    }


def _resolve_position_eval_generation(conn, arg: str):
    """Resolve a generation query arg: a specific index, or the newest recorded one for
    '', 'latest', or None (or None when nothing is recorded)."""
    if conn is not None and arg in ("", "latest"):
        gens = db.read_position_eval_generations(conn)
        return gens[-1]["generation"] if gens else None
    return int(arg) if arg not in ("", "latest") else None


# Per-(file, mtime) ONNX session cache. The alternate-leave what-if runs the selected
# generation's exported model on demand; fp32 onnxruntime reproduces the torch model
# the stored predictions came from, so the two are directly comparable.
_ONNX_SESSIONS: dict = {}


def _position_eval_onnx_session(onnx_path: Path):
    key = (str(onnx_path), onnx_path.stat().st_mtime)
    sess = _ONNX_SESSIONS.get(key)
    if sess is None:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _ONNX_SESSIONS[key] = sess
    return sess


def _position_eval_onnx_feed(sess, flat_input: np.ndarray) -> dict:
    """The ``{input_spatial, input_scalar}`` feed for `sess` from the dashboard
    session's full flat input row.

    The model declares its own arm in its ONNX metadata_props
    ("contingent_features", stamped at export; absent means off). A baseline model
    consumes the base layout, whose spatial planes and scalars are prefixes of the
    full blocks, so its declared input widths select the right slices."""
    meta = sess.get_modelmeta().custom_metadata_map
    contingent = meta.get("contingent_features") == "true"
    model_inputs = {i.name: i.shape for i in sess.get_inputs()}
    planes = int(model_inputs["input_spatial"][1])
    scalars = int(model_inputs["input_scalar"][1])
    full_planes = {s.name: s.dims for s in get_input_shapes()}["input_spatial"][0]
    if contingent != (planes == full_planes):
        raise ValueError("ONNX metadata arm disagrees with the declared input widths")
    cells = position_eval_analysis.BOARD_SIZE**2
    spatial = flat_input[: planes * cells].reshape(1, planes, 15, 15).astype(np.float32)
    scalar = flat_input[full_planes * cells :][:scalars].reshape(1, -1).astype(np.float32)
    return {"input_spatial": spatial, "input_scalar": scalar}


def _position_eval_onnx_fits_encoder(sess) -> bool:
    """True iff `sess`'s declared input widths fit within the dashboard session's
    current encoder row -- its planes and scalars are prefixes of the full blocks. A
    model exported under an earlier, differently sized encoding does not fit and
    cannot be run on today's inputs."""
    model_inputs = {i.name: i.shape for i in sess.get_inputs()}
    full = {s.name: s.dims for s in get_input_shapes()}
    return (
        int(model_inputs["input_spatial"][1]) <= full["input_spatial"][0]
        and int(model_inputs["input_scalar"][1]) <= full["input_scalar"][0]
    )


def _run_position_eval_onnx(onnx_path: Path, flat_input: np.ndarray) -> dict:
    """Run the exported post-move model on one flat input tensor and decode the value
    outputs: W/L/D probabilities and the score-delta mean/std."""
    sess = _position_eval_onnx_session(onnx_path)
    wld, sd = sess.run(["wld", "score_diff"], _position_eval_onnx_feed(sess, flat_input))
    probs = np.exp(wld[0] - wld[0].max())
    probs /= probs.sum()
    return {
        "wld": {"win": float(probs[0]), "draw": float(probs[1]), "loss": float(probs[2])},
        "sd_mean": float(sd[0, 0]),
        "sd_std": float(sd[0, 1]),
    }


def _run_position_eval_masks(onnx_path: Path, flat_input: np.ndarray) -> dict:
    """The exported model's four placement-mask heads for one flat input, as
    board-frame (15, 15) sigmoid-probability arrays keyed by head name.

    The analysis encoder never flips the board (apply_flip=False in
    position_eval_analysis.cpp), so the convolutional mask heads emit their squares
    in the same row/col frame as the board -- the frame the Monte-Carlo ground-truth
    planes use -- and need no transpose."""
    sess = _position_eval_onnx_session(onnx_path)
    feed = _position_eval_onnx_feed(sess, flat_input)
    outs = sess.run(list(MASK_HEAD_NAMES), feed)
    return {
        name: 1.0 / (1.0 + np.exp(-out[0])) for name, out in zip(MASK_HEAD_NAMES, outs, strict=True)
    }


@lru_cache(maxsize=256)
def _position_eval_placement_pred_for_path(onnx_path_str: str, position: int):
    """The mask predictions for one on-disk ONNX file + dataset position: board-frame
    sigmoid (15, 15) arrays keyed by head name, or None when the file's declared input
    widths don't fit the dashboard session's current encoder (an earlier, differently
    sized encoding era).

    Cached per (onnx_path, position): an ONNX export is atomic (a temp file renamed
    into place), so a path that exists is always complete and its content never
    changes -- the memoized planes never go stale. The fits-encoder=False result is
    likewise permanent for that path, so caching it is fine too."""
    onnx_path = Path(onnx_path_str)
    if not _position_eval_onnx_fits_encoder(_position_eval_onnx_session(onnx_path)):
        return None
    gcg = _position_eval_dataset_files()[position]
    return _run_position_eval_masks(onnx_path, analyze_position_eval_gcg(gcg.read_text()))


def _position_eval_placement_pred(tag, task, mount_root, generation, position):
    """The selected generation's ONNX placement-mask predictions for a dataset
    position: board-frame sigmoid (15, 15) arrays keyed by head name, or None when
    the generation has no exported ONNX (or one from an incompatible encoding era).

    File existence is checked uncached on every call: memoizing a miss would pin a
    null prediction to the generation even if its export appears later. Only once
    the file exists is the (cacheable) lookup in
    `_position_eval_placement_pred_for_path` consulted."""
    if generation is None:
        return None
    onnx_path = TagPaths(tag, task, mount_root).onnx_path(generation)
    if not onnx_path.exists():
        return None
    return _position_eval_placement_pred_for_path(str(onnx_path), position)


class _Base(tornado.web.RequestHandler):
    @property
    def mount_root(self) -> str:
        return self.settings["mount_root"]

    def _params(self) -> dict:
        return {k: self.get_query_argument(k) for k in self.request.query_arguments}

    def _open_conn(self) -> sqlite3.Connection | None:
        return _open(
            self.mount_root, self.get_query_argument("task"), self.get_query_argument("tag")
        )


class TagsHandler(_Base):
    def get(self):
        task = self.get_query_argument("task")
        self.write({"tags": db.list_tags(self.mount_root, task)})


class VersionHandler(_Base):
    def get(self):
        conn = self._open_conn()
        if conn is None:
            self.set_status(404)
            self.write({"error": "unknown tag"})
            return
        try:
            self.write(version_token(conn))
        finally:
            conn.close()


def _meta_payload(meta: dict) -> dict:
    """The Info tab's view of a run's meta row: the run config (parsed from the
    stored args JSON), model size, and timestamps."""
    args = json.loads(meta["args_json"]) if meta.get("args_json") else {}
    return {
        "tag": meta.get("tag"),
        "model_params": meta.get("model_params"),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "args": args,
    }


class MetaHandler(_Base):
    """A run's recorded config (training args), model size, and timestamps."""

    def get(self):
        conn = self._open_conn()
        if conn is None:
            self.write({"meta": None})
            return
        try:
            meta = db.read_meta(conn)
        finally:
            conn.close()
        self.write({"meta": _meta_payload(meta) if meta else None})


class ControlsHandler(_Base):
    """Live operator controls (e.g. dataloader_workers). GET returns the current
    values and the rows-clock change events (which also carry the LR schedule's
    phase boundaries); POST {name, value} sets one, which the trainer adopts at
    its next generation. Values persist in the tag's dashboard.db."""

    def get(self):
        conn = self._open_conn()
        if conn is None:
            self.write({"controls": {}, "events": []})
            return
        try:
            self.write({"controls": db.read_controls(conn), "events": db.read_control_events(conn)})
        finally:
            conn.close()

    def post(self):
        body = json.loads(self.request.body or b"{}")
        name, value = body.get("name"), body.get("value")
        if not isinstance(name, str) or not name or isinstance(value, bool):
            self.set_status(400)
            self.write({"error": "expected {name: non-empty str, value: number}"})
            return
        if not isinstance(value, (int, float)):
            self.set_status(400)
            self.write({"error": "value must be a number"})
            return
        # Open (creating if needed) directly rather than via the exists-gated
        # _open_conn, so a control can be set before the first training run.
        path = Path(
            TagPaths(
                self.get_query_argument("tag"), self.get_query_argument("task"), self.mount_root
            ).dashboard_db
        )
        conn = db.connect(path)
        try:
            db.write_control(conn, name, float(value))
            self.write({"controls": db.read_controls(conn)})
        finally:
            conn.close()


class FigureHandler(_Base):
    def get(self, name: str):
        if name not in FIGURES:
            self.set_status(404)
            self.write({"error": f"unknown figure {name!r}"})
            return
        conn = self._open_conn()
        if conn is None:
            self.set_status(404)
            self.write({"error": "unknown tag"})
            return
        try:
            item = build_figure_item(conn, name, self._params(), self.mount_root)
            self.write({"item": item})
        finally:
            conn.close()


class LanePositionsHandler(_Base):
    """The lane-analysis dataset's positions (the UI's position selector)."""

    def get(self):
        try:
            self.write({"positions": [gcg.stem for gcg in _dataset_files()]})
        except OSError:
            self.write({"positions": []})


class LaneGenerationsHandler(_Base):
    """The model generations a tag has lane-analysis predictions for (the slider)."""

    def get(self):
        conn = self._open_conn()
        if conn is None:
            self.write({"generations": []})
            return
        try:
            self.write({"generations": db.read_lane_generations(conn)})
        finally:
            conn.close()


class LanePositionHandler(_Base):
    """Board + per-lane ground truth merged with one generation's prediction.
    `generation` may be omitted or 'latest' to use the newest recorded one."""

    def get(self):
        files = _dataset_files()
        position = int(self.get_query_argument("position", "0"))
        if not 0 <= position < len(files):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        conn = self._open_conn()
        try:
            generation = self._resolve_generation(conn)
            self.write(lane_position_payload(conn, position, generation))
        except OSError:  # lexicon unavailable -> can't build ground truth
            self.set_status(503)
            self.write({"error": "lexicon unavailable; cannot compute ground truth"})
        finally:
            if conn is not None:
                conn.close()

    def _resolve_generation(self, conn):
        arg = self.get_query_argument("generation", "latest")
        if conn is not None and arg in ("", "latest"):
            gens = db.read_lane_generations(conn)
            return gens[-1]["generation"] if gens else None
        return int(arg) if arg not in ("", "latest") else None


class PositionEvalPositionsHandler(_Base):
    """The position evaluation dataset's positions (the UI's position selector)."""

    def get(self):
        try:
            self.write({"positions": [gcg.stem for gcg in _position_eval_dataset_files()]})
        except OSError:
            self.write({"positions": []})


class PositionEvalGenerationsHandler(_Base):
    """The model generations a tag has position evaluation predictions for (the slider)."""

    def get(self):
        conn = self._open_conn()
        if conn is None:
            self.write({"generations": []})
            return
        try:
            self.write({"generations": db.read_position_eval_generations(conn)})
        finally:
            conn.close()


class PositionEvalPositionHandler(_Base):
    """Board + Monte-Carlo ground truth merged with one generation's prediction.
    `generation` may be omitted or 'latest' to use the newest recorded one."""

    def get(self):
        files = _position_eval_dataset_files()
        position = int(self.get_query_argument("position", "0"))
        if not 0 <= position < len(files):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        conn = self._open_conn()
        try:
            generation = _resolve_position_eval_generation(
                conn, self.get_query_argument("generation", "latest")
            )
            self.write(
                position_eval_position_payload(
                    conn,
                    position,
                    generation,
                    self.get_query_argument("tag"),
                    self.get_query_argument("task"),
                    self.mount_root,
                )
            )
        except OSError:  # engine unavailable -> can't build the board
            self.set_status(503)
            self.write({"error": "engine unavailable; cannot build board"})
        finally:
            if conn is not None:
                conn.close()


class PositionEvalAltLeaveHandler(_Base):
    """Evaluate the selected generation's model on a position with an alternate leave (a
    what-if). Query: position, generation, leave. Returns the model's W/L/D + score-delta
    mean/std, or a 400 with a human-readable reason for an invalid/unavailable leave."""

    def get(self):
        files = _position_eval_dataset_files()
        position = int(self.get_query_argument("position", "0"))
        if not 0 <= position < len(files):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        leave = self.get_query_argument("leave", "").strip()
        conn = self._open_conn()
        try:
            generation = _resolve_position_eval_generation(
                conn, self.get_query_argument("generation", "latest")
            )
        finally:
            if conn is not None:
                conn.close()
        if generation is None:
            self.set_status(404)
            self.write({"error": "no model generations recorded yet"})
            return
        try:
            inp = analyze_position_eval_gcg_leave(files[position].read_text(), leave)
        except ValueError as e:  # bad size / unavailable tiles -> show the reason
            self.set_status(400)
            self.write({"error": str(e)})
            return
        except OSError:
            self.set_status(503)
            self.write({"error": "engine unavailable; cannot encode position"})
            return
        onnx_path = TagPaths(
            self.get_query_argument("tag"), self.get_query_argument("task"), self.mount_root
        ).onnx_path(generation)
        if not onnx_path.exists():
            self.set_status(404)
            self.write({"error": f"model for generation {generation} is not available"})
            return
        self.write(
            {
                "leave": leave,
                "generation": generation,
                "model": _run_position_eval_onnx(onnx_path, inp),
            }
        )


def make_app(mount_root: str, worker_manager=None) -> tornado.web.Application:
    """The full API app: the read-only training data plane plus (when a
    WorkerManager is supplied) the master dashboard's control plane."""
    return tornado.web.Application(
        [
            *master_api.MASTER_ROUTES,
            (r"/api/tags", TagsHandler),
            (r"/api/version", VersionHandler),
            (r"/api/meta", MetaHandler),
            (r"/api/controls", ControlsHandler),
            (r"/api/figure/([a-z_]+)", FigureHandler),
            (r"/api/lane/positions", LanePositionsHandler),
            (r"/api/lane/generations", LaneGenerationsHandler),
            (r"/api/lane/position", LanePositionHandler),
            (r"/api/position_eval/positions", PositionEvalPositionsHandler),
            (r"/api/position_eval/generations", PositionEvalGenerationsHandler),
            (r"/api/position_eval/position", PositionEvalPositionHandler),
            (r"/api/position_eval/alt_leave", PositionEvalAltLeaveHandler),
        ],
        mount_root=mount_root,
        worker_manager=worker_manager,
    )


# Held for the process lifetime so the flock is not dropped by garbage
# collection; the kernel releases it automatically when the process exits.
_CONTROL_LOCK = None


def _acquire_control_lock(mount_root: str):
    """Guarantee a single dashboard control plane per mount root. The
    WorkerManager owns local worker processes and reconciles every task's slots
    against its task.json; two dashboards on the same mount would fight over the
    same slots -- double-spawning local workers and issuing conflicting
    pause/gate enforcement. An exclusive advisory lock on <mount>/.dashboard.lock
    serializes them. Because flock is released by the kernel when its holder
    dies, a crashed dashboard never leaves a stale lock behind; only a live one
    blocks a second start. Port reclaim (react_server) only dedups a single
    port, so a dashboard on another port would otherwise slip through."""
    global _CONTROL_LOCK
    path = Path(mount_root) / ".dashboard.lock"
    path.touch(exist_ok=True)
    fd = open(path, "r+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = fd.read().strip() or "unknown"
        sys.exit(
            f"A dashboard is already managing {mount_root} (pid {holder}). Stop it first, "
            f"or point this one at a different --mount-root."
        )
    fd.seek(0)
    fd.truncate()
    fd.write(str(os.getpid()))
    fd.flush()
    _CONTROL_LOCK = fd


def run(port: int, mount_root: str):
    """Serve the API on `port` until SIGTERM/interrupt (used by the dashboard
    launcher). Binds to localhost only: the control plane holds cloud
    credentials and launches processes, so it must not be reachable
    off-machine (the browser reaches it through the Vite /api proxy).

    Refuses to start if another dashboard already manages this mount root
    (a single control plane owns the local workers). The WorkerManager
    reconciles worker slots at boot (relaunching local workers that should be
    running) and every RECONCILE_SECONDS thereafter (restarting interruptible
    pods Runpod reclaimed). On shutdown, owned local workers get SIGTERM (they
    flush and exit); cloud pods keep running.
    """
    _acquire_control_lock(mount_root)
    manager = WorkerManager()
    make_app(mount_root, manager).listen(port, address="127.0.0.1")
    loop = tornado.ioloop.IOLoop.current()

    def reconcile():
        try:
            manager.reconcile()
        except Exception as e:  # noqa: BLE001 -- reconciliation must keep ticking
            print(f"reconcile: {e}")

    def stop(signum, frame):
        loop.add_callback_from_signal(loop.stop)

    signal.signal(signal.SIGTERM, stop)
    loop.add_callback(reconcile)
    tornado.ioloop.PeriodicCallback(reconcile, RECONCILE_SECONDS * 1000).start()
    try:
        loop.start()
    finally:
        manager.shutdown()


def main():
    p = argparse.ArgumentParser(description="Serve the React dashboard's data API.")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--mount-root", default="/workspace/mount")
    args = p.parse_args()
    run(args.port, args.mount_root)


if __name__ == "__main__":
    main()

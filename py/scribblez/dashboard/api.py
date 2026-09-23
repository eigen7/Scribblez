"""The dashboard's Tornado server: the training data plane, and the process
that hosts the control plane (master_api.py, workers.py).

The data plane serves what the React app renders for a tag: the tag list, a
cheap change token for polling, run meta and live controls, each metrics
figure as a Bokeh ``json_item`` (plots.py) with incremental updates
(figure_delta.py), and the Lane analysis and Positions tabs, whose predictions
come from the DB or are computed on demand from a generation's ONNX export.

One server serves every run: task and tag are request parameters, and each
request opens that tag's dashboard.db. The React app reaches it through Vite's
``/api`` proxy (web/vite.config.ts), so the browser sees one origin. Tornado
comes with Bokeh, so the API adds no dependency.

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

from scribblez import lane_analysis, workloads
from scribblez import params as params_mod
from scribblez.dashboard import db, figure_delta, master_api, plots, tasks, trajectories_api
from scribblez.dashboard.workers import WorkerManager
from scribblez.ffi import (
    InputArm,
    analyze_gcg,
    analyze_position_eval_gcg,
    analyze_position_eval_gcg_leaves,
    collapse_position_eval_placement,
    legal_position_eval_placement,
    position_eval_board_json,
)
from scribblez.generational.records import write_controls_file
from scribblez.paths import TagPaths
from scribblez.position_eval import analysis as position_eval_analysis
from scribblez.position_eval.model import PLACEMENT_HEAD_NAMES
from scribblez.workloads.position_eval import PositionEvalParams

# The lane-union tile kinds in order: 26 letters then the collapsed blank.
_LANE_KINDS = [chr(ord("A") + k) for k in range(26)] + ["?"]

# How often the reconcile pass runs. It is the only observer of ssh containers
# and rented machines, and the only thing that acts on a scheduler gate, so its
# period is also how long a released worker waits before resuming. Every step
# runs off the event loop (WorkerManager.offload), which is what makes a period
# this short affordable.
RECONCILE_SECONDS = 5

# The tables whose row counts form the per-tag change token the React app polls:
# a change in any count means some tab's data advanced.
VERSION_TABLES = (
    "metrics",
    "control_event",
    "match_eval",
    "match_arm",
)


def _loss(conn, params, mount_root):
    """The Loss tab's top panel. The value-quality curves are a separate figure
    (`eval_quality`) so the client can place its own controls between the two."""
    return plots.metrics_loss_grid(conn)


def _eval_quality(conn, params, mount_root):
    """The Loss tab's value-quality panel. `secondary` names a second tag whose
    curves are overlaid for comparison."""
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
    """move_set_eval's Training tab: the generic curves plus distillation quality."""
    groups = plots.TRAINING + plots.MSET_QUALITY
    return plots.series_grid(conn, groups) if _row_count(conn, "metrics") else None


def _evidence_metrics(conn, params, mount_root):
    """evidence_trajectories' Training tab: the generic curves plus the
    conditioned-vs-plain quality figures."""
    groups = plots.TRAINING + plots.EVIDENCE_QUALITY
    return plots.series_grid(conn, groups) if _row_count(conn, "metrics") else None


def _match_eval(conn, params, mount_root):
    return plots.match_eval_grid(conn)


def _match_arms(conn, params, mount_root):
    return plots.match_arms_grid(conn)


# Figure name -> builder(conn, params, mount_root) -> Bokeh model | None, where
# `params` are the request's query arguments. `mount_root` lets a builder open a
# second tag's DB (eval_quality's overlay).
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
    """Per-tag row counts the client polls to decide when to refetch figures."""
    return {table: _row_count(conn, table) for table in VERSION_TABLES}


def build_figure_item(conn: sqlite3.Connection, name: str, params: dict, mount_root: str):
    """(json_item, structure key) for figure `name`, or (None, None) when it has
    no data. The client echoes the structure key in its incremental-update
    requests (figure_delta.py)."""
    builder = FIGURES.get(name)
    if builder is None:
        return None, None
    model = builder(conn, params, mount_root)
    if model is None:
        return None, None
    return json_item(model), figure_delta.structure_key(model)


def _open(mount_root: str, task: str, tag: str) -> sqlite3.Connection | None:
    """Open a tag's dashboard DB, or None if it does not exist yet. Never creates
    one, so a request for an unknown tag leaves nothing behind."""
    path = Path(TagPaths(tag, task, mount_root).dashboard_db)
    return db.connect(path) if path.exists() else None


# --- Lane analysis tab (max_move_per_lane) ---------------------------------
# The position dataset and its ground truth (board + per-lane best moves) are
# fixed, so they are computed by the engine once and cached; only the
# per-generation predictions come from the tag's DB.

# The score head's last bin, a catch-all for scores >= 99 (as in the C++ label
# encoding).
_TOP_SCORE_BIN = 99


@lru_cache(maxsize=1)
def _dataset_files() -> tuple:
    return tuple(lane_analysis.dataset_gcgs(lane_analysis.DEFAULT_DATASET))


@lru_cache(maxsize=64)
def _ground_truth(position: int) -> tuple:
    """(name, engine analysis bundle) for a dataset position: the board and its
    per-lane ground truth."""
    gcg = _dataset_files()[position]
    bundle, _input = analyze_gcg(gcg.read_text())
    return gcg.stem, bundle


def _pred_placed(occ_lane: np.ndarray) -> list:
    """A predicted lane union (15, 27) as per-cell letter lists, the shape of the
    ground truth's `placed`, so the UI can diff them cell by cell."""
    return [[_LANE_KINDS[k] for k in range(27) if occ_lane[c, k]] for c in range(15)]


def _merge_lane(gt: dict, occ_lane, pmf_lane, has_lane) -> dict:
    """A lane's ground truth, plus the model's prediction and its correctness
    when a prediction exists."""
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
    """The Lane analysis tab's view of one position: board and rack, and each
    lane's ground truth merged with the selected generation's prediction."""
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


# --- Positions tab (position_eval) -----------------------------------------
# Board, legality and Monte-Carlo truth depend only on the position and are
# cached per file; predictions are computed on demand from each generation's
# ONNX export.


def _position_eval_dataset_files() -> tuple:
    """The position_eval dataset's GCG files in natural order. Listed afresh on
    every call: the set is hand-maintained and edited in place, and a running
    dashboard should follow it without a restart."""
    return tuple(position_eval_analysis.dataset_gcgs(position_eval_analysis.DEFAULT_DATASET))


def _position_eval_gcg_key(position: int) -> tuple[str, int]:
    """A dataset position's cache key for the memos below: its file's path and
    mtime, so a file rewritten in place misses rather than hitting stale."""
    gcg = _position_eval_dataset_files()[position]
    return str(gcg), gcg.stat().st_mtime_ns


def _position_eval_board(position: int) -> tuple:
    return _position_eval_board_for(_position_eval_gcg_key(position))


@lru_cache(maxsize=64)
def _position_eval_board_for(gcg_key: tuple[str, int]) -> tuple:
    """(name, board bundle for rendering) for a _position_eval_gcg_key."""
    gcg = Path(gcg_key[0])
    return gcg.stem, position_eval_board_json(gcg.read_text())


def _position_eval_face_up_leaves(task: str, tag: str) -> bool:
    """The information condition a position_eval tag trains under (its frozen
    `face_up_leaves` param). It decides which Monte-Carlo truth the tag is
    measured against and how much of the opponent's rack the Positions tab
    shows. KeyError for a tag with no task.json."""
    record = tasks.load_task(workloads.get(task), tag)
    if record is None:
        raise KeyError(f"tag {tag!r} has no task.json")
    return params_mod.validate(PositionEvalParams, record.params).face_up_leaves


@lru_cache(maxsize=2)
def _mc_ground_truth(face_up_leaves: bool) -> dict:
    """The committed Monte-Carlo ground truth for an information condition,
    keyed by position name."""
    path = position_eval_analysis.ground_truth_path(
        position_eval_analysis.DEFAULT_DATASET, face_up_leaves
    )
    return json.loads(path.read_text()) if path.exists() else {}


def _mc_payload(name: str, face_up_leaves: bool) -> dict:
    """A position's Monte-Carlo ground truth shaped for the UI: W/L/D as
    fractions, the score-delta histogram as sorted [delta, count] pairs, and
    its mean (the UI derives the std from the histogram)."""
    gt = _mc_ground_truth(face_up_leaves).get(name, {})
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


def _position_eval_legal(position: int) -> np.ndarray:
    return _position_eval_legal_for(_position_eval_gcg_key(position))


@lru_cache(maxsize=64)
def _position_eval_legal_for(gcg_key: tuple[str, int]) -> np.ndarray:
    """(4, 15, 15) bool: which cells each placement head can legally reach at
    the position. Depends only on the position, not the generation."""
    return legal_position_eval_placement(Path(gcg_key[0]).read_text())


def _placement_block(name: str, face_up_leaves: bool, pred, legal: np.ndarray) -> dict | None:
    """The Positions tab's residual heat map: per placement head, the rollouts'
    per-cell occupancy fraction, the model's prediction (`pred`, or None), and
    which cells the head can reach at all, so the UI can tell an unreachable
    cell from one that is merely unlikely.

    None when the ground-truth file has no placement planes; the UI then hides
    the overlay."""
    gt = _mc_ground_truth(face_up_leaves).get(name, {})
    planes = gt.get("placement")
    if planes is None:
        return None
    n = gt.get("n", 0)
    denom = n or 1
    heads = {}
    for i, head in enumerate(PLACEMENT_HEAD_NAMES):
        truth = (np.asarray(planes[head], dtype=np.float64) / denom).tolist()
        heads[head] = {
            "truth": truth,
            "pred": None if pred is None else pred[head].tolist(),
            "legal": legal[i].tolist(),
        }
    return {"n": n, "heads": heads}


def position_eval_position_payload(position: int, generation, tag, task, mount_root) -> dict:
    """The Positions tab's view of one position: board and racks, the
    Monte-Carlo truth for the tag's information condition, and the selected
    generation's prediction, computed on demand from its ONNX export (None
    without a usable export).

    The opponent's rack is their retained leave plus tiles drawn since. The
    leave's tiles are sent only under face-up leaves; under hidden leaves the
    client gets just its size, so it never receives tiles the condition says it
    cannot see."""
    face_up_leaves = _position_eval_face_up_leaves(task, tag)
    name, bundle = _position_eval_board(position)
    pred = _position_eval_prediction(tag, task, mount_root, generation, position)
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
        "face_up_leaves": face_up_leaves,
        "opp_leave": bundle["opp_leave"] if face_up_leaves else None,
        "opp_leave_size": len(bundle["opp_leave"]),
        "generation": generation,
        "has_prediction": pred is not None,
        "mc": _mc_payload(name, face_up_leaves),
        "model": pred["model"] if pred else None,
        "placement": _placement_block(
            name,
            face_up_leaves,
            pred["placement"] if pred else None,
            _position_eval_legal(position),
        ),
    }


def position_eval_generations(conn, tag, task, mount_root) -> list[dict]:
    """The Positions tab's generation slider: every generation with an ONNX
    export (predictions are computed from it), ascending. `positions` is the
    rows trained at that generation, None until its record has been ingested,
    which can trail the export by a tick."""
    clock = db.read_rows_clock(conn) if conn is not None else {}
    return [
        {"generation": g, "positions": clock.get(g)}
        for g in TagPaths(tag, task, mount_root).exported_generations()
    ]


def _resolve_position_eval_generation(tag, task, mount_root, arg: str):
    """A generation query arg as an index: '' or 'latest' is the newest export
    (None when there is none)."""
    if arg in ("", "latest"):
        exported = TagPaths(tag, task, mount_root).exported_generations()
        return exported[-1] if exported else None
    return int(arg)


# ONNX sessions by (path, mtime). The Positions tab runs exports on the CPU
# under fp32 onnxruntime, which reproduces the torch model they came from.
_ONNX_SESSIONS: dict = {}


def _position_eval_onnx_session(onnx_path: Path):
    key = (str(onnx_path), onnx_path.stat().st_mtime)
    sess = _ONNX_SESSIONS.get(key)
    if sess is None:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _ONNX_SESSIONS[key] = sess
    return sess


def _position_eval_model_arm(sess) -> InputArm:
    """The input arm an exported model consumes, read from its ONNX metadata
    (an absent flag means off) and its declared input widths. One dashboard
    serves models of every arm, so each position is encoded under the model's
    own. The engine rejects widths it cannot produce at the encode call."""
    meta = sess.get_modelmeta().custom_metadata_map
    model_inputs = {i.name: i.shape for i in sess.get_inputs()}
    return InputArm(
        opp_leave_input=meta.get("opp_leave_input") == "true",
        spatial_planes=int(model_inputs["input_spatial"][1]),
        scalar_size=int(model_inputs["input_scalar"][1]),
    )


def _position_eval_onnx_feed(arm: InputArm, flat_input: np.ndarray) -> dict:
    """The ``{input_spatial, input_scalar}`` feed from a row encoded under `arm`."""
    spatial, scalar = arm.split(flat_input)
    return {
        "input_spatial": spatial[None].astype(np.float32),
        "input_scalar": scalar[None].astype(np.float32),
    }


def _decode_value_outputs(wld: np.ndarray, sd: np.ndarray) -> dict:
    """One row's value outputs for the UI: W/L/D probabilities and the
    score-delta mean and std."""
    probs = np.exp(wld - wld.max())
    probs /= probs.sum()
    return {
        "wld": {"win": float(probs[0]), "draw": float(probs[1]), "loss": float(probs[2])},
        "sd_mean": float(sd[0]),
        "sd_std": float(sd[1]),
    }


def _collapse_placement_outputs(outs: list, gcg_text: str) -> dict:
    """The placement heads' raw footprint logits for one row, in
    PLACEMENT_HEAD_NAMES order, as (15, 15) per-cell occupancy marginals by
    head: the frame the Monte-Carlo placement planes use.

    The engine masks illegal footprints, softmaxes, and spreads each
    footprint's probability over the cells it covers. The analysis encoder
    never transposes the board, so the planes need no un-transpose."""
    raw = np.stack([out[0] for out in outs], axis=0)  # (4, FOOTPRINT_CLASSES)
    planes = collapse_position_eval_placement(gcg_text, raw)  # (4, 15, 15)
    return dict(zip(PLACEMENT_HEAD_NAMES, planes, strict=True))


def _run_position_eval_onnx(sess, arm: InputArm, flat_input: np.ndarray) -> dict:
    """Run an export's value heads on one encoded row (the alternate-leave
    what-if)."""
    wld, sd = sess.run(["wld", "score_diff"], _position_eval_onnx_feed(arm, flat_input))
    return _decode_value_outputs(wld[0], sd[0])


def _run_position_eval_prediction(
    sess, arm: InputArm, flat_input: np.ndarray, gcg_text: str
) -> dict:
    """Run every head of an export on one dataset position: {"model": value
    outputs, "placement": per-head planes}."""
    outs = sess.run(
        ["wld", "score_diff", *PLACEMENT_HEAD_NAMES], _position_eval_onnx_feed(arm, flat_input)
    )
    return {
        "model": _decode_value_outputs(outs[0][0], outs[1][0]),
        "placement": _collapse_placement_outputs(outs[2:], gcg_text),
    }


@lru_cache(maxsize=256)
def _position_eval_prediction_for(onnx_path_str: str, gcg_key: tuple[str, int]) -> dict | None:
    """One export's prediction on one dataset GCG, or None when the engine
    cannot encode the model's input widths.

    Memoizing is sound because an export is written atomically and never
    rewritten, and a GCG rewritten in place changes its key's mtime."""
    sess = _position_eval_onnx_session(Path(onnx_path_str))
    arm = _position_eval_model_arm(sess)
    gcg_text = Path(gcg_key[0]).read_text()
    try:
        flat = analyze_position_eval_gcg(gcg_text, arm)
    except ValueError:  # the model's widths are not today's layout
        return None
    return _run_position_eval_prediction(sess, arm, flat, gcg_text)


def _position_eval_prediction(tag, task, mount_root, generation, position) -> dict | None:
    """The generation's prediction on a dataset position, or None when it has
    no export or one the engine cannot encode. Existence is checked before the
    memo, so a missing export is never cached as a permanent None."""
    if generation is None:
        return None
    onnx_path = TagPaths(tag, task, mount_root).onnx_path(generation)
    if not onnx_path.exists():
        return None
    return _position_eval_prediction_for(str(onnx_path), _position_eval_gcg_key(position))


# --- Handlers ----------------------------------------------------------------


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
    values and the change events, LR schedule phases included; POST {name,
    value} sets one, which the trainer adopts at its next generation.

    Values persist in the tag's dashboard.db. Every set also rewrites the tag's
    controls file, which is what the trainer actually reads
    (generational/records.py)."""

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
        # Creates the DB if needed (unlike _open_conn), so a control can be set
        # before the first training run.
        paths = TagPaths(
            self.get_query_argument("tag"), self.get_query_argument("task"), self.mount_root
        )
        conn = db.connect(paths.dashboard_db)
        try:
            db.write_control(conn, name, float(value))
            controls = db.read_controls(conn)
        finally:
            conn.close()
        write_controls_file(paths, controls)
        self.write({"controls": controls})


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
            item, structure = build_figure_item(conn, name, self._params(), self.mount_root)
            self.write({"item": item, "structure": structure})
        finally:
            conn.close()


class FigureDeltaHandler(_Base):
    """Incremental update for an embedded figure (figure_delta.py)."""

    def post(self, name: str):
        if name not in FIGURES:
            self.set_status(404)
            self.write({"error": f"unknown figure {name!r}"})
            return
        try:
            client = json.loads(self.request.body)
        except ValueError:
            self.set_status(400)
            self.write({"error": "invalid JSON body"})
            return
        conn = self._open_conn()
        if conn is None:
            self.set_status(404)
            self.write({"error": "unknown tag"})
            return
        try:
            model = FIGURES[name](conn, self._params(), self.mount_root)
            self.write(figure_delta.delta_response(model, client))
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
    """One position's lane view. `generation` defaults to the newest recorded."""

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
    """The position_eval dataset's positions (the UI's position selector)."""

    def get(self):
        try:
            self.write({"positions": [gcg.stem for gcg in _position_eval_dataset_files()]})
        except OSError:
            self.write({"positions": []})


class PositionEvalGenerationsHandler(_Base):
    """The model generations a tag has exported (the slider)."""

    def get(self):
        conn = self._open_conn()
        try:
            self.write(
                {
                    "generations": position_eval_generations(
                        conn,
                        self.get_query_argument("tag"),
                        self.get_query_argument("task"),
                        self.mount_root,
                    )
                }
            )
        finally:
            if conn is not None:
                conn.close()


class PositionEvalPositionHandler(_Base):
    """One position's Positions-tab view. `generation` defaults to the newest
    export."""

    def get(self):
        files = _position_eval_dataset_files()
        position = int(self.get_query_argument("position", "0"))
        if not 0 <= position < len(files):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        tag, task = self.get_query_argument("tag"), self.get_query_argument("task")
        try:
            generation = _resolve_position_eval_generation(
                tag, task, self.mount_root, self.get_query_argument("generation", "latest")
            )
            self.write(
                position_eval_position_payload(position, generation, tag, task, self.mount_root)
            )
        except KeyError as e:  # a tag with no task.json: its condition is unknown
            self.set_status(404)
            self.write({"error": str(e)})
        except OSError:  # engine unavailable -> can't build the board
            self.set_status(503)
            self.write({"error": "engine unavailable; cannot build board"})


class PositionEvalAltLeaveHandler(_Base):
    """What-if: the selected generation's value outputs on a position with
    alternate leaves. Query: position, generation, leave, and optionally
    opp_leave (only for a model with an opponent-leave input). An invalid or
    unavailable leave is a 400 whose message says why."""

    def get(self):
        files = _position_eval_dataset_files()
        position = int(self.get_query_argument("position", "0"))
        if not 0 <= position < len(files):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        leave = self.get_query_argument("leave", "").strip()
        opp_leave = self.get_query_argument("opp_leave", None)
        tag, task = self.get_query_argument("tag"), self.get_query_argument("task")
        generation = _resolve_position_eval_generation(
            tag, task, self.mount_root, self.get_query_argument("generation", "latest")
        )
        if generation is None:
            self.set_status(404)
            self.write({"error": "no model generations recorded yet"})
            return
        onnx_path = TagPaths(tag, task, self.mount_root).onnx_path(generation)
        if not onnx_path.exists():
            self.set_status(404)
            self.write({"error": f"model for generation {generation} is not available"})
            return
        sess = _position_eval_onnx_session(onnx_path)
        arm = _position_eval_model_arm(sess)
        if opp_leave is not None and not arm.opp_leave_input:
            self.set_status(400)
            self.write({"error": "this model has no opponent-leave input"})
            return
        try:
            inp = analyze_position_eval_gcg_leaves(
                files[position].read_text(), leave, opp_leave, arm
            )
        except ValueError as e:  # bad size / unavailable tiles / stale model -> show the reason
            self.set_status(400)
            self.write({"error": str(e)})
            return
        except OSError:
            self.set_status(503)
            self.write({"error": "engine unavailable; cannot encode position"})
            return
        self.write(
            {
                "leave": leave,
                "opp_leave": opp_leave,
                "generation": generation,
                "model": _run_position_eval_onnx(sess, arm, inp),
            }
        )


def make_app(mount_root: str, worker_manager=None) -> tornado.web.Application:
    """The API app: the training data plane plus the master control plane,
    whose handlers need `worker_manager`."""
    return tornado.web.Application(
        [
            *master_api.MASTER_ROUTES,
            (r"/api/tags", TagsHandler),
            (r"/api/version", VersionHandler),
            (r"/api/meta", MetaHandler),
            (r"/api/controls", ControlsHandler),
            (r"/api/figure/([a-z_]+)", FigureHandler),
            (r"/api/figure_delta/([a-z_]+)", FigureDeltaHandler),
            (r"/api/lane/positions", LanePositionsHandler),
            (r"/api/lane/generations", LaneGenerationsHandler),
            (r"/api/lane/position", LanePositionHandler),
            (r"/api/position_eval/positions", PositionEvalPositionsHandler),
            (r"/api/position_eval/generations", PositionEvalGenerationsHandler),
            (r"/api/position_eval/position", PositionEvalPositionHandler),
            (r"/api/position_eval/alt_leave", PositionEvalAltLeaveHandler),
            *trajectories_api.ROUTES,
        ],
        mount_root=mount_root,
        worker_manager=worker_manager,
    )


# Held for the process lifetime so the flock is not dropped by garbage
# collection; the kernel releases it automatically when the process exits.
_CONTROL_LOCK = None


def _acquire_control_lock(mount_root: str):
    """Exit unless this is the only dashboard managing `mount_root`.

    Two dashboards on one mount would fight over the same slots,
    double-spawning local workers and enforcing conflicting pauses and gates.
    react_server's port reclaim does not prevent that (a second dashboard can
    use other ports), so an exclusive flock on <mount>/.dashboard.lock does.
    The kernel drops the lock when its holder dies, so a crashed dashboard
    never leaves a stale one."""
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
    """Serve the API on `port` until SIGTERM or interrupt, reconciling worker
    slots at boot and every RECONCILE_SECONDS.

    Binds to localhost only: the control plane holds cloud credentials and
    launches processes, so it must not be reachable off-machine. The browser
    reaches it through Vite's /api proxy.

    A pass that overruns delays the next one (PeriodicCallback awaits it). On
    shutdown, local workers get SIGTERM and flush; remote machines and their
    containers keep running.
    """
    _acquire_control_lock(mount_root)
    manager = WorkerManager()
    make_app(mount_root, manager).listen(port, address="127.0.0.1")
    loop = tornado.ioloop.IOLoop.current()

    async def reconcile():
        try:
            await manager.reconcile()
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

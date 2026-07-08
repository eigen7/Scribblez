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
import json
import signal
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tornado.ioloop
import tornado.web
from bokeh.embed import json_item
from bokeh.layouts import column

from scribblez import lane_analysis
from scribblez.dashboard import db, master_api, plots
from scribblez.dashboard.workers import WorkerManager
from scribblez.ffi import (
    analyze_gcg,
    analyze_position_eval_gcg_leave,
    get_input_shapes,
    position_eval_board_json,
)
from scribblez.paths import TagPaths
from scribblez.position_eval import analysis as position_eval_analysis

# The lane-union tile kinds in order: 26 letters then the collapsed blank.
_LANE_KINDS = [chr(ord("A") + k) for k in range(26)] + ["?"]

# The tables whose row counts form the per-tag change token the React shell polls
# (a change in any count means that tab's data advanced). Mirrors the Bokeh shell's
# per-tab ``watch()``.
# How often the WorkerManager closes desired-vs-actual worker-slot gaps.
RECONCILE_SECONDS = 30

VERSION_TABLES = (
    "metrics",
    "monotonicity",
    "calibration",
    "score_belief",
    "control_event",
)


def _loss(conn, params, image_dir, mount_root):
    """The Loss tab's top panel: the loss/accuracy curves from the per-epoch
    metrics table, with control-change markers. The value-quality curves are a
    separate figure (`eval_quality`), so the client can place its own controls
    between the two."""
    return plots.metrics_loss_grid(conn, normalized=_truthy(params.get("normalized")))


def _eval_quality(conn, params, image_dir, mount_root):
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


def _training_metrics(conn, params, image_dir, mount_root):
    return plots.series_grid(conn, plots.TRAINING) if _row_count(conn, "metrics") else None


def _gen_idx(params) -> int | None:
    """The requested generation index (`?gen_idx=`), or None for the newest."""
    v = params.get("gen_idx")
    return int(v) if v not in (None, "", "latest") else None


def _positions(conn, params, image_dir, mount_root):
    view = plots.probes_view(
        conn, image_dir, init_gen=_gen_idx(params), follow=False, external_gen=True
    )
    return column(view.layout, plots.series_grid(conn, plots.PROBE_CURVES)) if view else None


def _calibration(conn, params, image_dir, mount_root):
    view = plots.calibration_view(conn, init_gen=_gen_idx(params), follow=False, external_gen=True)
    return column(view.layout, plots.series_grid(conn, plots.CALIB_CURVES)) if view else None


# Figure name -> builder(conn, params, image_dir, mount_root) -> Bokeh model | None.
# Reuses the plots.py builders; the model is serialized with json_item for client-
# side embedding. `image_dir` is the tag's board-image dir (only the probes view
# uses it); `mount_root` lets a builder open a second tag's DB (eval_quality's
# secondary-tag overlay).
FIGURES = {
    "loss": _loss,
    "eval_quality": _eval_quality,
    "training_metrics": _training_metrics,
    "positions": _positions,
    "calibration": _calibration,
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


# Tables a per-generation tab (Positions, Calibration) scrubs with a GenerationSlider.
_GENERATION_TABLES = ("monotonicity", "calibration")


def _table_generations(conn: sqlite3.Connection, table: str) -> list:
    """The recorded epochs in `table`, oldest first (the slider's generation list)."""
    try:
        return [r[0] for r in conn.execute(f"SELECT epoch FROM {table} ORDER BY epoch")]
    except sqlite3.OperationalError:
        return []


def build_figure_item(
    conn: sqlite3.Connection, name: str, params: dict, image_dir, mount_root: str
):
    """The Bokeh ``json_item`` dict for figure `name`, or None when there's no data
    (or no such figure). The handler turns None into ``{"item": null}``."""
    builder = FIGURES.get(name)
    if builder is None:
        return None
    model = builder(conn, params, image_dir, mount_root)
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


def position_eval_position_payload(conn, position: int, generation) -> dict:
    """The full per-position view: board + leave for rendering, the Monte-Carlo ground
    truth, and the selected generation's model prediction (WLD + score-delta Gaussian)
    -- or None when no prediction exists for this generation/position."""
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


def _run_position_eval_onnx(onnx_path: Path, flat_input: np.ndarray) -> dict:
    """Run the exported post-move model on one flat input tensor and decode the value
    outputs: W/L/D probabilities and the score-delta mean/std.

    `flat_input` is the dashboard session's full row layout; the model declares
    its own arm in its ONNX metadata_props ("contingent_features", stamped at
    export). A baseline model consumes the base layout, whose spatial planes and
    scalars are prefixes of the full blocks, so its declared input widths select
    the right slices."""
    sess = _position_eval_onnx_session(onnx_path)
    meta = sess.get_modelmeta().custom_metadata_map
    contingent = meta["contingent_features"] == "true"
    model_inputs = {i.name: i.shape for i in sess.get_inputs()}
    planes = int(model_inputs["input_spatial"][1])
    scalars = int(model_inputs["input_scalar"][1])
    full_planes = {s.name: s.dims for s in get_input_shapes()}["input_spatial"][0]
    if contingent != (planes == full_planes):
        raise ValueError(f"{onnx_path}: metadata arm disagrees with the declared input widths")
    cells = position_eval_analysis.BOARD_SIZE**2
    spatial = flat_input[: planes * cells].reshape(1, planes, 15, 15).astype(np.float32)
    scalar = flat_input[full_planes * cells :][:scalars].reshape(1, -1).astype(np.float32)
    wld, sd = sess.run(["wld", "score_diff"], {"input_spatial": spatial, "input_scalar": scalar})
    probs = np.exp(wld[0] - wld[0].max())
    probs /= probs.sum()
    return {
        "wld": {"win": float(probs[0]), "draw": float(probs[1]), "loss": float(probs[2])},
        "sd_mean": float(sd[0, 0]),
        "sd_std": float(sd[0, 1]),
    }


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

    def _image_dir(self):
        """The tag's board-image dir (the probes figure renders boards from it)."""
        return TagPaths(
            self.get_query_argument("tag"), self.get_query_argument("task"), self.mount_root
        ).test_subset_dir


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
    """Live operator controls (e.g. base_lr). GET returns the current values and
    the rows-clock change events; POST {name, value} sets one, which the trainer
    adopts at its next epoch. Values persist in the tag's dashboard.db."""

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


class GenerationsHandler(_Base):
    """The generations (epochs) recorded in a table -- drives a tab's GenerationSlider."""

    def get(self):
        table = self.get_query_argument("table")
        if table not in _GENERATION_TABLES:
            self.set_status(404)
            self.write({"error": "unknown table"})
            return
        conn = self._open_conn()
        if conn is None:
            self.write({"generations": []})
            return
        try:
            self.write({"generations": _table_generations(conn, table)})
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
            item = build_figure_item(conn, name, self._params(), self._image_dir(), self.mount_root)
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
            self.write(position_eval_position_payload(conn, position, generation))
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
            (r"/api/generations", GenerationsHandler),
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


def run(port: int, mount_root: str):
    """Serve the API on `port` until SIGTERM/interrupt (used by the dashboard
    launcher). Binds to localhost only: the control plane holds cloud
    credentials and launches processes, so it must not be reachable
    off-machine (the browser reaches it through the Vite /api proxy).

    The WorkerManager reconciles worker slots at boot (relaunching local
    workers that should be running) and every RECONCILE_SECONDS thereafter
    (restarting interruptible pods Runpod reclaimed). On shutdown, owned local
    workers get SIGTERM (they flush and exit); cloud pods keep running.
    """
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

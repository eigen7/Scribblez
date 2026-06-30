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
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import tornado.ioloop
import tornado.web
from bokeh.embed import json_item
from bokeh.layouts import column

from scribblez import lane_analysis
from scribblez.dashboard import db, plots
from scribblez.ffi import analyze_gcg, post_move_board_json
from scribblez.paths import TagPaths
from scribblez.post_move_value import analysis as post_move_analysis

# The lane-union tile kinds in order: 26 letters then the collapsed blank.
_LANE_KINDS = [chr(ord("A") + k) for k in range(26)] + ["?"]

# The tables whose row counts form the per-tag change token the React shell polls
# (a change in any count means that tab's data advanced). Mirrors the Bokeh shell's
# per-tab ``watch()``.
VERSION_TABLES = (
    "train_step",
    "metrics",
    "throughput",
    "monotonicity",
    "calibration",
    "score_belief",
)


def _train_step(conn, params, image_dir):
    return plots.train_step_grid(conn, normalized=_truthy(params.get("normalized")))


def _throughput(conn, params, image_dir):
    return plots.throughput_grid(conn)


def _training_metrics(conn, params, image_dir):
    return plots.series_grid(conn, plots.TRAINING) if _row_count(conn, "metrics") else None


def _gen_idx(params) -> int | None:
    """The requested generation index (`?gen_idx=`), or None for the newest."""
    v = params.get("gen_idx")
    return int(v) if v not in (None, "", "latest") else None


def _positions(conn, params, image_dir):
    view = plots.probes_view(
        conn, image_dir, init_gen=_gen_idx(params), follow=False, external_gen=True
    )
    return column(view.layout, plots.series_grid(conn, plots.PROBE_CURVES)) if view else None


def _calibration(conn, params, image_dir):
    view = plots.calibration_view(conn, init_gen=_gen_idx(params), follow=False, external_gen=True)
    return column(view.layout, plots.series_grid(conn, plots.CALIB_CURVES)) if view else None


# Figure name -> builder(conn, params, image_dir) -> Bokeh model | None. Reuses the
# plots.py builders; the model is serialized with json_item for client-side
# embedding. `image_dir` is the tag's board-image dir (only the probes view uses it).
FIGURES = {
    "train_step": _train_step,
    "throughput": _throughput,
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


def build_figure_item(conn: sqlite3.Connection, name: str, params: dict, image_dir):
    """The Bokeh ``json_item`` dict for figure `name`, or None when there's no data
    (or no such figure). The handler turns None into ``{"item": null}``."""
    builder = FIGURES.get(name)
    if builder is None:
        return None
    model = builder(conn, params, image_dir)
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
def _post_move_dataset_files() -> tuple:
    return tuple(post_move_analysis.dataset_gcgs(post_move_analysis.DEFAULT_DATASET))


@lru_cache(maxsize=64)
def _post_move_board(position: int) -> tuple:
    """(name, board_bundle) for a post-move dataset position -- board / bonuses / leave
    rack for rendering. Cached: it is fixed for the dataset."""
    gcg = _post_move_dataset_files()[position]
    return gcg.stem, post_move_board_json(gcg.read_text())


@lru_cache(maxsize=1)
def _mc_ground_truth() -> dict:
    """The committed Monte-Carlo ground truth, keyed by position name (pos-1, ...)."""
    path = post_move_analysis.DEFAULT_DATASET / post_move_analysis.GROUND_TRUTH_FILENAME
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


def post_move_position_payload(conn, position: int, generation) -> dict:
    """The full per-position view: board + leave for rendering, the Monte-Carlo ground
    truth, and the selected generation's model prediction (WLD + score-delta Gaussian)
    -- or None when no prediction exists for this generation/position."""
    name, bundle = _post_move_board(position)
    pred = (
        db.read_post_move_pred(conn, generation, position)
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
        "board": bundle["board"],
        "bonuses": bundle["bonuses"],
        "rack": bundle["rack"],
        "tile_scores": bundle["tile_scores"],
        "scores": bundle["scores"],
        "generation": generation,
        "has_prediction": pred is not None,
        "mc": _mc_payload(name),
        "model": model,
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
            self.write({"item": build_figure_item(conn, name, self._params(), self._image_dir())})
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


class PostMovePositionsHandler(_Base):
    """The post-move-value dataset's positions (the UI's position selector)."""

    def get(self):
        try:
            self.write({"positions": [gcg.stem for gcg in _post_move_dataset_files()]})
        except OSError:
            self.write({"positions": []})


class PostMoveGenerationsHandler(_Base):
    """The model generations a tag has post-move-value predictions for (the slider)."""

    def get(self):
        conn = self._open_conn()
        if conn is None:
            self.write({"generations": []})
            return
        try:
            self.write({"generations": db.read_post_move_generations(conn)})
        finally:
            conn.close()


class PostMovePositionHandler(_Base):
    """Board + Monte-Carlo ground truth merged with one generation's prediction.
    `generation` may be omitted or 'latest' to use the newest recorded one."""

    def get(self):
        files = _post_move_dataset_files()
        position = int(self.get_query_argument("position", "0"))
        if not 0 <= position < len(files):
            self.set_status(404)
            self.write({"error": "position out of range"})
            return
        conn = self._open_conn()
        try:
            generation = self._resolve_generation(conn)
            self.write(post_move_position_payload(conn, position, generation))
        except OSError:  # engine unavailable -> can't build the board
            self.set_status(503)
            self.write({"error": "engine unavailable; cannot build board"})
        finally:
            if conn is not None:
                conn.close()

    def _resolve_generation(self, conn):
        arg = self.get_query_argument("generation", "latest")
        if conn is not None and arg in ("", "latest"):
            gens = db.read_post_move_generations(conn)
            return gens[-1]["generation"] if gens else None
        return int(arg) if arg not in ("", "latest") else None


def make_app(mount_root: str) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/api/tags", TagsHandler),
            (r"/api/version", VersionHandler),
            (r"/api/meta", MetaHandler),
            (r"/api/generations", GenerationsHandler),
            (r"/api/figure/([a-z_]+)", FigureHandler),
            (r"/api/lane/positions", LanePositionsHandler),
            (r"/api/lane/generations", LaneGenerationsHandler),
            (r"/api/lane/position", LanePositionHandler),
            (r"/api/post_move/positions", PostMovePositionsHandler),
            (r"/api/post_move/generations", PostMoveGenerationsHandler),
            (r"/api/post_move/position", PostMovePositionHandler),
        ],
        mount_root=mount_root,
    )


def run(port: int, mount_root: str):
    """Serve the API on `port` until interrupted (used by the dashboard launcher)."""
    make_app(mount_root).listen(port, address="0.0.0.0")
    tornado.ioloop.IOLoop.current().start()


def main():
    p = argparse.ArgumentParser(description="Serve the React dashboard's data API.")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--mount-root", default="/workspace/mount")
    args = p.parse_args()
    run(args.port, args.mount_root)


if __name__ == "__main__":
    main()

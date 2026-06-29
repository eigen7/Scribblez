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

import sqlite3
from pathlib import Path

import tornado.web
from bokeh.embed import json_item

from scribblez.dashboard import db, plots
from scribblez.paths import TagPaths

# The tables whose row counts form the per-tag change token the React shell polls
# (a change in any count means that tab's data advanced). Mirrors the Bokeh shell's
# per-tab ``watch()``.
VERSION_TABLES = ("train_step", "metrics", "throughput", "monotonicity", "calibration", "score_belief")


def _train_step(conn: sqlite3.Connection, params: dict):
    return plots.train_step_grid(conn, normalized=_truthy(params.get("normalized")))


# Figure name -> builder(conn, params) -> Bokeh model | None. Reuses plots.py
# unchanged; the model is serialized with json_item for client-side embedding.
FIGURES = {
    "train_step": _train_step,
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


def build_figure_item(conn: sqlite3.Connection, name: str, params: dict):
    """The Bokeh ``json_item`` dict for figure `name`, or None when there's no data
    (or no such figure). The handler turns None into ``{"item": null}``."""
    builder = FIGURES.get(name)
    if builder is None:
        return None
    model = builder(conn, params)
    return json_item(model) if model is not None else None


def _open(mount_root: str, task: str, tag: str) -> sqlite3.Connection | None:
    """Open a tag's dashboard DB, or None if it doesn't exist yet (no spurious
    empty DB is created for an unknown tag)."""
    path = Path(TagPaths(tag, task, mount_root).dashboard_db)
    return db.connect(path) if path.exists() else None


class _Base(tornado.web.RequestHandler):
    @property
    def mount_root(self) -> str:
        return self.settings["mount_root"]

    def _params(self) -> dict:
        return {k: self.get_query_argument(k) for k in self.request.query_arguments}


class TagsHandler(_Base):
    def get(self):
        task = self.get_query_argument("task")
        self.write({"tags": db.list_tags(self.mount_root, task)})


class VersionHandler(_Base):
    def get(self):
        conn = _open(self.mount_root, self.get_query_argument("task"), self.get_query_argument("tag"))
        if conn is None:
            self.set_status(404)
            self.write({"error": "unknown tag"})
            return
        try:
            self.write(version_token(conn))
        finally:
            conn.close()


class FigureHandler(_Base):
    def get(self, name: str):
        if name not in FIGURES:
            self.set_status(404)
            self.write({"error": f"unknown figure {name!r}"})
            return
        conn = _open(self.mount_root, self.get_query_argument("task"), self.get_query_argument("tag"))
        if conn is None:
            self.set_status(404)
            self.write({"error": "unknown tag"})
            return
        try:
            self.write({"item": build_figure_item(conn, name, self._params())})
        finally:
            conn.close()


def make_app(mount_root: str) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/api/tags", TagsHandler),
            (r"/api/version", VersionHandler),
            (r"/api/figure/([a-z_]+)", FigureHandler),
        ],
        mount_root=mount_root,
    )


def run(port: int, mount_root: str):
    """Serve the API on `port` until interrupted (used by the dashboard launcher)."""
    import tornado.ioloop

    make_app(mount_root).listen(port, address="0.0.0.0")
    tornado.ioloop.IOLoop.current().start()


def main():
    import argparse

    p = argparse.ArgumentParser(description="Serve the React dashboard's data API.")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--mount-root", default="/workspace/mount")
    args = p.parse_args()
    run(args.port, args.mount_root)


if __name__ == "__main__":
    main()

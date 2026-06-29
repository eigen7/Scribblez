"""Shared shell for the per-task training dashboards.

Each model has its own Bokeh app (``app_<task>.py``) that picks the tabs relevant
to it; everything those apps share -- the sidebar/tag-selector chrome, the tab
machinery, the always-present Loss and Streaming tabs, and the ``run`` entry
point -- lives here. A `Tab` owns its content, knows how to (re)build it from the
DB, and declares what to watch so the periodic poll refreshes it only when its
data grows. Task-specific tabs live in sibling modules (e.g. post_move_tabs.py).
"""

from __future__ import annotations

import argparse
import sys
from functools import partial

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, CustomJS, Div, InlineStyleSheet, RadioButtonGroup, Select

from scribblez.dashboard import db, plots
from scribblez.paths import TagPaths

POLL_MS = 3000

BODY_STYLE = (
    "<style>html,body{background:#eef2f7;margin:0;"
    "font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;}</style>"
)
SIDEBAR_CSS = InlineStyleSheet(css="""
:host { background:#1f2733; padding:18px 12px; min-height:100vh;
        box-shadow:2px 0 12px rgba(15,22,33,.18); }
""")
CARD_CSS = InlineStyleSheet(css="""
:host { background:#ffffff; border-radius:10px; padding:14px; margin:0 0 16px 0;
        box-shadow:0 1px 5px rgba(20,30,50,.08); }
""")


def card():
    """A stretch-width white content card (children are replaced on refresh)."""
    return column(sizing_mode="stretch_width", stylesheets=[CARD_CSS])


# ---------------------------------------------------------------------------
# Tab protocol
# ---------------------------------------------------------------------------


class Tab:
    """One dashboard tab. `node` is the layout shown when the tab is active.

    refresh(ctx) (re)builds the content from ctx.conn -- called on tag switch.
    watch(ctx) returns a value the poll compares; when it changes, poll(ctx) runs
    (by default a full refresh) so the tab updates as its data grows.
    """

    label = "Tab"

    def __init__(self):
        self.node = card()

    def refresh(self, ctx):
        raise NotImplementedError

    def watch(self, ctx):
        return None

    def poll(self, ctx):
        self.refresh(ctx)


class LossTab(Tab):
    """Per-minibatch streaming loss/accuracy (the Loss tab). Shared by every task;
    the trainer's recorded loss-component weights drive the stacked view, and the
    Absolute/% toggle is shown only when those weights exist. `fallback_groups`,
    if given, is the per-epoch loss view used when no streaming data exists (the
    disk pipeline)."""

    label = "Loss"

    def __init__(self, fallback_groups=None):
        super().__init__()
        self._fallback = fallback_groups
        self._conn = None
        self._normalized = False
        self._mode = RadioButtonGroup(labels=["Absolute", "% of total"], active=0, width=180)
        self._mode.on_change("active", lambda a, o, n: self._set_mode(n))

    def _set_mode(self, active):
        self._normalized = active == 1
        if self._conn is not None:
            self._render()

    def refresh(self, ctx):
        self._conn = ctx.conn
        self._render()

    def _render(self):
        grid = plots.train_step_grid(self._conn, normalized=self._normalized)
        if grid is None:
            self.node.children = (
                [plots.series_grid(self._conn, self._fallback, ncols=1)]
                if self._fallback
                else [Div(text="<i>No loss data yet.</i>")]
            )
            return
        # The Absolute/% toggle only applies to the weighted stack (needs weights).
        head = [self._mode] if db.read_loss_weights(self._conn) else []
        self.node.children = head + [grid]

    def watch(self, ctx):
        return ctx.row_count("train_step")


class StreamingTab(Tab):
    """Live throughput / backpressure for the streaming pipeline. Shared by every
    task (both trainers stream self-play through the ring buffer)."""

    label = "Streaming"

    def refresh(self, ctx):
        self.node.children = [plots.throughput_grid(ctx.conn)]

    def watch(self, ctx):
        return ctx.row_count("throughput")


# ---------------------------------------------------------------------------
# Dashboard shell
# ---------------------------------------------------------------------------


class Dashboard:
    """Sidebar-navigated dashboard over a configurable list of `Tab`s."""

    def __init__(self, mount_root: str, tabs: list[Tab], initial_tag: str | None = None):
        self.mount_root = mount_root
        self.tabs = tabs
        self.conn = None
        self.tag = None
        self.active = 0
        self._watch = [None] * len(tabs)

        tags = db.list_tags(mount_root)
        default = initial_tag if initial_tag in tags else (tags[0] if tags else "(none)")
        self.tag_select = Select(options=tags or ["(none)"], value=default, width=120)
        self.tag_select.on_change("value", lambda a, o, n: self.on_tag())
        # Reflect the selected tag in the URL (?tag=<tag>) so it stays shareable.
        self.tag_select.js_on_change("value", CustomJS(code="""
            const url = new URL(window.location.href);
            url.searchParams.set('tag', cb_obj.value);
            window.history.replaceState({}, '', url);
        """))
        self.nav = [Button(label=t.label, button_type="light", sizing_mode="stretch_width")
                    for t in tabs]
        for i, b in enumerate(self.nav):
            b.on_click(partial(self.select_tab, i))
        self.on_tag()

    # --- context the tabs read -------------------------------------------

    def image_dir(self):
        return TagPaths(self.tag, self.mount_root).test_subset_dir

    def row_count(self, table: str) -> int:
        if not self.conn:
            return 0
        try:
            return int(self.conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])
        except Exception:  # noqa: BLE001 -- table may not exist on an old DB
            return 0

    # --- layout / navigation ---------------------------------------------

    def _sidebar(self):
        brand = Div(text="<div style='color:#eef2f7;font-size:19px;font-weight:700'>Scribblez</div>"
                         "<div style='color:#7d8aa0;font-size:11px'>training dashboard</div>",
                    styles={"margin-bottom": "16px"})
        tag_lbl = Div(text="<span style='color:#aab4c5;font-size:13px'>Tag</span>",
                      styles={"align-self": "center"})
        spacer = Div(text="", styles={"height": "14px"})
        return column(brand, row(tag_lbl, self.tag_select), spacer, *self.nav,
                      width=210, stylesheets=[SIDEBAR_CSS])

    def layout(self):
        main = column(*[t.node for t in self.tabs], sizing_mode="stretch_width",
                      styles={"padding": "18px"})
        return column(Div(text=BODY_STYLE),
                      row(self._sidebar(), main, sizing_mode="stretch_width"),
                      sizing_mode="stretch_width")

    def select_tab(self, i: int):
        self.active = i
        for idx, t in enumerate(self.tabs):
            t.node.visible = idx == i
        for idx, b in enumerate(self.nav):
            b.button_type = "primary" if idx == i else "light"

    # --- data refresh -----------------------------------------------------

    def on_tag(self):
        tag = self.tag_select.value
        if tag == "(none)":
            return
        self.tag = tag
        self.conn = db.connect(TagPaths(tag, self.mount_root).dashboard_db)
        for i, t in enumerate(self.tabs):
            t.refresh(self)
            self._watch[i] = t.watch(self)
        self.select_tab(self.active)

    def poll(self):
        new_tags = db.list_tags(self.mount_root)
        if new_tags and new_tags != self.tag_select.options:
            self.tag_select.options = new_tags
        if not self.conn:
            return
        for i, t in enumerate(self.tabs):
            w = t.watch(self)
            if w != self._watch[i]:
                self._watch[i] = w
                t.poll(self)


# ---------------------------------------------------------------------------
# Entry point (each app_<task>.py calls run(...) with its tab list)
# ---------------------------------------------------------------------------


def _parse_mount_root() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount-root", default="/workspace/mount")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.mount_root


def _requested_tag() -> str | None:
    """The `?tag=<tag>` URL query argument for this session, if any."""
    try:
        arguments = curdoc().session_context.request.arguments
    except AttributeError:  # no session/request (e.g. standalone, tests)
        return None
    values = (arguments or {}).get("tag")
    if not values:
        return None
    value = values[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def run(tabs: list[Tab], title: str = "Scribblez training dashboard"):
    """Wire a Dashboard with `tabs` into the Bokeh document. Called at import time
    by each app_<task>.py (which `bokeh serve` runs)."""
    dash = Dashboard(_parse_mount_root(), tabs, _requested_tag())
    doc = curdoc()
    doc.title = title
    doc.add_root(dash.layout())
    doc.add_periodic_callback(dash.poll, POLL_MS)
    return dash

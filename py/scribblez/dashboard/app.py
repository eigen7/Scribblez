"""Bokeh server application for the training dashboard.

Served via ``bokeh serve app.py --args --mount-root <dir>``. A left sidebar holds
the tag selector and the tab navigation; the main area shows one tab at a time:

  * Loss          -- loss learning curves
  * Positions     -- board + monotonicity + score-belief for one position/gen,
                     plus the structural-probe learning curves
  * Calibration   -- reliability diagrams for one gen, plus the WLD/calibration
                     learning curves
  * Training      -- learning-rate and epoch-time curves
  * Streaming     -- live throughput + backpressure for the streaming pipeline

Per-generation views preload every generation and scrub client-side. A "latest"
checkbox on each locks to (and follows) the newest generation; a periodic callback
refreshes the curves every epoch and advances the followed views.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, Div, InlineStyleSheet, Select

from scribblez.dashboard import db, plots
from scribblez.paths import TagPaths

POLL_MS = 3000

TABS = ["Loss", "Positions", "Calibration", "Training", "Streaming"]
LOSS = [("Loss", ["loss", "loss_wld", "loss_score_diff", "loss_opp_next_placement"])]
TRAINING = [("Learning rate", ["lr"]), ("Epoch time (s)", ["elapsed_s"])]
PROBE_CURVES = [
    ("Structural probe", ["probe_mean_structural_score", "probe_mean_sigmoid_r2"]),
    ("Monotonicity violations", ["probe_total_violations"]),
]
CALIB_CURVES = [
    ("WLD accuracy & Brier / ECE", ["wld_acc", "calib_brier", "calib_ece"]),
    ("Calibration log-loss", ["calib_log_loss"]),
    ("Score-diff calibration", ["calib_scorediff_mae", "calib_scorediff_bias", "calib_scorediff_sharpness"]),
]

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


def _parse_mount_root() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount-root", default="/workspace/mount")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.mount_root


class Dashboard:
    """Sidebar-navigated dashboard with live-following per-generation views."""

    def __init__(self, mount_root: str):
        self.mount_root = mount_root
        self.conn = None
        self.tag = None
        self.last_max_epoch = -1
        self.last_throughput_n = -1
        self.active = 0
        self.probes_view = None
        self.calib_view = None

        tags = db.list_tags(mount_root)
        self.tag_select = Select(options=tags or ["(none)"], value=tags[0] if tags else "(none)",
                                 width=120)
        self.tag_select.on_change("value", lambda a, o, n: self.on_tag())
        self.nav = [Button(label=t, button_type="light", sizing_mode="stretch_width") for t in TABS]
        for i, b in enumerate(self.nav):
            b.on_click(partial(self.select_tab, i))

        # One card container per content block (replaced piecewise on refresh).
        self.loss_card = self._card()
        self.training_card = self._card()
        self.probes_card = self._card()
        self.probes_curves = self._card()
        self.calib_card = self._card()
        self.calib_curves = self._card()
        self.streaming_card = self._card()
        self.contents = [
            column(self.loss_card, sizing_mode="stretch_width"),
            column(self.probes_card, self.probes_curves, sizing_mode="stretch_width"),
            column(self.calib_card, self.calib_curves, sizing_mode="stretch_width"),
            column(self.training_card, sizing_mode="stretch_width"),
            column(self.streaming_card, sizing_mode="stretch_width"),
        ]
        self.on_tag()

    def _card(self):
        return column(sizing_mode="stretch_width", stylesheets=[CARD_CSS])

    # --- layout -----------------------------------------------------------

    def _sidebar(self):
        brand = Div(text="<div style='color:#eef2f7;font-size:19px;font-weight:700'>Scribblez</div>"
                         "<div style='color:#7d8aa0;font-size:11px'>training dashboard</div>",
                    styles={"margin-bottom": "16px"})
        tag_lbl = Div(text="<span style='color:#aab4c5;font-size:13px'>Tag</span>",
                      styles={"align-self": "center"})
        tag_row = row(tag_lbl, self.tag_select)
        spacer = Div(text="", styles={"height": "14px"})
        return column(brand, tag_row, spacer, *self.nav, width=210, stylesheets=[SIDEBAR_CSS])

    def layout(self):
        main = column(*self.contents, sizing_mode="stretch_width", styles={"padding": "18px"})
        return column(Div(text=BODY_STYLE),
                      row(self._sidebar(), main, sizing_mode="stretch_width"),
                      sizing_mode="stretch_width")

    def select_tab(self, i: int):
        self.active = i
        for idx, content in enumerate(self.contents):
            content.visible = idx == i
        for idx, b in enumerate(self.nav):
            b.button_type = "primary" if idx == i else "light"

    # --- data -------------------------------------------------------------

    def _image_dir(self):
        return TagPaths(self.tag, self.mount_root).test_subset_dir

    def _epochs(self):
        return db.read_monotonicity_epochs(self.conn) if self.conn else []

    def _throughput_count(self) -> int:
        if not self.conn:
            return 0
        try:
            return int(self.conn.execute("SELECT COUNT(*) AS c FROM throughput").fetchone()["c"])
        except Exception:  # noqa: BLE001 -- table may not exist on an old DB
            return 0

    def on_tag(self):
        tag = self.tag_select.value
        if tag == "(none)":
            return
        self.tag = tag
        self.conn = db.connect(TagPaths(tag, self.mount_root).dashboard_db)
        self.refresh_curves()
        self.refresh_streaming()
        self.rebuild_probes()
        self.rebuild_calibration()
        epochs = self._epochs()
        self.last_max_epoch = max(epochs) if epochs else -1
        self.last_throughput_n = self._throughput_count()
        self.select_tab(self.active)

    def refresh_curves(self):
        """Rebuild the scalar learning-curve cards (cheap; grows each epoch)."""
        self.loss_card.children = [plots.series_grid(self.conn, LOSS, ncols=1)]
        self.training_card.children = [plots.series_grid(self.conn, TRAINING)]
        self.probes_curves.children = [plots.series_grid(self.conn, PROBE_CURVES)]
        self.calib_curves.children = [plots.series_grid(self.conn, CALIB_CURVES)]

    def refresh_streaming(self):
        """Rebuild the streaming throughput/backpressure card."""
        self.streaming_card.children = [plots.throughput_grid(self.conn)]

    def rebuild_probes(self):
        """(Re)build the interactive probes view, following the latest generation."""
        init_pos = int(self.probes_view.pos.value) if (self.probes_view and self.probes_view.pos) else 0
        v = plots.probes_view(self.conn, self._image_dir(), init_pos=init_pos, follow=True)
        self.probes_view = v
        self.probes_card.children = [v.layout if v else Div(text="<i>No probe data yet.</i>")]
        if v:
            v.latest.on_change("active", lambda a, o, n: n and self.rebuild_probes())

    def rebuild_calibration(self):
        v = plots.calibration_view(self.conn, follow=True)
        self.calib_view = v
        self.calib_card.children = [v.layout if v else Div(text="<i>No calibration data yet.</i>")]
        if v:
            v.latest.on_change("active", lambda a, o, n: n and self.rebuild_calibration())

    def poll(self):
        new_tags = db.list_tags(self.mount_root)
        if new_tags and new_tags != self.tag_select.options:
            self.tag_select.options = new_tags
        if not self.conn:
            return
        # Throughput grows independently of (and faster than) checkpoints.
        tn = self._throughput_count()
        if tn != self.last_throughput_n:
            self.last_throughput_n = tn
            self.refresh_streaming()
        epochs = self._epochs()
        mx = max(epochs) if epochs else -1
        if mx == self.last_max_epoch:
            return
        self.last_max_epoch = mx
        self.refresh_curves()  # curves always grow
        if self.probes_view and self.probes_view.latest.active:
            self.rebuild_probes()
        if self.calib_view and self.calib_view.latest.active:
            self.rebuild_calibration()


_dash = Dashboard(_parse_mount_root())
doc = curdoc()
doc.title = "Scribblez training dashboard"
doc.add_root(_dash.layout())
doc.add_periodic_callback(_dash.poll, POLL_MS)

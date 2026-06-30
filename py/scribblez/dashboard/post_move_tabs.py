"""Tabs specific to the post-move value model.

These read the post-move eval artifacts (structural probes / monotonicity,
score-belief bands, WLD & score-diff calibration, and the lr / epoch-time
curves) which the max-move-per-lane task does not produce. Each watches its own
table so the poll refreshes it only when that data grows.
"""

from bokeh.layouts import column
from bokeh.models import Div

from scribblez.dashboard import plots
from scribblez.dashboard.shell import Tab, card

# Per-epoch scalar curves, by tab.
LOSS = [("Loss", ["loss", "loss_wld", "loss_score_diff", "loss_opp_next_placement"])]
TRAINING = [("Learning rate", ["lr"]), ("Epoch time (s)", ["elapsed_s"])]
PROBE_CURVES = [
    ("Structural probe", ["probe_mean_structural_score", "probe_mean_sigmoid_r2"]),
    ("Monotonicity violations", ["probe_total_violations"]),
]
CALIB_CURVES = [
    ("WLD accuracy & Brier / ECE", ["wld_acc", "calib_brier", "calib_ece"]),
    ("Calibration log-loss", ["calib_log_loss"]),
    (
        "Score-diff calibration",
        ["calib_scorediff_mae", "calib_scorediff_bias", "calib_scorediff_sharpness"],
    ),
]


class TrainingTab(Tab):
    """Learning-rate and epoch-time learning curves."""

    label = "Training"

    def refresh(self, ctx):
        self.node.children = [plots.series_grid(ctx.conn, TRAINING)]

    def watch(self, ctx):
        return ctx.row_count("metrics")


class PositionsTab(Tab):
    """A frozen position's board + monotonicity curve + score-belief bands
    (scrubbable across checkpoints), plus the structural-probe learning curves.
    Follows the latest checkpoint while its `latest` box is ticked."""

    label = "Positions"

    def __init__(self):
        super().__init__()
        self._board = card()
        self._curves = card()
        self.node = column(self._board, self._curves, sizing_mode="stretch_width")
        self._view = None

    def _rebuild_board(self, ctx):
        init_pos = int(self._view.pos.value) if (self._view and self._view.pos) else 0
        v = plots.probes_view(ctx.conn, ctx.image_dir(), init_pos=init_pos, follow=True)
        self._view = v
        self._board.children = [v.layout if v else Div(text="<i>No probe data yet.</i>")]
        if v:
            v.latest.on_change("active", lambda a, o, n: n and self._rebuild_board(ctx))

    def refresh(self, ctx):
        self._rebuild_board(ctx)
        self._curves.children = [plots.series_grid(ctx.conn, PROBE_CURVES)]

    def poll(self, ctx):
        # Advance the followed board to the new checkpoint, but don't yank a user
        # who has scrubbed to a fixed one (latest unticked).
        if self._view and self._view.latest.active:
            self._rebuild_board(ctx)
        self._curves.children = [plots.series_grid(ctx.conn, PROBE_CURVES)]

    def watch(self, ctx):
        return ctx.row_count("monotonicity")


class CalibrationTab(Tab):
    """Reliability diagrams for one checkpoint (scrubbable, latest-following) plus
    the WLD / calibration learning curves."""

    label = "Calibration"

    def __init__(self):
        super().__init__()
        self._diagram = card()
        self._curves = card()
        self.node = column(self._diagram, self._curves, sizing_mode="stretch_width")
        self._view = None

    def _rebuild_diagram(self, ctx):
        v = plots.calibration_view(ctx.conn, follow=True)
        self._view = v
        self._diagram.children = [v.layout if v else Div(text="<i>No calibration data yet.</i>")]
        if v:
            v.latest.on_change("active", lambda a, o, n: n and self._rebuild_diagram(ctx))

    def refresh(self, ctx):
        self._rebuild_diagram(ctx)
        self._curves.children = [plots.series_grid(ctx.conn, CALIB_CURVES)]

    def poll(self, ctx):
        if self._view and self._view.latest.active:
            self._rebuild_diagram(ctx)
        self._curves.children = [plots.series_grid(ctx.conn, CALIB_CURVES)]

    def watch(self, ctx):
        return ctx.row_count("calibration")

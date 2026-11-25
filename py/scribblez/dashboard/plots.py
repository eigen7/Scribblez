"""Bokeh figure builders for the training dashboard's metric tabs: learning
curves, the Loss tab's stacked losses, and match results.

Each builder reads a tag's dashboard.db and returns a Bokeh model (or None when
there is nothing to plot yet); api.py serializes it as a json_item that the
React app embeds with BokehJS. Figures that stream incremental updates name
their data sources (`_source_name`) for figure_delta.py.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from bokeh.layouts import column, gridplot
from bokeh.models import (
    ColumnDataSource,
    Div,
    HoverTool,
    Label,
    Range1d,
    Span,
    Whisker,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure

from . import db

# Figures are sized by the page, not in pixels: each scales to the width its
# grid cell gets and keeps this width:height ratio, so a grid always fits the
# dashboard's capped column.
SQUARE = 1
WIDE = 2


# ---------------------------------------------------------------------------
# Scalar learning curves
# ---------------------------------------------------------------------------


# Below this many points an EMA's debiased head just traces the raw points, so
# smoothing is skipped and the raw series drawn instead.
_SMOOTH_MIN_POINTS = 10


def _ema(values, weight: float = 0.85):
    """TensorBoard-style debiased exponential moving average. `weight` in [0, 1)
    sets the smoothing (higher is smoother); the debias term keeps the early
    points from being dragged toward the zero initialization."""
    out = np.empty(len(values), dtype=np.float64)
    smoothed = 0.0
    debias = 0.0
    for i, v in enumerate(values):
        smoothed = smoothed * weight + (1.0 - weight) * float(v)
        debias = debias * weight + (1.0 - weight)
        out[i] = smoothed / debias if debias else float(v)
    return out


def _plot_series(fig, x, y, color, label, smooth, name=None):
    """Draw one metric series from a data source named `name`. A smoothed series
    is drawn as its EMA alone, with no raw underlay, so it reads as unambiguously
    smoothed; an unsmoothed one as a line plus markers."""
    if smooth and len(y) >= _SMOOTH_MIN_POINTS:
        src = ColumnDataSource(dict(x=x, y=_ema(np.asarray(y, dtype=np.float64))), name=name)
        fig.line("x", "y", source=src, color=color, line_width=2, legend_label=label)
    else:
        src = ColumnDataSource(dict(x=x, y=y), name=name)
        fig.line("x", "y", source=src, color=color, line_width=2, legend_label=label)
        fig.scatter("x", "y", source=src, color=color, size=4)


def _source_name(title: str, label: str) -> str:
    """The stable name of one series' data source, unique within a figure row.
    The linear and log x-axis rows share names on purpose: their sources hold
    identical data, so one incremental update (figure_delta.py) feeds both."""
    return f"{title}|{label}"


def _padded_range(values, log):
    """An explicit padded Range1d over `values`, or None when there is no finite
    (for a log axis, positive) value. Needed because Bokeh's default range for a
    near-constant series spans roughly value +/- 1, which buries e.g. a flat 1e-3
    learning rate in a [-1, 1] band."""
    finite = values[np.isfinite(values)]
    if log:
        finite = finite[finite > 0.0]
    if len(finite) == 0:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    if log:
        lo, hi = (
            (lo / 3.0, hi * 3.0) if lo == hi else (lo / (hi / lo) ** 0.1, hi * (hi / lo) ** 0.1)
        )
    elif lo == hi:
        span = abs(lo) or 1.0
        lo, hi = lo - 0.5 * span, hi + 0.5 * span
    else:
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    return Range1d(lo, hi)


def _set_padded_range(fig, axis: str, values, log):
    """Give `fig`'s `axis` ('x' | 'y') the padded range of `values`, keeping
    Bokeh's auto range when there is nothing to fit."""
    rng = _padded_range(np.asarray(values, dtype=np.float64), log)
    if rng is not None:
        setattr(fig, f"{axis}_range", rng)


def _series_figure(
    sources,
    title: str,
    names: list[str],
    *,
    log: bool = False,
    smooth: bool = False,
    log_x: bool = False,
):
    """A square learning-curve figure of the metrics `names`, or None when no
    source has any of them.

    `sources` is a list of (conn, legend suffix): each (source, metric) pair gets
    its own color, and the suffix (e.g. ' [tagB]') names the tag, so a second
    tag's curves overlay the first for comparison. `log` / `log_x` make the y /
    epoch axis logarithmic; a log epoch axis gets an explicit positive range so
    an epoch-0 point does not break it."""
    fig = figure(
        sizing_mode="scale_width",
        aspect_ratio=SQUARE,
        title=title,
        x_axis_label="epoch",
        x_axis_type="log" if log_x else "linear",
        y_axis_type="log" if log else "linear",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    fig.add_tools(HoverTool(tooltips=[("epoch", "@x"), ("value", "@y{0.0000}")], mode="vline"))
    palette = Category10[10]
    all_epochs, all_values = [], []
    for s, (conn, suffix) in enumerate(sources):
        for i, name in enumerate(names):
            epochs, values = db.read_metric_series(conn, name)
            if len(epochs) == 0:
                continue
            color = palette[(s * len(names) + i) % len(palette)]
            _plot_series(
                fig,
                epochs,
                values,
                color,
                name + suffix,
                smooth,
                name=_source_name(title, name + suffix),
            )
            all_epochs.append(epochs)
            all_values.append(values)
    if not all_values:
        return None
    _set_padded_range(fig, "y", np.concatenate(all_values), log)
    if log_x:
        _set_padded_range(fig, "x", np.concatenate(all_epochs), log=True)
    fig.legend.label_text_font_size = "8pt"
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    return fig


# Metric groups for series_grid(): each entry is (figure title, metric names)
# or (title, names, {"log": True}) for a log y-axis.
# Only schedule-free runs record the averaging weight (a WSD run's panel is
# absent): they hold the rate constant and anneal by giving each new iterate a
# smaller share of the deployed average, so the weight is the curve that shows
# the anneal. The gradient-norm panels come from runs that record them (the
# position-evaluation trainer): the norm is measured before clipping, and the
# clipped fraction says whether the clip is a spike guard or a constant rescale.
TRAINING = [
    ("Iterate averaging weight", ["averaging_weight"], {"log": True}),
    ("Gradient norm before clipping", ["grad_norm_mean", "grad_norm_max"], {"log": True}),
    ("Fraction of steps clipped", ["clip_frac"]),
    ("Epoch time (s)", ["elapsed_s"]),
]
# position_eval's model-vs-Monte-Carlo quality curves on its held-out position
# set, shown on the Loss tab beneath the training curves.
POST_MOVE_QUALITY = [
    ("Value quality vs Monte-Carlo — WLD", ["eval_win_mae", "eval_wld_brier"]),
    (
        "Value quality vs Monte-Carlo — score diff (points)",
        ["eval_sd_mean_mae", "eval_sd_std_mae"],
    ),
    # Each placement head's per-cell occupancy against the rollouts' (the
    # Positions tab's residual heat map, aggregated by
    # position_eval/analysis.placement_metrics): misplaced coverage in tiles,
    # and how often the model's most covered cell is the rollouts'.
    (
        "Placement vs Monte-Carlo — misplaced coverage (tiles)",
        [
            "eval_place_l1_opp_next",
            "eval_place_l1_self_next",
            "eval_place_l1_opp_win",
            "eval_place_l1_self_win",
        ],
    ),
    (
        "Placement vs Monte-Carlo — top-cell agreement",
        [
            "eval_place_top1_opp_next",
            "eval_place_top1_self_next",
            "eval_place_top1_opp_win",
            "eval_place_top1_self_win",
        ],
    ),
]
# The leave inference model's read (the record scribblez/leave_inference/
# trainer.py assembles, its eval metrics from train_loop.py):
# held-out NLL of the opponent's true leave against the uniform prior (negative
# is better), broken down by leave length and bag size, the calibration of its
# implied letter presence and leave value, the same rows with every observer's
# private history hidden (as served for an opponent's view), and the win table
# it refits each generation against Macondo's.
LEAVE_INFERENCE_QUALITY = [
    ("NLL vs the uniform prior (nats per leave)", ["eval_nll_vs_uniform", "train_nll_vs_uniform"]),
    ("Held-out NLL (nats per leave)", ["eval_nll", "eval_hidden_nll", "eval_nll_uniform"]),
    ("Private history hidden: NLL gap to the own view (nats per leave)", ["eval_hidden_gap"]),
    (
        "Held-out NLL vs uniform, by leave length",
        [f"eval_nll_vs_uniform_len_{k}" for k in range(1, 7)],
    ),
    (
        "Held-out NLL vs uniform, by bag size",
        [f"eval_nll_vs_uniform_bag_{b}" for b in ("1_20", "21_40", "41_60", "61_100")],
    ),
    ("Letter calibration: mean |predicted - actual| presence", ["eval_letter_calibration"]),
    ("Leave value: mean predicted - true (points)", ["eval_value_bias"]),
    ("Leave value: true within one predicted sd", ["eval_value_coverage"]),
    ("Win table held-out log loss", ["eval_wintable_logloss", "eval_wintable_logloss_macondo"]),
    (
        "Rows without information",
        ["eval_invalid_no_opponent_move", "eval_invalid_empty_leave", "eval_invalid_pass"],
    ),
]
# The leaf eval read (deep_bot_leaf_eval/evaluation.py): on the held-out games'
# rows, the model's log loss of the mover's expected score beside the win
# tables' (the one fitted from the window, and Macondo's), overall, by pool
# size and by whether both refills empty the bag, the model's own WLD and
# score-differential errors, and its per-square and end-of-game heads' errors.
# deep_bot_leaf_eval/evaluation.py's POOL_BANDS and BAG_SPLITS, copied:
# importing it would load the engine library.
LEAF_EVAL_POOL_BANDS = ("0_20", "21_40", "41_60", "61_100")
LEAF_EVAL_BAG_SPLITS = (("bag_empty", "refills empty the bag"), ("bag_left", "bag left"))
LEAF_EVAL_SOURCES = ("model", "wintable", "wintable_macondo")
LEAF_EVAL_QUALITY = [
    (
        "Held-out log loss of the expected score: model vs win tables",
        [f"eval_{source}_logloss" for source in LEAF_EVAL_SOURCES],
    ),
    *(
        (
            f"Held-out log loss, pool of {band.replace('_', '-')} tiles",
            [f"eval_{source}_logloss_pool_{band}" for source in LEAF_EVAL_SOURCES],
        )
        for band in LEAF_EVAL_POOL_BANDS
    ),
    *(
        (
            f"Held-out log loss, {label}",
            [f"eval_{source}_logloss_{split}" for source in LEAF_EVAL_SOURCES],
        )
        for split, label in LEAF_EVAL_BAG_SPLITS
    ),
    ("Held-out WLD cross-entropy", ["eval_wld_nll"]),
    ("Held-out score-differential MAE (points)", ["eval_sd_mean_mae"]),
    ("Held-out ownership cross-entropy per empty square", ["eval_ownership_nll"]),
    ("Held-out square points RMSE (points)", ["eval_points_rmse"]),
    ("Square points: predicted variance / squared error", ["eval_points_var_ratio"]),
    (
        "Held-out going-out cross-entropy, overall, by pool and by bag",
        [
            "eval_going_out_nll",
            *(f"eval_going_out_nll_pool_{b}" for b in LEAF_EVAL_POOL_BANDS),
            *(f"eval_going_out_nll_{split}" for split, _ in LEAF_EVAL_BAG_SPLITS),
        ],
    ),
    ("Held-out end-of-game points RMSE (points)", ["eval_end_points_rmse"]),
    ("End-of-game points: predicted variance / squared error", ["eval_end_points_var_ratio"]),
]


def _beside_release_0(title: str, key: str, *others: str):
    """A multi-model trainer's chart group: the newest release's `key`
    beside release 0's on the same games, and any `others`."""
    return (title, [key, f"r0_{key}", *others])


# A multi-model trainer's read (leaf_bot_generation/trainer.py's records):
# each generation's games scored before the trainer trains on them, by the
# newest release's models and by release 0's (the "r0_" prefix). The gap
# between the two curves is the tag's gain over its seeds on that
# generation's games.
LEAVE_INFERENCE_SCORES = [
    _beside_release_0(
        "Leave inference: NLL vs the uniform prior (nats per leave)", "leave_eval_nll_vs_uniform"
    ),
    _beside_release_0(
        "Leave inference, private history hidden: NLL vs uniform",
        "leave_eval_hidden_nll_vs_uniform",
    ),
]
LEAF_NETWORK_SCORES = [
    _beside_release_0(
        "Leaf network: log loss of the expected score",
        "leaf_eval_model_logloss",
        "leaf_eval_wintable_macondo_logloss",
    ),
    _beside_release_0("Leaf network: WLD cross-entropy", "leaf_eval_wld_nll"),
    _beside_release_0(
        "Leaf network: ownership cross-entropy per empty square", "leaf_eval_ownership_nll"
    ),
]


def training_by_model(kinds: tuple[str, ...]) -> list[tuple]:
    """A multi-model trainer's training curves, each model's beside the
    others' (deep_bot/unit_set.py's record keys)."""
    return [
        ("Training loss, by model", [f"loss_{k}" for k in kinds], {"log": True}),
        ("Learning rate, by model", [f"{k}_lr" for k in kinds], {"log": True}),
        ("Training time (s), by model", [f"{k}_train_s" for k in kinds]),
    ]


LEAF_BOT_GENERATION_QUALITY = [
    *LEAVE_INFERENCE_SCORES,
    *LEAF_NETWORK_SCORES,
    # FP16 serves the leaf network: an activation past 65504 overflows
    # (scribblez/norm_inputs.py).
    _beside_release_0("Leaf network: largest activation", "leaf_activation_peak"),
    *training_by_model(("leave", "leaf")),
]
# Figures per row of every learning-curve grid.
SERIES_NCOLS = 2


def _grid(figs):
    """`figs` laid out SERIES_NCOLS to a row, stretched to the page's width.
    Each figure keeps its own toolbar; the grid gets none."""
    return gridplot(
        figs,
        ncols=SERIES_NCOLS,
        sizing_mode="stretch_width",
        merge_tools=False,
        toolbar_location=None,
    )


def series_grid(conn, groups, smooth: bool = False):
    """A grid of learning-curve figures for one tag, one per metric group that
    has data. `smooth` draws each curve as its EMA."""
    sources = [(conn, "")]
    figs = []
    for title, names, *rest in groups:
        opts = rest[0] if rest else {}
        f = _series_figure(sources, title, names, log=opts.get("log", False), smooth=smooth)
        if f is not None:
            figs.append(f)
    if not figs:
        return Div(text="<i>No scalar metrics recorded yet.</i>")
    return _grid(figs)


# The Loss tab's figures carry every knob variant as a pre-built named row, so
# the tab's knobs (linear/log x; the loss figure's absolute/percent) just flip
# row visibility inside the embedded document, with no round trip or re-embed.
# The value-quality figure has the two x-axis rows; the loss figure crosses them
# with the normalizations as "<x axis>|<norm>". web/src/components/
# TrainingTabs.tsx addresses the rows by these names; keep the two in sync.
X_AXIS_LINEAR = "x_linear"
X_AXIS_LOG = "x_log"
NORM_ABSOLUTE = "abs"
NORM_PERCENT = "pct"


def _variant_rows(builders):
    """A Loss-tab figure's knob-variant rows, stacked and named: `builders` maps
    each row name to its zero-arg row builder."""
    rows = []
    for name, build in builders.items():
        r = build()
        r.name = name
        rows.append(r)
    return column(*rows, sizing_mode="stretch_width")


def _quality_row(sources, smooth, log_x):
    """One x-axis variant of the value-quality panel."""
    figs = [
        f
        for title, group in POST_MOVE_QUALITY
        if (f := _series_figure(sources, title, group, smooth=smooth, log_x=log_x))
    ]
    return _grid(figs)


def eval_quality_grid(conn, tag: str, smooth: bool = False, secondary=None):
    """The POST_MOVE_QUALITY curves in both x-axis variants, or None when the tag
    has recorded none yet (the Loss tab then omits the panel). `smooth` draws each
    curve as its EMA. `secondary`, a (conn, tag) pair, overlays that tag's curves
    for comparison, with each legend label suffixed by its tag."""
    names = [name for _title, group in POST_MOVE_QUALITY for name in group]
    if not any(len(db.read_metric_series(conn, name)[0]) for name in names):
        return None
    if secondary is not None:
        sec_conn, sec_tag = secondary
        sources = [(conn, f" [{tag}]"), (sec_conn, f" [{sec_tag}]")]
    else:
        sources = [(conn, "")]
    return _variant_rows(
        {
            X_AXIS_LINEAR: partial(_quality_row, sources, smooth, False),
            X_AXIS_LOG: partial(_quality_row, sources, smooth, True),
        }
    )


# ---------------------------------------------------------------------------
# Match eval (per-generation match play vs a fixed opponent)
# ---------------------------------------------------------------------------


def _match_source(rows) -> ColumnDataSource:
    return ColumnDataSource(
        {
            "x": [r["epoch"] for r in rows],
            "score": [r["score"] for r in rows],
            "lower": [r["score"] - r["ci_half_width"] for r in rows],
            "upper": [r["score"] + r["ci_half_width"] for r in rows],
            "games": [r["games"] for r in rows],
            "wdl": [f"{r['wins']}/{r['draws']}/{r['losses']}" for r in rows],
        }
    )


_MATCH_TOOLTIPS = [
    ("generation", "@x"),
    ("score", "@score{0.000}"),
    ("games", "@games"),
    ("W/D/L", "@wdl"),
]


def _dashed_hline(fig, location: float, color: str = "#888888"):
    fig.add_layout(Span(location=location, dimension="width", line_color=color, line_dash="dashed"))


def match_eval_grid(conn):
    """The Match tab: mean pair score per generation against the fixed opponent,
    with its CI band. None when no match has been recorded."""
    rows = db.read_all_match_eval(conn)
    if not rows:
        return None
    src = _match_source(rows)
    opponents = " / ".join(sorted({r["opponent"] for r in rows}))

    fig = figure(
        sizing_mode="scale_width",
        aspect_ratio=WIDE,
        title=f"Match win rate vs {opponents}",
        x_axis_label="generation",
        y_axis_label="pair score",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    fig.varea(x="x", y1="lower", y2="upper", source=src, fill_alpha=0.15)
    fig.line(x="x", y="score", source=src, line_width=2)
    dots = fig.scatter(x="x", y="score", source=src, size=8)
    fig.add_tools(HoverTool(tooltips=_MATCH_TOOLTIPS, renderers=[dots]))
    _dashed_hline(fig, 0.5)
    return fig


# How many releases the release-match chart's trailing mean covers: one
# match's standard error is several points, a window's about one.
RELEASE_MATCH_WINDOW = 10

# A release's opponents (deep_bot/matches.py), as the chart titles them.
RELEASE_OPPONENT_TITLES = {"hastybot": "HastyBot", "release_0": "release 0"}


def trailing_mean(values: list[float], window: int) -> list[float]:
    """Each value's mean with the up to `window` - 1 before it."""
    return [
        sum(values[max(0, i + 1 - window) : i + 1]) / min(i + 1, window) for i in range(len(values))
    ]


def _release_match_figure(opponent: str, rows: list[dict]):
    """One opponent's panel: each release's win rate with its CI band, and
    the trailing mean over RELEASE_MATCH_WINDOW releases."""
    src = _match_source([r | {"epoch": r["release"]} for r in rows])
    src.data["trailing"] = trailing_mean(src.data["score"], RELEASE_MATCH_WINDOW)
    fig = figure(
        sizing_mode="scale_width",
        aspect_ratio=WIDE,
        title=f"Win rate vs {RELEASE_OPPONENT_TITLES.get(opponent, opponent)}",
        x_axis_label="release",
        y_axis_label="pair score",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    fig.varea(x="x", y1="lower", y2="upper", source=src, fill_alpha=0.15)
    dots = fig.scatter(x="x", y="score", source=src, size=6, legend_label="release")
    fig.line(
        x="x",
        y="trailing",
        source=src,
        line_width=2,
        legend_label=f"mean of the last {RELEASE_MATCH_WINDOW}",
    )
    tooltips = [
        ("release", "@x"),
        *_MATCH_TOOLTIPS[1:],
        ("last releases' mean", "@trailing{0.000}"),
    ]
    fig.add_tools(HoverTool(tooltips=tooltips, renderers=[dots]))
    fig.legend.location = "bottom_right"
    _dashed_hline(fig, 0.5)
    return fig


def release_match_grid(conn):
    """leaf_bot_generation's Matches tab: a panel per opponent. None when no match has
    been recorded."""
    rows = db.read_release_matches(conn)
    if not rows:
        return None
    by_opponent: dict[str, list[dict]] = {}
    for r in rows:
        by_opponent.setdefault(r["opponent"], []).append(r)
    return column(
        [_release_match_figure(name, by_opponent[name]) for name in sorted(by_opponent)],
        sizing_mode="scale_width",
    )


def match_arms_grid(conn):
    """The Arms tab: each arm's mean pair score against the fixed opponent, with
    its CI whisker, and under it the arm's throughput, in declared arm order.
    None when no arm has been measured."""
    rows = db.read_all_match_arms(conn)
    if not rows:
        return None
    scores = _arm_scores(rows)
    return column(scores, _arm_throughput(rows, scores.x_range), sizing_mode="stretch_width")


def _arm_scores(rows: list[dict]):
    """Each arm's mean pair score, with its CI whisker."""
    names = [r["arm"] for r in rows]
    src = ColumnDataSource(
        {
            "x": names,
            "score": [r["score"] for r in rows],
            "lower": [r["score"] - r["ci_half_width"] for r in rows],
            "upper": [r["score"] + r["ci_half_width"] for r in rows],
            "games": [r["games"] for r in rows],
            "wdl": [f"{r['wins']}/{r['draws']}/{r['losses']}" for r in rows],
            "spec": [r["player_spec"] for r in rows],
        }
    )
    opponents = " / ".join(sorted({r["opponent"] for r in rows}))
    fig = figure(
        sizing_mode="scale_width",
        aspect_ratio=WIDE,
        x_range=names,
        title=f"Arm win rates vs {opponents}",
        x_axis_label="arm",
        y_axis_label="pair score",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    fig.add_layout(Whisker(source=src, base="x", upper="upper", lower="lower"))
    dots = fig.scatter(x="x", y="score", source=src, size=9)
    fig.add_tools(
        HoverTool(
            tooltips=[
                ("arm", "@x"),
                ("score", "@score{0.000}"),
                ("games", "@games"),
                ("W/D/L", "@wdl"),
                ("player", "@spec"),
            ],
            renderers=[dots],
        )
    )
    _dashed_hline(fig, 0.5)
    return fig


def _arm_throughput(rows: list[dict], x_range):
    """Each arm's games per minute: its games over the wall time of its whole
    match, both players' moves and each round's play_game start included. The
    opponent's share is the same for every arm, so differences are the arm's."""
    src = ColumnDataSource(
        {
            "x": [r["arm"] for r in rows],
            "rate": [60.0 * r["games"] / r["elapsed_s"] for r in rows],
            "per_game": [r["elapsed_s"] / r["games"] for r in rows],
            "minutes": [r["elapsed_s"] / 60.0 for r in rows],
            "games": [r["games"] for r in rows],
        }
    )
    fig = figure(
        sizing_mode="scale_width",
        aspect_ratio=WIDE,
        x_range=x_range,
        title="Arm throughput",
        x_axis_label="arm",
        y_axis_label="games per minute",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    bars = fig.vbar(x="x", top="rate", source=src, width=0.6)
    fig.y_range.start = 0
    fig.add_tools(
        HoverTool(
            tooltips=[
                ("arm", "@x"),
                ("games/min", "@rate{0.0}"),
                ("s/game", "@per_game{0.00}"),
                ("match", "@games games in @minutes{0.0} min"),
            ],
            renderers=[bars],
        )
    )
    return fig


# ---------------------------------------------------------------------------
# Loss / accuracy (per-checkpoint metrics over epoch)
# ---------------------------------------------------------------------------


def _epoch_figure(title: str, x, y_label: str, log_x: bool):
    """An empty square Loss-tab figure over the epoch axis `x`. A log axis gets
    an explicit positive range so an epoch-0 point does not break it."""
    fig = figure(
        sizing_mode="scale_width",
        aspect_ratio=SQUARE,
        title=title,
        x_axis_label="epoch",
        y_axis_label=y_label,
        x_axis_type="log" if log_x else "linear",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    if log_x:
        _set_padded_range(fig, "x", x, log=True)
    return fig


def _step_figure(title: str, x, series, y_label: str, log_x: bool = False):
    fig = _epoch_figure(title, x, y_label, log_x)
    fig.add_tools(HoverTool(tooltips=[("epoch", "@x"), ("value", "@y{0.0000}")], mode="vline"))
    palette = Category10[10]
    xs = list(x)
    for i, (y, label) in enumerate(series):
        src = ColumnDataSource(dict(x=xs, y=list(y)), name=_source_name(title, label))
        fig.line(
            "x",
            "y",
            source=src,
            color=palette[i % len(palette)],
            line_width=1.5,
            legend_label=label,
        )
    fig.legend.label_text_font_size = "8pt"
    fig.legend.click_policy = "hide"
    return fig


def _stacked_loss_figure(x, bands, title: str, y_label: str, log_x: bool):
    """Stacked area of per-component losses, `bands` = (label, y) bottom to top.
    Legend entries hide on click, but the stack is precomputed: a band left
    visible keeps its stacked position rather than dropping to zero."""
    fig = _epoch_figure(title, x, y_label, log_x)
    palette = Category10[10]
    xs = list(x)
    cum = np.zeros(len(xs), dtype=np.float64)
    for i, (label, y) in enumerate(bands):
        lo, hi = cum, cum + np.asarray(y, dtype=np.float64)
        src = ColumnDataSource(
            dict(x=xs, y1=list(lo), y2=list(hi)), name=_source_name(title, label)
        )
        fig.varea(
            x="x",
            y1="y1",
            y2="y2",
            source=src,
            fill_color=palette[i % len(palette)],
            fill_alpha=0.85,
            legend_label=label,
        )
        cum = hi
    fig.y_range.start = 0
    fig.legend.location = "top_right"
    fig.legend.label_text_font_size = "8pt"
    fig.legend.click_policy = "hide"
    return fig


def _loss_bands(series, weights, normalized):
    """Weighted loss bands [(label, y), ...], bottom to top in `weights` order,
    each labeled by its loss's name (the weights are the chart title's).
    `normalized` divides each column by its total, so band heights read as a
    share of the loss."""
    bands = [
        (name, np.asarray(series[name], dtype=np.float64) * w)
        for name, w in weights.items()
        if name in series
    ]
    if normalized and bands:
        total = sum(y for _, y in bands)
        total = np.where(total == 0.0, 1.0, total)  # leave all-zero columns at 0
        bands = [(label, y / total) for label, y in bands]
    return bands


def _loss_accuracy_row(x, series, weights, normalized, conn, pos_by_epoch, log_x):
    """One knob variant of the Loss tab's figure row: the loss panel (stacked
    weighted bands when loss weights were recorded, plain lines otherwise) with
    LR-change markers, plus an Accuracy panel of every '<x>_acc' series."""
    if weights:
        title, y_label = (
            ("Train loss (stacked, % of total)", "fraction of total loss")
            if normalized
            else ("Train loss (stacked, weighted)", "loss")
        )
        loss_fig = _stacked_loss_figure(
            x, _loss_bands(series, weights, normalized), title, y_label, log_x
        )
    else:
        loss_names = [k for k in ("loss",) if k in series] + sorted(
            k for k in series if k.startswith("loss_")
        )
        loss_fig = _step_figure(
            "Train loss", x, [(series[k], k) for k in loss_names], "loss", log_x
        )
    add_control_markers(loss_fig, conn, x, pos_by_epoch)
    figs = [loss_fig]
    acc_names = sorted(k for k in series if k.endswith("_acc"))
    if acc_names:
        figs.append(
            _step_figure("Accuracy", x, [(series[k], k) for k in acc_names], "accuracy", log_x)
        )
    return _grid(figs)


def add_control_markers(fig, conn, epochs, pos_by_epoch):
    """Mark each LR change the run recorded (its 'lr' control events) with a
    dashed vertical line labeled with the new rate, so the loss curve shows
    where a decay started or a restart landed.

    Events are recorded on the rows clock, not by epoch, so each is placed by
    interpolating against `pos_by_epoch` (epoch -> rows trained)."""
    events = db.read_control_events(conn, "lr")
    if not events:
        return
    positions = np.array([pos_by_epoch[e] for e in epochs], dtype=np.float64)
    for e in events:
        loc = float(np.interp(e["positions"], positions, epochs))
        fig.add_layout(
            Span(
                location=loc,
                dimension="height",
                line_color="#a05a00",
                line_dash="dashed",
                line_width=1,
            )
        )
        fig.add_layout(
            Label(
                x=loc,
                y=6,
                y_units="screen",
                text=f"{e['value']:.0e}",
                text_font_size="8pt",
                text_color="#a05a00",
                x_offset=2,
            )
        )


def _metrics_series(conn):
    """(x, series, pos_by_epoch): the epochs that recorded a rows clock, the
    'loss', 'loss_<head>' and '<x>_acc' series aligned to them (NaN where a
    series lacks an epoch), and epoch -> rows trained for placing control
    markers. (None, {}, {}) when nothing is recorded."""
    pos_by_epoch = dict(zip(*db.read_metric_series(conn, "positions"), strict=True))
    if not pos_by_epoch:
        return None, {}, {}
    epochs = sorted(pos_by_epoch)
    x = np.array(epochs, dtype=np.float64)
    series = {}
    for name in db.read_metric_names(conn):
        if name == "loss" or name.startswith("loss_") or name.endswith("_acc"):
            by_epoch = dict(zip(*db.read_metric_series(conn, name), strict=True))
            series[name] = np.array([by_epoch.get(e, np.nan) for e in epochs], dtype=np.float64)
    return x, series, pos_by_epoch


def metrics_loss_grid(conn):
    """The Loss tab's loss + accuracy row in all four knob variants ("<x
    axis>|<norm>"), or None when no loss metric exists."""
    x, series, pos_by_epoch = _metrics_series(conn)
    if not any(k == "loss" or k.startswith("loss_") for k in series):
        return None
    weights = db.read_loss_weights(conn)
    return _variant_rows(
        {
            f"{x_name}|{norm}": partial(
                _loss_accuracy_row,
                x,
                series,
                weights,
                norm == NORM_PERCENT,
                conn,
                pos_by_epoch,
                log_x,
            )
            for norm in (NORM_ABSOLUTE, NORM_PERCENT)
            for x_name, log_x in ((X_AXIS_LINEAR, False), (X_AXIS_LOG, True))
        }
    )

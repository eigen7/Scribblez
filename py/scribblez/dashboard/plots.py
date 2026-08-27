"""Bokeh plot builders for the training dashboard: scalar learning curves over
epochs (square figures, server-rendered)."""

from __future__ import annotations

from functools import partial

import numpy as np
from bokeh.layouts import column, row
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

SERIES_SIZE = 800  # square-ish learning-curve figures


# ---------------------------------------------------------------------------
# Scalar learning curves
# ---------------------------------------------------------------------------


# Below this many points an EMA is more misleading than helpful (its debiased head
# just traces the raw points), so smoothing is a no-op until a curve has at least
# this many samples -- the raw series is drawn as-is.
_SMOOTH_MIN_POINTS = 10


def _ema(values, weight: float = 0.85):
    """TensorBoard-style debiased exponential moving average of a 1-D array, for
    reading the trend of a noisy per-checkpoint curve. `weight` in [0, 1) sets the
    smoothing (higher = smoother); the debias term cancels the zero-initialization
    bias so the early points aren't dragged toward zero."""
    out = np.empty(len(values), dtype=np.float64)
    smoothed = 0.0
    debias = 0.0
    for i, v in enumerate(values):
        smoothed = smoothed * weight + (1.0 - weight) * float(v)
        debias = debias * weight + (1.0 - weight)
        out[i] = smoothed / debias if debias else float(v)
    return out


def _plot_series(fig, x, y, color, label, smooth, name=None):
    """Draw one metric series, its data source named `name` (see `_source_name`).
    With smoothing (once the series has at least `_SMOOTH_MIN_POINTS` points) the
    plotted line is a debiased exponential moving average, so the trend reads
    clearly; without it, the raw points are drawn as a line plus markers. A single
    line is drawn either way -- no faint raw underlay -- so a smoothed curve reads
    as unambiguously smooth."""
    if smooth and len(y) >= _SMOOTH_MIN_POINTS:
        src = ColumnDataSource(dict(x=x, y=_ema(np.asarray(y, dtype=np.float64))), name=name)
        fig.line("x", "y", source=src, color=color, line_width=2, legend_label=label)
    else:
        src = ColumnDataSource(dict(x=x, y=y), name=name)
        fig.line("x", "y", source=src, color=color, line_width=2, legend_label=label)
        fig.scatter("x", "y", source=src, color=color, size=4)


def _source_name(title: str, label: str) -> str:
    """The stable name of one series' data source: "<figure title>|<legend label>",
    unique within a figure row. The linear and log x-axis rows deliberately share
    names -- their sources hold identical data, so one incremental update (see
    figure_delta.py) feeds both."""
    return f"{title}|{label}"


def _padded_range(values, log):
    """An explicit padded Range1d over `values`, so a (near-)constant series is
    not drawn against Bokeh's degenerate default (which spans roughly value +/- 1,
    burying e.g. a flat 1e-3 learning rate in a [-1, 1] band). Log axes pad
    multiplicatively and clamp to positive data; linear axes pad additively.
    None when no finite (log: positive) value exists."""
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
    """Give `fig`'s `axis` ('x' | 'y') the explicit padded range of `values` (see
    `_padded_range`); leave Bokeh's auto-range when there is nothing to fit."""
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
    """A square learning-curve figure of the metric `names`, each drawn once per
    entry in `sources` -- a list of (conn, label_suffix). Every (source, metric)
    pair gets its own color, and the legend suffix (e.g. ' [tagB]') names the
    source, so a second tag's curves overlay the first as distinctly colored,
    distinctly labeled lines for comparison. `log` / `log_x` draw the y / epoch
    axis logarithmically (a log epoch axis gets an explicit positive range, so an
    epoch-0 checkpoint does not pin it). None when no source has any of the
    metrics."""
    fig = figure(
        width=SERIES_SIZE,
        height=SERIES_SIZE,
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


# Per-epoch scalar-curve groups, by dashboard tab. Each entry is (figure title,
# metric-series names) and feeds series_grid(). Shared by the dashboard's tab
# builders (Bokeh shell and the React data API).
LOSS = [
    (
        "Loss",
        [
            "loss",
            "loss_wld",
            "loss_score_diff",
            "loss_opp_next_placement",
            "loss_self_next_placement",
            "loss_opp_win_placement",
            "loss_self_win_placement",
        ],
    )
]
# The learning rate spans orders of magnitude, so it reads best on a log y-axis.
# The averaging-weight panel is there for schedule-free runs (the arm records
# it, a WSD run does not, and the panel is then absent): that arm holds its rate
# constant and anneals by giving each new base iterate a smaller share of the
# deployed average, so the weight is the curve the rate would otherwise show.
TRAINING = [
    ("Learning rate", ["lr"], {"log": True}),
    ("Iterate averaging weight", ["averaging_weight"], {"log": True}),
    ("Epoch time (s)", ["elapsed_s"]),
]
# Move-set-eval distillation quality: the teacher win-equity the student's
# top-K forfeits (lower is better), with the incumbent ranking's (played
# move, then equity head) regret@1 as the flat reference line. The
# recall/Spearman curves and their baselines ride the Loss tab's Accuracy
# panel instead.
# The evidence trainer's go/no-go read (docs/roadmap.md item 5): the
# conditioned pass against the plain one on the same held-out rows, overall
# and on the evidence-bearing (prefix > 0) rows, plus the proves-best head's
# gain error. The gain hit rates ride the Loss tab's Accuracy panel. An
# unfrozen run (the backbone trained jointly under the .mset distillation
# anchor) adds the frozen student's soft-CE as the flat reference its moving
# plain pass is read against, and the plain pass's distillation health on
# the .mset holdout; a frozen run records neither and those panels are absent.
EVIDENCE_QUALITY = [
    (
        "Held-out WLD soft-CE: student reference vs plain vs conditioned",
        ["student_wld_ce", "plain_wld_ce", "cond_wld_ce"],
    ),
    (
        "Held-out WLD soft-CE, evidence-bearing rows",
        ["student_wld_ce_ev", "plain_wld_ce_ev", "cond_wld_ce_ev"],
    ),
    ("Held-out value MAE: conditioned vs plain", ["cond_value_mae", "plain_value_mae"]),
    ("Proves-best gain MAE", ["gain_mae", "gain_mae_ev"]),
    (
        "Distillation health: held-out recall@1 / Spearman vs the teacher",
        ["distill_recall1", "distill_recall1_baseline", "distill_spearman"],
    ),
    (
        "Distillation health: held-out loss",
        ["distill_loss", "distill_loss_wld", "distill_plane_bce"],
    ),
    # The hand-maintained position set (positions/NWL23/face-up-trajectory-set,
    # what the Trajectories tab shows): over every position and evidence
    # prefix, the rank of the sim-best simmed candidate under the conditioned
    # value vs the plain one (0 = best; lower is better), and how often each
    # value's argmax over the simmed candidates is that sim-best one.
    (
        "Position set: sim-best rank, conditioned vs plain",
        ["posset_cond_rank", "posset_plain_rank"],
    ),
    ("Position set: sim-best hit rate", ["posset_cond_hit", "posset_plain_hit"]),
]
MSET_QUALITY = [
    ("Teacher-value regret (win-equity)", ["regret1", "regret3", "regret5", "regret1_baseline"]),
    # The exchange slice (the A4 dedicated-head readout): how well the student
    # ranks WHICH tiles to keep, against the incumbent leave-value ordering,
    # and how often the teacher's best exchange survives the global top-K.
    (
        "Exchange rank regret (win-equity)",
        ["exch_rank_regret", "exch_rank_regret_baseline"],
    ),
    (
        "Best-exchange retention@K",
        ["exch_retention1", "exch_retention3", "exch_retention5", "exch_retention1_baseline"],
    ),
]
# Aggregate model-vs-Monte-Carlo quality curves over the large position-evaluation
# dataset, shown on the Loss tab beneath the training curves. Lower is better for all:
# how far the model's predicted value is from the Monte-Carlo ground truth, split by
# head (win/draw/loss vs. score-differential mean/std).
POST_MOVE_QUALITY = [
    ("Value quality vs Monte-Carlo — WLD", ["eval_win_mae", "eval_wld_brier"]),
    (
        "Value quality vs Monte-Carlo — score diff (points)",
        ["eval_sd_mean_mae", "eval_sd_std_mae"],
    ),
]


def series_grid(conn, groups, ncols: int = 3, smooth: bool = False):
    """A grid of square learning-curve figures for a single tag. Each group is
    (title, metric-names) or (title, metric-names, opts), where opts may set
    {"log": True} for a log y-axis. `smooth` overlays a debiased-EMA trend line on
    each noisy curve."""
    sources = [(conn, "")]
    figs = []
    for title, names, *rest in groups:
        opts = rest[0] if rest else {}
        f = _series_figure(sources, title, names, log=opts.get("log", False), smooth=smooth)
        if f is not None:
            figs.append(f)
    if not figs:
        return Div(text="<i>No scalar metrics recorded yet.</i>")
    rows = [row(*figs[i : i + ncols]) for i in range(0, len(figs), ncols)]
    return column(*rows)


# The Loss tab's figures carry EVERY knob variant as pre-built named rows, so the
# tab's knobs (Linear x/Log x; the loss figure's Absolute/%) flip the rows'
# visibility inside the embedded BokehJS document -- no round trip to this API,
# no re-embed. The value-quality figure has the two x-axis rows; the loss figure
# crosses them with the normalization variants as "<x axis>|<norm>". The web
# client (TrainingTabs.tsx) addresses the rows by these names; change them in
# both places.
X_AXIS_LINEAR = "x_linear"
X_AXIS_LOG = "x_log"
NORM_ABSOLUTE = "abs"
NORM_PERCENT = "pct"


def _variant_rows(builders):
    """The named knob-variant rows of a Loss-tab figure, stacked: `builders` maps
    each row name (see the constants above) to its zero-arg row builder."""
    rows = []
    for name, build in builders.items():
        r = build()
        r.name = name
        rows.append(r)
    return column(*rows)


def _quality_row(sources, smooth, log_x):
    """One x-axis variant of the value-quality row: a figure per POST_MOVE_QUALITY
    group that any source has data for."""
    figs = [
        f
        for title, group in POST_MOVE_QUALITY
        if (f := _series_figure(sources, title, group, smooth=smooth, log_x=log_x))
    ]
    return row(*figs)


def eval_quality_grid(conn, tag: str, smooth: bool = False, secondary=None):
    """The aggregate model-vs-Monte-Carlo quality curves over checkpoints, in both
    x-axis variants (`_x_axis_variants`), or None when the primary tag has recorded
    no quality metric yet (so the Loss tab can omit the panel rather than show an
    empty placeholder). `smooth` overlays an EMA trend on each curve (they are noisy
    checkpoint-to-checkpoint). `secondary`, when given as (conn, tag), overlays that
    tag's curves in their own colors for comparison; the legend labels are then
    suffixed with each tag."""
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
    """The Match tab: the win-rate curve against the fixed opponent -- mean
    pair score with its CI band, the 0.5 line dashed. None when no match has
    been recorded."""
    rows = db.read_all_match_eval(conn)
    if not rows:
        return None
    src = _match_source(rows)
    opponents = " / ".join(sorted({r["opponent"] for r in rows}))

    fig = figure(
        width=2 * SERIES_SIZE,
        height=SERIES_SIZE,
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


def match_arms_grid(conn):
    """The Arms tab: each arm's mean pair score with its CI whisker against the
    experiment's fixed opponent, in the experiment's declared arm order, the
    0.5 line dashed. None when no arm has been measured."""
    rows = db.read_all_match_arms(conn)
    if not rows:
        return None
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
        width=2 * SERIES_SIZE,
        height=SERIES_SIZE,
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


# ---------------------------------------------------------------------------
# Loss / accuracy (per-checkpoint metrics over positions trained)
# ---------------------------------------------------------------------------


def _positions_figure(title: str, x, y_label: str, log_x: bool):
    """The Loss tab's square figure shell over the positions-trained x-axis `x`:
    `log_x` draws that axis logarithmically, with an explicit positive range so
    the auto-range does not degrade on a positions=0 first checkpoint."""
    fig = figure(
        width=SERIES_SIZE,
        height=SERIES_SIZE,
        title=title,
        x_axis_label="positions",
        y_axis_label=y_label,
        x_axis_type="log" if log_x else "linear",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    if log_x:
        _set_padded_range(fig, "x", x, log=True)
    return fig


def _step_figure(title: str, x, series, y_label: str, log_x: bool = False):
    fig = _positions_figure(title, x, y_label, log_x)
    fig.add_tools(HoverTool(tooltips=[("positions", "@x"), ("value", "@y{0.0000}")], mode="vline"))
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
    """Stacked area of per-component losses, `bands` = (label, y) bottom-to-top.
    Click a legend entry to hide it -- hide all but one to read a single
    component's own curve (from zero)."""
    fig = _positions_figure(title, x, y_label, log_x)
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
    """Weighted per-component loss bands [(label, y), ...] bottom-to-top, drawn
    from the aligned `series` dict (name -> y-array). When `normalized`, each point
    is divided by that point's stack total, so every column sums to 1 and band
    heights read as a share of the loss."""
    bands = [
        (name if w == 1 else f"{w:g} x {name}", np.asarray(series[name], dtype=np.float64) * w)
        for name, w in weights.items()
        if name in series
    ]
    if normalized and bands:
        total = sum(y for _, y in bands)
        total = np.where(total == 0.0, 1.0, total)  # leave all-zero columns at 0
        bands = [(label, y / total) for label, y in bands]
    return bands


def _loss_accuracy_row(x, series, weights, normalized, conn, log_x):
    """The Loss tab's figure row over aligned per-point `series` (name -> y-array)
    and x-axis `x`: a stacked area of the WEIGHTED per-component losses -- band
    heights show each term's share of the optimized total, and `normalized`
    rescales every column to sum to 1 -- when loss coefficients (`weights`) were
    recorded, else overlaid loss lines; plus an Accuracy panel for every '<x>_acc'
    series. Both panels share the positions x-axis, logarithmic when `log_x`.
    LR-change markers overlay the loss panel."""
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
    add_control_markers(loss_fig, conn)
    figs = [loss_fig]
    acc_names = sorted(k for k in series if k.endswith("_acc"))
    if acc_names:
        figs.append(
            _step_figure("Accuracy", x, [(series[k], k) for k in acc_names], "accuracy", log_x)
        )
    return row(*figs)


def add_control_markers(fig, conn):
    """Overlay dashed vertical markers on a positions-axis figure at each LR-schedule
    phase boundary (from the control_event table), labeled with the rate there, so
    the loss curve shows where a decay started or a restart landed. A no-op when the
    run recorded none."""
    for e in db.read_control_events(conn, "lr"):
        fig.add_layout(
            Span(
                location=e["positions"],
                dimension="height",
                line_color="#a05a00",
                line_dash="dashed",
                line_width=1,
            )
        )
        fig.add_layout(
            Label(
                x=e["positions"],
                y=6,
                y_units="screen",
                text=f"{e['value']:.0e}",
                text_font_size="8pt",
                text_color="#a05a00",
                x_offset=2,
            )
        )


def _metrics_series(conn):
    """The metrics table's loss and accuracy series as an aligned {name: y-array}
    dict over a shared positions x-axis. Only 'loss', 'loss_<head>', and '<x>_acc'
    metrics are collected; they are co-written per checkpoint, so all share the
    metrics table's epoch index. Returns (x, series) -- (None, {}) when nothing is
    recorded, and NaN for any epoch a series happens to miss."""
    pos_by_epoch = dict(zip(*db.read_metric_series(conn, "positions"), strict=True))
    if not pos_by_epoch:
        return None, {}
    epochs = sorted(pos_by_epoch)
    x = np.array([pos_by_epoch[e] for e in epochs], dtype=np.float64)
    series = {}
    for name in db.read_metric_names(conn):
        if name == "loss" or name.startswith("loss_") or name.endswith("_acc"):
            by_epoch = dict(zip(*db.read_metric_series(conn, name), strict=True))
            series[name] = np.array([by_epoch.get(e, np.nan) for e in epochs], dtype=np.float64)
    return x, series


def metrics_loss_grid(conn):
    """The Loss tab's stacked-loss + accuracy grid built from the per-checkpoint
    `metrics` table vs positions trained, in all four knob variants
    ("<x axis>|<norm>", see `_variant_rows`): stacked weighted per-component
    losses (the percent variants -> per-column fractions), an accuracy panel,
    control-change markers. None when no loss metric exists."""
    x, series = _metrics_series(conn)
    if not any(k == "loss" or k.startswith("loss_") for k in series):
        return None
    weights = db.read_loss_weights(conn)
    return _variant_rows(
        {
            f"{x_name}|{norm}": partial(
                _loss_accuracy_row, x, series, weights, norm == NORM_PERCENT, conn, log_x
            )
            for norm in (NORM_ABSOLUTE, NORM_PERCENT)
            for x_name, log_x in ((X_AXIS_LINEAR, False), (X_AXIS_LOG, True))
        }
    )

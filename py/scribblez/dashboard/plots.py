"""Bokeh plot builders for the training dashboard.

Two kinds of view:
  * Scalar learning curves over epochs (square figures, server-rendered).
  * Per-generation interactive views (probes, calibration): every generation's
    data is preloaded into the browser and a CustomJS slider/arrows swap what's
    shown, so scrubbing never hits the server.

Each interactive view returns a `View` bundling its layout with the stateful
widgets (generation slider, "latest" follow checkbox, position selector) so the
app can preserve scrub state across live rebuilds.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    Checkbox,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    InlineStyleSheet,
    Label,
    LinearAxis,
    Range1d,
    Select,
    Slider,
    Span,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure

from . import db

SERIES_SIZE = 800  # square-ish learning-curve figures


@dataclass
class View:
    """An interactive per-generation view plus its stateful widgets."""

    layout: object
    gen: Slider
    latest: Checkbox
    pos: Select | None = None


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


def _plot_series(fig, x, y, color, label, smooth):
    """Draw one metric series. With smoothing (once the series has at least
    `_SMOOTH_MIN_POINTS` points) the plotted line is a debiased exponential moving
    average, so the trend reads clearly; without it, the raw points are drawn as a
    line plus markers. A single line is drawn either way -- no faint raw underlay --
    so a smoothed curve reads as unambiguously smooth."""
    if smooth and len(y) >= _SMOOTH_MIN_POINTS:
        src = ColumnDataSource(dict(x=x, y=_ema(np.asarray(y, dtype=np.float64))))
        fig.line("x", "y", source=src, color=color, line_width=2, legend_label=label)
    else:
        src = ColumnDataSource(dict(x=x, y=y))
        fig.line("x", "y", source=src, color=color, line_width=2, legend_label=label)
        fig.scatter("x", "y", source=src, color=color, size=4)


def _set_y_range(fig, values, log):
    """Give the figure an explicit padded y-range, so a (near-)constant series is
    not drawn against Bokeh's degenerate default (which spans roughly value +/- 1,
    burying e.g. a flat 1e-3 learning rate in a [-1, 1] band). Log axes pad
    multiplicatively and clamp to positive data; linear axes pad additively."""
    finite = values[np.isfinite(values)]
    if log:
        finite = finite[finite > 0.0]
    if len(finite) == 0:
        return
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
    fig.y_range = Range1d(lo, hi)


def _series_figure(
    sources, title: str, names: list[str], *, log: bool = False, smooth: bool = False
):
    """A square learning-curve figure of the metric `names`, each drawn once per
    entry in `sources` -- a list of (conn, label_suffix). Every (source, metric)
    pair gets its own color, and the legend suffix (e.g. ' [tagB]') names the
    source, so a second tag's curves overlay the first as distinctly colored,
    distinctly labeled lines for comparison. None when no source has any of the
    metrics."""
    fig = figure(
        width=SERIES_SIZE,
        height=SERIES_SIZE,
        title=title,
        x_axis_label="epoch",
        y_axis_type="log" if log else "linear",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    fig.add_tools(HoverTool(tooltips=[("epoch", "@x"), ("value", "@y{0.0000}")], mode="vline"))
    palette = Category10[10]
    all_values = []
    for s, (conn, suffix) in enumerate(sources):
        for i, name in enumerate(names):
            epochs, values = db.read_metric_series(conn, name)
            if len(epochs) == 0:
                continue
            color = palette[(s * len(names) + i) % len(palette)]
            _plot_series(fig, epochs, values, color, name + suffix, smooth)
            all_values.append(values)
    if not all_values:
        return None
    _set_y_range(fig, np.concatenate(all_values), log)
    fig.legend.label_text_font_size = "8pt"
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    return fig


# Per-epoch scalar-curve groups, by dashboard tab. Each entry is (figure title,
# metric-series names) and feeds series_grid(). Shared by the dashboard's tab
# builders (Bokeh shell and the React data API).
LOSS = [("Loss", ["loss", "loss_wld", "loss_score_diff", "loss_opp_next_placement"])]
# The learning rate is stepped down multiplicatively (by hand from the Controls
# tab) and spans orders of magnitude, so it reads best on a log y-axis.
TRAINING = [("Learning rate", ["lr"], {"log": True}), ("Epoch time (s)", ["elapsed_s"])]
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
# Aggregate model-vs-Monte-Carlo quality curves over the large penultimate-bingo
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


def eval_quality_grid(conn, tag: str, smooth: bool = False, secondary=None):
    """The aggregate model-vs-Monte-Carlo quality curves over checkpoints, or None
    when the primary tag has recorded no quality metric yet (so the Loss tab can
    omit the panel rather than show an empty placeholder). `smooth` overlays an EMA
    trend on each curve (they are noisy checkpoint-to-checkpoint). `secondary`, when
    given as (conn, tag), overlays that tag's curves in their own colors for
    comparison; the legend labels are then suffixed with each tag."""
    names = [name for _title, group in POST_MOVE_QUALITY for name in group]
    if not any(len(db.read_metric_series(conn, name)[0]) for name in names):
        return None
    if secondary is not None:
        sec_conn, sec_tag = secondary
        sources = [(conn, f" [{tag}]"), (sec_conn, f" [{sec_tag}]")]
    else:
        sources = [(conn, "")]
    figs = [
        f
        for title, group in POST_MOVE_QUALITY
        if (f := _series_figure(sources, title, group, smooth=smooth))
    ]
    return column(row(*figs)) if figs else None


# ---------------------------------------------------------------------------
# Streaming throughput + backpressure (time series over positions trained)
# ---------------------------------------------------------------------------


def _throughput_figure(rows, title, series, y_label, scale=1.0):
    """A line figure of the named throughput columns vs positions trained."""
    fig = figure(
        width=SERIES_SIZE,
        height=SERIES_SIZE,
        title=title,
        x_axis_label="positions trained",
        y_axis_label=y_label,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    fig.add_tools(HoverTool(tooltips=[("positions", "@x"), ("value", "@y{0.0}")], mode="vline"))
    palette = Category10[10]
    x = [r["positions"] for r in rows]
    for i, (key, label) in enumerate(series):
        src = ColumnDataSource(dict(x=x, y=[r[key] * scale for r in rows]))
        color = palette[i % len(palette)]
        fig.line("x", "y", source=src, color=color, line_width=2, legend_label=label)
    fig.y_range.start = 0  # rates / cumulative waits are non-negative
    fig.legend.location = "top_left"
    fig.legend.label_text_font_size = "9pt"
    fig.legend.click_policy = "hide"
    return fig


def throughput_grid(conn):
    """Throughput rate + cumulative backpressure figures for the streaming run.

    The two backpressure curves are the C++/Python wait times: when the consumer
    (training) curve climbs faster, game generation is the bottleneck (CPU-bound);
    when the producer curve climbs faster, training is the bottleneck (GPU-bound).
    """
    rows = db.read_throughput(conn)
    if not rows:
        return Div(text="<i>No throughput data yet — start a streaming run.</i>")
    # positions/s == games/s (one position sampled per game), so a single curve.
    rate = _throughput_figure(
        rows,
        "Throughput",
        [("positions_per_s", "positions/s")],
        "positions per second",
    )
    backpressure = _throughput_figure(
        rows,
        "Backpressure — cumulative wait (s)",
        [
            ("consumer_blocked_ns", "training waits (CPU-bound ↑)"),
            ("producer_blocked_ns", "gen waits (GPU-bound ↑)"),
        ],
        "blocked time (s)",
        scale=1e-9,
    )
    return column(row(rate, backpressure))


# ---------------------------------------------------------------------------
# Per-minibatch loss / accuracy (streaming pipeline)
# ---------------------------------------------------------------------------


def _stride_idx(n: int, max_points: int):
    """A slice/index that thins `n` points down to at most `max_points` (keeps plots light)."""
    if n <= max_points:
        return slice(None)
    return np.arange(0, n, (n + max_points - 1) // max_points)


def _step_figure(title: str, x, series, y_label: str, x_label: str = "positions"):
    fig = figure(
        width=SERIES_SIZE,
        height=SERIES_SIZE,
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    fig.add_tools(HoverTool(tooltips=[(x_label, "@x"), ("value", "@y{0.0000}")], mode="vline"))
    palette = Category10[10]
    xs = list(x)
    for i, (y, label) in enumerate(series):
        src = ColumnDataSource(dict(x=xs, y=list(y)))
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


def _stacked_loss_figure(x, bands, title="Train loss (stacked, weighted)", y_label="loss"):
    """Stacked area of per-component losses, `bands` = (label, y) bottom-to-top.
    Click a legend entry to hide it -- hide all but one to read a single
    component's own curve (from zero)."""
    fig = figure(
        width=SERIES_SIZE,
        height=SERIES_SIZE,
        title=title,
        x_axis_label="positions",
        y_axis_label=y_label,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    palette = Category10[10]
    xs = list(x)
    cum = np.zeros(len(xs), dtype=np.float64)
    for i, (label, y) in enumerate(bands):
        lo, hi = cum, cum + np.asarray(y, dtype=np.float64)
        src = ColumnDataSource(dict(x=xs, y1=list(lo), y2=list(hi)))
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


def _loss_accuracy_grid(x, series, weights, normalized, conn):
    """The Loss tab's figure row over aligned per-point `series` (name -> y-array)
    and x-axis `x`: a stacked area of the WEIGHTED per-component losses -- band
    heights show each term's share of the optimized total, and `normalized`
    rescales every column to sum to 1 -- when loss coefficients (`weights`) were
    recorded, else overlaid loss lines; plus an Accuracy panel for every '<x>_acc'
    series. LR-change markers overlay the loss panel. Shared by the streaming
    per-minibatch view and the per-checkpoint metrics view."""
    if weights:
        title, y_label = (
            ("Train loss (stacked, % of total)", "fraction of total loss")
            if normalized
            else ("Train loss (stacked, weighted)", "loss")
        )
        loss_fig = _stacked_loss_figure(
            x, _loss_bands(series, weights, normalized), title=title, y_label=y_label
        )
    else:
        loss_names = [k for k in ("loss",) if k in series] + sorted(
            k for k in series if k.startswith("loss_")
        )
        loss_fig = _step_figure("Train loss", x, [(series[k], k) for k in loss_names], "loss")
    add_control_markers(loss_fig, conn)
    figs = [loss_fig]
    acc_names = sorted(k for k in series if k.endswith("_acc"))
    if acc_names:
        figs.append(_step_figure("Accuracy", x, [(series[k], k) for k in acc_names], "accuracy"))
    return column(row(*figs))


def add_control_markers(fig, conn):
    """Overlay dashed vertical markers on a positions-axis figure at each base-LR
    change (from the control_event table), labeled with the new value, so the loss
    curve shows where the operator stepped the learning rate. A no-op when the run
    recorded no control changes."""
    for e in db.read_control_events(conn, "base_lr"):
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


def metrics_loss_grid(conn, normalized: bool = False):
    """The Loss tab's stacked-loss + accuracy grid built from the per-checkpoint
    `metrics` table vs positions trained: the loss view for trainers that record
    per-epoch metrics rather than the streaming per-minibatch train_step curve
    (e.g. the generational trainer). Renders identically to train_step_grid --
    stacked weighted per-component losses (`normalized` -> per-column fractions),
    an accuracy panel, control-change markers. None when no loss metric exists."""
    x, series = _metrics_series(conn)
    if not any(k == "loss" or k.startswith("loss_") for k in series):
        return None
    return _loss_accuracy_grid(x, series, db.read_loss_weights(conn), normalized, conn)


def train_step_grid(conn, normalized: bool = False):
    """Streaming loss + accuracy curves vs. positions trained.

    Task-agnostic: it discovers the series recorded by whichever trainer wrote the
    DB. The loss panel is a stacked area of the WEIGHTED per-component losses when
    the trainer recorded their coefficients (see db.write_loss_weights) -- band
    heights then show each term's share of the optimized total -- and falls back
    to overlaid lines otherwise. `normalized` switches the stack to per-column
    fractions (each band = share of the total, regardless of overall scale). Every
    '<x>_acc' name goes on the accuracy panel. Returns None when no streaming data
    exists, so the caller can fall back to the per-checkpoint loss view (the disk
    pipeline records only the latter).

    Points are logged at an adaptive resolution (dense early, then aggregated;
    see TrainStepWriter), so the x-axis is positions trained -- the dense and
    aggregated regions then line up by true progress -- and a stride caps the
    rendered count.
    """
    ts = db.read_train_steps(conn)
    x_all = ts["positions"]
    if len(x_all) == 0:
        return None
    idx = _stride_idx(len(x_all), 4000)
    series = {k: v[idx] for k, v in ts.items()}
    return _loss_accuracy_grid(
        series["positions"], series, db.read_loss_weights(conn), normalized, conn
    )


# ---------------------------------------------------------------------------
# Generation slider + "latest" follow checkbox
# ---------------------------------------------------------------------------


def _label(text: str) -> Div:
    """A small inline control label."""
    return Div(text=f"<b style='font-size:13px;color:#445'>{text}</b>")


def _gen_controls(num_gens: int, init_gen: int, follow: bool):
    """A generation slider + 'latest' checkbox.

    The checkbox tracks "is the slider at the rightmost (newest) generation": sliding
    away unchecks it, sliding back rechecks it, and (re)checking it jumps to the newest.
    The slider is never disabled. ``latest.active`` (synced to the server) drives the
    server-side follow-on-new-epoch behavior.
    """
    end = max(num_gens - 1, 1)
    gen = Slider(start=0, end=end, value=(end if follow else init_gen), step=1, title="", width=560)
    latest = Checkbox(label="latest", active=follow)
    gen.js_on_change(
        "value",
        CustomJS(args=dict(latest=latest), code="latest.active = (cb_obj.value >= cb_obj.end);"),
    )
    latest.js_on_change(
        "active", CustomJS(args=dict(gen=gen), code="if (cb_obj.active) { gen.value = gen.end; }")
    )
    return gen, latest


# ---------------------------------------------------------------------------
# Probes view: board + monotonicity + score belief for one (position, gen)
# ---------------------------------------------------------------------------


def _board_uri(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def probes_view(
    conn,
    image_dir: Path,
    init_pos: int = 0,
    init_gen: int | None = None,
    follow: bool = True,
    external_gen: bool = False,
) -> View | None:
    # external_gen: drop the in-figure generation slider + latest checkbox (the React
    # dashboard drives the generation with its own GenerationSlider, re-fetching the
    # figure per generation). The position scrubber stays in the figure.
    epochs, diffs, win_rate, curve_scores = db.read_all_monotonicity(conn)
    _, _, _, bands = db.read_all_score_belief(conn)
    if not epochs or bands is None:
        return None

    g, n, r = win_rate.shape
    q = bands.shape[-1]
    diffs = diffs.astype(float)
    imgs = [_board_uri(Path(image_dir) / f"pos-{k:02d}.png") for k in range(n)]
    gi = (g - 1) if init_gen is None else max(0, min(init_gen, g - 1))
    k0 = max(0, min(init_pos, n - 1))

    board = Div(text="", width=560)
    info = Div(styles={"font-size": "13px", "color": "#445", "font-weight": "600"})

    mono_src = ColumnDataSource(dict(x=list(diffs), y=list(win_rate[gi, k0])))
    mono = figure(
        width=620,
        height=580,
        title="",
        x_range=Range1d(diffs[0], diffs[-1]),
        y_range=Range1d(-0.02, 1.02),
        x_axis_label="input score differential",
        y_axis_label="predicted win rate",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    mono.line([diffs[0], diffs[-1]], [0.5, 0.5], color="#cccccc", line_dash="dashed")
    mono.line([0, 0], [0, 1], color="#cccccc", line_dash="dashed")
    mono.line("x", "y", source=mono_src, color="#1f77b4", line_width=2)

    b0 = bands[gi, k0]
    c0 = float(np.mean(b0[:, q // 2] - diffs))  # best fit y = x + C to the median
    outer = ColumnDataSource(dict(x=list(diffs), y1=list(b0[:, 0]), y2=list(b0[:, q - 1])))
    inner = ColumnDataSource(dict(x=list(diffs), y1=list(b0[:, 1]), y2=list(b0[:, q - 2])))
    med = ColumnDataSource(dict(x=list(diffs), y=list(b0[:, q // 2])))
    fit = ColumnDataSource(dict(x=list(diffs), y=list(diffs + c0)))
    belief_yr = Range1d(float(b0.min()), float(b0.max()))
    belief = figure(
        width=620,
        height=580,
        title="",
        x_range=Range1d(diffs[0], diffs[-1]),
        y_range=belief_yr,
        x_axis_label="input score differential",
        y_axis_label="predicted final score differential",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    belief.varea("x", "y1", "y2", source=outer, fill_color="#1f77b4", fill_alpha=0.18)
    belief.varea("x", "y1", "y2", source=inner, fill_color="#1f77b4", fill_alpha=0.36)
    belief.line("x", "y", source=med, color="#08306b", line_width=1.5, legend_label="median")
    belief.line(
        "x",
        "y",
        source=fit,
        color="#e74c3c",
        line_width=1.5,
        line_dash="dashed",
        legend_label="best fit y = x + C",
    )
    belief.legend.location = "top_left"
    belief.legend.label_text_font_size = "9pt"

    gen, latest = _gen_controls(g, gi, follow)
    pos = Select(title="", options=[str(k) for k in range(n)], value=str(k0), width=80)

    update = CustomJS(
        args=dict(
            pos=pos,
            gen=gen,
            N=n,
            R=r,
            Q=q,
            xs=list(diffs),
            epochs=epochs,
            imgs=imgs,
            winrate=win_rate.astype(np.float32).ravel(),
            cscores=curve_scores.astype(np.float32).ravel(),
            bands=bands.astype(np.float32).ravel(),
            mono_src=mono_src,
            outer=outer,
            inner=inner,
            med=med,
            fit=fit,
            mono=mono,
            belief=belief,
            belief_yr=belief_yr,
            board=board,
            info=info,
        ),
        code="""
        const k = parseInt(pos.value)|0, gj = gen.value|0;
        const base = (gj*N + k)*R;
        const y = new Array(R);
        for (let i=0;i<R;i++) y[i] = winrate[base+i];
        mono_src.data = {x: xs, y: y}; mono_src.change.emit();
        const cb = (gj*N + k)*3;
        mono.title.text = `Monotonicity — R²=${cscores[cb+1].toFixed(2)}  viol=${cscores[cb+2]|0}`;

        const bb = (gj*N + k)*R*Q;
        const lo=[], hi=[], i1=[], i3=[], me=[];
        let ymin=1e18, ymax=-1e18, csum=0;
        for (let s=0;s<R;s++){
            const o = bb + s*Q;
            const a=bands[o], b=bands[o+Q-1], m=bands[o+(Q>>1)], c=bands[o+1], d=bands[o+Q-2];
            lo.push(a); hi.push(b); i1.push(c); i3.push(d); me.push(m); csum += (m - xs[s]);
            if (a<ymin) ymin=a; if (b>ymax) ymax=b;
        }
        outer.data={x:xs, y1:lo, y2:hi}; outer.change.emit();
        inner.data={x:xs, y1:i1, y2:i3}; inner.change.emit();
        med.data={x:xs, y:me}; med.change.emit();
        const C = csum / R; fit.data={x:xs, y: xs.map(v => v + C)}; fit.change.emit();
        const pad=(ymax-ymin)*0.05+1; belief_yr.start=ymin-pad; belief_yr.end=ymax+pad;
        belief.title.text = `Score belief (5/25/50/75/95 percentiles)   best fit C=${C.toFixed(1)}`;

        const imgStyle = "width:100%;height:auto;border-radius:6px;";
        board.text = imgs[k] ? `<img src="${imgs[k]}" style="${imgStyle}">`
                             : "<i>no board image</i>";
        info.text = `position #${k} &nbsp;•&nbsp; generation epoch ${epochs[gj]}`;
        """,
    )
    pos.js_on_change("value", update)
    gen.js_on_change("value", update)

    # Prev/next arrows that step the position (client-side).
    arrow_css = InlineStyleSheet(
        css=".bk-btn { font-size: 34px; font-weight: 700; color: #2c7be5; }"
    )
    prev_btn = Button(label="❮", width=52, height=160, button_type="light", stylesheets=[arrow_css])
    next_btn = Button(label="❯", width=52, height=160, button_type="light", stylesheets=[arrow_css])
    prev_btn.js_on_click(
        CustomJS(
            args=dict(pos=pos), code="pos.value = String(Math.max(0, (parseInt(pos.value)|0) - 1));"
        )
    )
    next_btn.js_on_click(
        CustomJS(
            args=dict(pos=pos, N=n),
            code="pos.value = String(Math.min(N-1, (parseInt(pos.value)|0) + 1));",
        )
    )

    # Initial render (CustomJS only fires on change).
    mono.title.text = (
        f"Monotonicity — R²={curve_scores[gi, k0, 1]:.2f}  viol={int(curve_scores[gi, k0, 2])}"
    )
    belief.title.text = f"Score belief (5/25/50/75/95 percentiles)   best fit C={c0:.1f}"
    board.text = (
        f'<img src="{imgs[k0]}" style="width:100%;height:auto;border-radius:6px;">'
        if imgs[k0]
        else "<i>no board image</i>"
    )
    info.text = f"position #{k0} &nbsp;•&nbsp; generation epoch {epochs[gi]}"

    center_css = InlineStyleSheet(css=":host { align-items: center; gap: 10px; }")
    main = row(prev_btn, mono, belief, column(info, board), next_btn, stylesheets=[center_css])
    # `gen` stays referenced by the CustomJS (its value pins the generation) even when
    # not shown, so the figure renders the requested generation.
    cells = [_label("position"), pos]
    if not external_gen:
        cells += [_label("generation"), gen, latest]
    controls = row(*cells, stylesheets=[center_css])
    return View(layout=column(main, controls), gen=gen, latest=latest, pos=pos)


# ---------------------------------------------------------------------------
# Calibration view: reliability diagrams for one generation
# ---------------------------------------------------------------------------


def _aligned_series(conn, name: str, epochs: list[int]) -> list[float]:
    eps, vals = db.read_metric_series(conn, name)
    lut = {int(e): float(v) for e, v in zip(eps, vals, strict=True)}
    return [lut.get(e, float("nan")) for e in epochs]


def calibration_view(
    conn, init_gen: int | None = None, follow: bool = True, external_gen: bool = False
) -> View | None:
    epochs, cal = db.read_all_calibration(conn)
    if not epochs:
        return None

    rel_edges, sd_edges = cal["rel_edges"], cal["sd_edges"]
    rel_centers = 0.5 * (rel_edges[:-1] + rel_edges[1:])
    g = len(epochs)
    gi = (g - 1) if init_gen is None else max(0, min(init_gen, g - 1))
    K, M = cal["rel_pred"].shape[1], cal["sd_pred"].shape[1]
    count_max = float(max(cal["rel_count"].max(), 1)) * 1.1
    sd_lim = float(np.abs(sd_edges).max())

    def masked(pred, actual, count):
        m = count > 0
        return list(pred[m]), list(actual[m])

    wx, wy = masked(cal["rel_pred"][gi], cal["rel_actual"][gi], cal["rel_count"][gi])
    wld_src = ColumnDataSource(dict(x=wx, y=wy))
    count_src = ColumnDataSource(dict(x=list(rel_centers), top=list(cal["rel_count"][gi])))
    wld = figure(
        width=720,
        height=620,
        x_range=Range1d(0, 1),
        y_range=Range1d(0, 1),
        x_axis_label="predicted win rate",
        y_axis_label="empirical win rate",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        title="",
    )
    wld.extra_y_ranges = {"count": Range1d(start=0, end=count_max)}
    wld.add_layout(LinearAxis(y_range_name="count", axis_label="count"), "right")
    wld.vbar(
        x="x",
        top="top",
        width=0.9 / K,
        source=count_src,
        fill_color="#dddddd",
        line_color=None,
        y_range_name="count",
    )
    wld.line([0, 1], [0, 1], color="#bbbbbb", line_dash="dashed")
    wld.line("x", "y", source=wld_src, color="#1f77b4", line_width=2)
    wld.scatter("x", "y", source=wld_src, color="#1f77b4", size=7)

    sx, sy = masked(cal["sd_pred"][gi], cal["sd_actual"][gi], cal["sd_count"][gi])
    sd_src = ColumnDataSource(dict(x=sx, y=sy))
    sd = figure(
        width=500,
        height=440,
        x_range=Range1d(-sd_lim, sd_lim),
        y_range=Range1d(-sd_lim, sd_lim),
        x_axis_label="predicted mean score diff",
        y_axis_label="actual mean score diff",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        title="",
    )
    sd.line([-sd_lim, sd_lim], [-sd_lim, sd_lim], color="#bbbbbb", line_dash="dashed")
    sd.line("x", "y", source=sd_src, color="#08306b", line_width=2)
    sd.scatter("x", "y", source=sd_src, color="#08306b", size=7)

    brier = _aligned_series(conn, "calib_brier", epochs)
    ece = _aligned_series(conn, "calib_ece", epochs)
    mae = _aligned_series(conn, "calib_scorediff_mae", epochs)

    gen, latest = _gen_controls(g, gi, follow)
    update = CustomJS(
        args=dict(
            gen=gen,
            K=K,
            M=M,
            centers=list(rel_centers),
            epochs=epochs,
            rel_pred=cal["rel_pred"].astype(np.float32).ravel(),
            rel_actual=cal["rel_actual"].astype(np.float32).ravel(),
            rel_count=cal["rel_count"].astype(np.float32).ravel(),
            sd_pred=cal["sd_pred"].astype(np.float32).ravel(),
            sd_actual=cal["sd_actual"].astype(np.float32).ravel(),
            sd_count=cal["sd_count"].astype(np.float32).ravel(),
            brier=brier,
            ece=ece,
            mae=mae,
            wld_src=wld_src,
            count_src=count_src,
            sd_src=sd_src,
            wld=wld,
            sd=sd,
        ),
        code="""
        const gj = gen.value|0, rb = gj*K;
        const wx=[], wy=[], cx=[], ct=[];
        for (let i=0;i<K;i++){
            ct.push(rel_count[rb+i]); cx.push(centers[i]);
            if (rel_count[rb+i] > 0){ wx.push(rel_pred[rb+i]); wy.push(rel_actual[rb+i]); }
        }
        wld_src.data={x:wx, y:wy}; wld_src.change.emit();
        count_src.data={x:cx, top:ct}; count_src.change.emit();
        const sb = gj*M; const sx=[], sy=[];
        for (let i=0;i<M;i++){
            if (sd_count[sb+i] > 0){ sx.push(sd_pred[sb+i]); sy.push(sd_actual[sb+i]); }
        }
        sd_src.data={x:sx, y:sy}; sd_src.change.emit();
        wld.title.text = `WLD reliability — epoch ${epochs[gj]}  ` +
            `Brier=${brier[gj].toFixed(4)}  ECE=${ece[gj].toFixed(4)}`;
        sd.title.text = `Score-diff calibration — epoch ${epochs[gj]}  MAE=${mae[gj].toFixed(1)}`;
        """,
    )
    gen.js_on_change("value", update)
    wld.title.text = (
        f"WLD reliability — epoch {epochs[gi]}  Brier={brier[gi]:.4f}  ECE={ece[gi]:.4f}"
    )
    sd.title.text = f"Score-diff calibration — epoch {epochs[gi]}  MAE={mae[gi]:.1f}"

    # external_gen: drop the in-figure generation slider + latest (React drives the
    # generation). `gen` stays referenced by the CustomJS so the figure renders the
    # requested generation.
    center = InlineStyleSheet(css=":host { align-items: center; }")
    rows = [row(wld, sd)]
    if not external_gen:
        rows.append(row(gen, latest, stylesheets=[center]))
    return View(layout=column(*rows, sizing_mode="stretch_width"), gen=gen, latest=latest)

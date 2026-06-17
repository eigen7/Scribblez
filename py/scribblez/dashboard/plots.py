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
    Button, Checkbox, ColumnDataSource, CustomJS, Div, HoverTool, InlineStyleSheet,
    LinearAxis, Range1d, Select, Slider,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure

from . import db

SERIES_SIZE = 400  # square-ish learning-curve figures


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


def _series_figure(conn, title: str, names: list[str]):
    fig = figure(width=SERIES_SIZE, height=SERIES_SIZE, title=title, x_axis_label="epoch",
                 tools="pan,box_zoom,wheel_zoom,reset,save")
    fig.add_tools(HoverTool(tooltips=[("epoch", "@x"), ("value", "@y{0.0000}")], mode="vline"))
    palette = Category10[10]
    plotted = 0
    for i, name in enumerate(names):
        epochs, values = db.read_metric_series(conn, name)
        if len(epochs) == 0:
            continue
        src = ColumnDataSource(dict(x=epochs, y=values))
        color = palette[i % len(palette)]
        fig.line("x", "y", source=src, color=color, line_width=2, legend_label=name)
        fig.scatter("x", "y", source=src, color=color, size=4)
        plotted += 1
    if plotted == 0:
        return None
    fig.legend.label_text_font_size = "8pt"
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    return fig


def series_grid(conn, groups: list[tuple[str, list[str]]], ncols: int = 2):
    """A grid of square learning-curve figures for the given (title, metric-names) groups."""
    figs = [f for title, names in groups if (f := _series_figure(conn, title, names))]
    if not figs:
        return Div(text="<i>No scalar metrics recorded yet.</i>")
    rows = [row(*figs[i:i + ncols]) for i in range(0, len(figs), ncols)]
    return column(*rows)


# ---------------------------------------------------------------------------
# Generation slider + "latest" follow checkbox
# ---------------------------------------------------------------------------


def _gen_controls(num_gens: int, epochs: list[int], init_gen: int, follow: bool):
    """A generation slider + 'latest' checkbox; the checkbox locks/follows client-side."""
    gen = Slider(start=0, end=max(num_gens - 1, 1), value=init_gen, step=1,
                 title="generation", disabled=(follow or num_gens < 2), width=420)
    latest = Checkbox(label="latest", active=follow)
    latest.js_on_change("active", CustomJS(args=dict(gen=gen), code="""
        gen.disabled = this.active;
        if (this.active) { gen.value = gen.end; }
    """))
    return gen, latest


# ---------------------------------------------------------------------------
# Probes view: board + monotonicity + score belief for one (position, gen)
# ---------------------------------------------------------------------------


def _board_uri(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def probes_view(conn, image_dir: Path, init_pos: int = 0, init_gen: int | None = None,
                follow: bool = True) -> View | None:
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

    board = Div(text="", width=450)
    info = Div(styles={"font-size": "12px", "color": "#445", "font-weight": "600"})

    mono_src = ColumnDataSource(dict(x=list(diffs), y=list(win_rate[gi, k0])))
    mono = figure(width=430, height=430, title="", x_range=Range1d(diffs[0], diffs[-1]),
                  y_range=Range1d(-0.02, 1.02), x_axis_label="score differential",
                  y_axis_label="win rate", tools="pan,box_zoom,wheel_zoom,reset,save")
    mono.line([diffs[0], diffs[-1]], [0.5, 0.5], color="#cccccc", line_dash="dashed")
    mono.line([0, 0], [0, 1], color="#cccccc", line_dash="dashed")
    mono.line("x", "y", source=mono_src, color="#1f77b4", line_width=2)

    b0 = bands[gi, k0]
    outer = ColumnDataSource(dict(x=list(diffs), y1=list(b0[:, 0]), y2=list(b0[:, q - 1])))
    inner = ColumnDataSource(dict(x=list(diffs), y1=list(b0[:, 1]), y2=list(b0[:, q - 2])))
    med = ColumnDataSource(dict(x=list(diffs), y=list(b0[:, q // 2])))
    belief_yr = Range1d(float(b0.min()), float(b0.max()))
    belief = figure(width=430, height=430, title="", x_range=Range1d(diffs[0], diffs[-1]),
                    y_range=belief_yr, x_axis_label="input score differential",
                    y_axis_label="predicted final score differential",
                    tools="pan,box_zoom,wheel_zoom,reset,save")
    belief.varea("x", "y1", "y2", source=outer, fill_color="#1f77b4", fill_alpha=0.18)
    belief.varea("x", "y1", "y2", source=inner, fill_color="#1f77b4", fill_alpha=0.36)
    belief.line("x", "y", source=med, color="#08306b", line_width=1.5)
    belief.line(diffs, diffs, color="#bbbbbb", line_dash="dashed")

    gen, latest = _gen_controls(g, epochs, gi, follow)
    pos = Select(title="position", options=[str(k) for k in range(n)], value=str(k0), width=110)

    update = CustomJS(
        args=dict(
            pos=pos, gen=gen, N=n, R=r, Q=q, xs=list(diffs), epochs=epochs, imgs=imgs,
            winrate=win_rate.astype(np.float32).ravel(),
            cscores=curve_scores.astype(np.float32).ravel(),
            bands=bands.astype(np.float32).ravel(),
            mono_src=mono_src, outer=outer, inner=inner, med=med,
            mono=mono, belief=belief, belief_yr=belief_yr, board=board, info=info,
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
        let ymin=1e18, ymax=-1e18;
        for (let s=0;s<R;s++){
            const o = bb + s*Q;
            const a=bands[o], b=bands[o+Q-1], m=bands[o+(Q>>1)], c=bands[o+1], d=bands[o+Q-2];
            lo.push(a); hi.push(b); i1.push(c); i3.push(d); me.push(m);
            if (a<ymin) ymin=a; if (b>ymax) ymax=b;
        }
        outer.data={x:xs, y1:lo, y2:hi}; outer.change.emit();
        inner.data={x:xs, y1:i1, y2:i3}; inner.change.emit();
        med.data={x:xs, y:me}; med.change.emit();
        const pad=(ymax-ymin)*0.05+1; belief_yr.start=ymin-pad; belief_yr.end=ymax+pad;
        belief.title.text = "Score belief (5/25/50/75/95 percentiles)";

        board.text = imgs[k] ? `<img src="${imgs[k]}" style="width:100%;height:auto;border-radius:6px;">`
                             : "<i>no board image</i>";
        info.text = `position #${k} &nbsp;•&nbsp; generation epoch ${epochs[gj]}`;
        """,
    )
    pos.js_on_change("value", update)
    gen.js_on_change("value", update)

    # Prev/next arrows that step the position (client-side).
    arrow_css = InlineStyleSheet(css="""
        .bk-btn { font-size: 28px; font-weight: 700; color: #2c7be5; }
    """)
    prev_btn = Button(label="❮", width=46, height=120, button_type="light", stylesheets=[arrow_css])
    next_btn = Button(label="❯", width=46, height=120, button_type="light", stylesheets=[arrow_css])
    prev_btn.js_on_click(CustomJS(args=dict(pos=pos), code=
        "pos.value = String(Math.max(0, (parseInt(pos.value)|0) - 1));"))
    next_btn.js_on_click(CustomJS(args=dict(pos=pos, N=n), code=
        "pos.value = String(Math.min(N-1, (parseInt(pos.value)|0) + 1));"))

    # Initial render (CustomJS only fires on change).
    mono.title.text = f"Monotonicity — R²={curve_scores[gi,k0,1]:.2f}  viol={int(curve_scores[gi,k0,2])}"
    belief.title.text = "Score belief (5/25/50/75/95 percentiles)"
    board.text = (f'<img src="{imgs[k0]}" style="width:100%;height:auto;border-radius:6px;">'
                  if imgs[k0] else "<i>no board image</i>")
    info.text = f"position #{k0} &nbsp;•&nbsp; generation epoch {epochs[gi]}"

    center_css = InlineStyleSheet(css=":host { align-items: center; gap: 8px; }")
    main = row(prev_btn, mono, belief, column(info, board), next_btn, stylesheets=[center_css])
    controls = row(pos, gen, latest, stylesheets=[center_css])
    return View(layout=column(main, controls), gen=gen, latest=latest, pos=pos)


# ---------------------------------------------------------------------------
# Calibration view: reliability diagrams for one generation
# ---------------------------------------------------------------------------


def _aligned_series(conn, name: str, epochs: list[int]) -> list[float]:
    eps, vals = db.read_metric_series(conn, name)
    lut = {int(e): float(v) for e, v in zip(eps, vals)}
    return [lut.get(e, float("nan")) for e in epochs]


def calibration_view(conn, init_gen: int | None = None, follow: bool = True) -> View | None:
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
    wld = figure(width=500, height=440, x_range=Range1d(0, 1), y_range=Range1d(0, 1),
                 x_axis_label="predicted win rate", y_axis_label="empirical win rate",
                 tools="pan,box_zoom,wheel_zoom,reset,save", title="")
    wld.extra_y_ranges = {"count": Range1d(start=0, end=count_max)}
    wld.add_layout(LinearAxis(y_range_name="count", axis_label="count"), "right")
    wld.vbar(x="x", top="top", width=0.9 / K, source=count_src, fill_color="#dddddd",
             line_color=None, y_range_name="count")
    wld.line([0, 1], [0, 1], color="#bbbbbb", line_dash="dashed")
    wld.line("x", "y", source=wld_src, color="#1f77b4", line_width=2)
    wld.scatter("x", "y", source=wld_src, color="#1f77b4", size=7)

    sx, sy = masked(cal["sd_pred"][gi], cal["sd_actual"][gi], cal["sd_count"][gi])
    sd_src = ColumnDataSource(dict(x=sx, y=sy))
    sd = figure(width=500, height=440, x_range=Range1d(-sd_lim, sd_lim), y_range=Range1d(-sd_lim, sd_lim),
                x_axis_label="predicted mean score diff", y_axis_label="actual mean score diff",
                tools="pan,box_zoom,wheel_zoom,reset,save", title="")
    sd.line([-sd_lim, sd_lim], [-sd_lim, sd_lim], color="#bbbbbb", line_dash="dashed")
    sd.line("x", "y", source=sd_src, color="#08306b", line_width=2)
    sd.scatter("x", "y", source=sd_src, color="#08306b", size=7)

    brier = _aligned_series(conn, "calib_brier", epochs)
    ece = _aligned_series(conn, "calib_ece", epochs)
    mae = _aligned_series(conn, "calib_scorediff_mae", epochs)

    gen, latest = _gen_controls(g, epochs, gi, follow)
    update = CustomJS(
        args=dict(
            gen=gen, K=K, M=M, centers=list(rel_centers), epochs=epochs,
            rel_pred=cal["rel_pred"].astype(np.float32).ravel(),
            rel_actual=cal["rel_actual"].astype(np.float32).ravel(),
            rel_count=cal["rel_count"].astype(np.float32).ravel(),
            sd_pred=cal["sd_pred"].astype(np.float32).ravel(),
            sd_actual=cal["sd_actual"].astype(np.float32).ravel(),
            sd_count=cal["sd_count"].astype(np.float32).ravel(),
            brier=brier, ece=ece, mae=mae,
            wld_src=wld_src, count_src=count_src, sd_src=sd_src, wld=wld, sd=sd,
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
        for (let i=0;i<M;i++){ if (sd_count[sb+i] > 0){ sx.push(sd_pred[sb+i]); sy.push(sd_actual[sb+i]); } }
        sd_src.data={x:sx, y:sy}; sd_src.change.emit();
        wld.title.text = `WLD reliability — epoch ${epochs[gj]}  Brier=${brier[gj].toFixed(4)}  ECE=${ece[gj].toFixed(4)}`;
        sd.title.text = `Score-diff calibration — epoch ${epochs[gj]}  MAE=${mae[gj].toFixed(1)}`;
        """,
    )
    gen.js_on_change("value", update)
    wld.title.text = f"WLD reliability — epoch {epochs[gi]}  Brier={brier[gi]:.4f}  ECE={ece[gi]:.4f}"
    sd.title.text = f"Score-diff calibration — epoch {epochs[gi]}  MAE={mae[gi]:.1f}"

    center = InlineStyleSheet(css=":host { align-items: center; }")
    return View(layout=column(row(wld, sd), row(gen, latest, stylesheets=[center]),
                              sizing_mode="stretch_width"),
                gen=gen, latest=latest)

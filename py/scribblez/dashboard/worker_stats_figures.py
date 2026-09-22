"""Bokeh figures for the generic worker Stats tab.

Built from the per-worker stats records (stats/<worker_id>.json under the
tag's root; see scribblez/workloads/worker.py) and shaped by the role's
StatsSpec (unit noun + timing phases), so any workload role that publishes
stats gets the same summary tiles/table and figures. Each builder takes the
parsed records plus the StatsSpec and returns a Bokeh model or None when
there is nothing to plot; the API serializes with json_item for the React
BokehFigure embed.
"""

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.palettes import Blues, Category10
from bokeh.plotting import figure

from scribblez.workloads.base import StatsSpec

# Worker summary rates and phase means average over this many trailing
# cycles, so they track current behavior rather than the whole run.
WINDOW = 20


def read_stats(stats_dir: Path) -> list[dict]:
    """All worker stats records under the tag, newest-updated first."""
    records = []
    for path in sorted(stats_dir.glob("*.json")) if stats_dir.is_dir() else []:
        records.append(json.loads(path.read_text()))
    return sorted(records, key=lambda r: -r.get("updated_at", 0))


def _recent(record: dict) -> list[dict]:
    return record.get("recent", [])[-WINDOW:]


def worker_summary(record: dict, stats: StatsSpec) -> dict:
    """The per-worker roll-up the Stats tab tabulates: recent-window rates and
    cycle-phase means, plus the cumulative counters."""
    recent = _recent(record)
    span = recent[-1]["t"] - recent[0]["t"] if len(recent) > 1 else 0.0
    units_recent = sum(s["units"] for s in recent[1:])  # rate over the span between samples
    upload_bytes = sum(s["bytes"] for s in recent)
    upload_s = sum(s.get("upload_s", 0.0) for s in recent)

    def mean(key: str) -> float:
        return sum(s.get(key, 0.0) for s in recent) / len(recent) if recent else 0.0

    return {
        "worker_id": record["worker_id"],
        "role": record.get("role"),
        "kind": record["kind"],
        "threads": record.get("threads"),
        "bundle_id": record.get("bundle_id"),
        "host_arch": record.get("host_arch"),
        "bundle_arch": record.get("bundle_arch"),
        "units_total": record["units_total"],
        "cycles_total": record["cycles_total"],
        "updated_at": record["updated_at"],
        "units_per_hour": units_recent / span * 3600 if span > 0 else None,
        "phases": {p: mean(p) for p in stats.phases},
        "upload_mbps": (upload_bytes / 1e6) / upload_s if upload_s > 0 else None,
    }


def _rate_points(samples: list[dict]) -> tuple[list[float], list[float]]:
    """Units/hour between consecutive samples, stamped at each interval's end."""
    steps = [(a, b) for a, b in pairwise(samples) if b["t"] > a["t"]]
    xs = [b["t"] for _, b in steps]
    ys = [(b["units_total"] - a["units_total"]) / (b["t"] - a["t"]) * 3600 for a, b in steps]
    return xs, ys


def _time_range(ts: list[float]) -> Range1d:
    """An x range spanning every timestamp in `ts`, with a margin. Set
    explicitly because Bokeh's auto range around a lone point collapses to
    zero width, and the datetime axis then labels it in microseconds."""
    lo, hi = min(ts), max(ts)
    pad = max((hi - lo) * 0.03, 60.0)
    return Range1d(_datetime(lo - pad), _datetime(hi + pad))


def _datetime(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=UTC)


def _time_figure(title: str, y_label: str, ts: list[float]):
    fig = figure(
        title=title,
        x_axis_type="datetime",
        x_range=_time_range(ts),
        height=280,
        sizing_mode="stretch_width",
    )
    fig.yaxis.axis_label = y_label
    fig.y_range.start = 0
    return fig


def _worker_series(fig, index: int, worker_id: str, xs: list[float], ys: list, step: bool):
    """One worker's line (a step for counts, straight for rates) plus a marker
    per point, so a series still shows when it has a single point. Drawn
    even with no points, so every worker has its legend entry."""
    color = Category10[10][index % 10]
    xs = [_datetime(t) for t in xs]
    if step:
        fig.step(xs, ys, mode="after", color=color, legend_label=worker_id, line_width=2)
    else:
        fig.line(xs, ys, color=color, legend_label=worker_id, line_width=2)
    fig.scatter(xs, ys, color=color, legend_label=worker_id, size=6)


def rate(records: list[dict], stats: StatsSpec):
    """Units/hour per worker over its recent samples: the slope of the
    cumulative timeline, plotted directly, so throughput dips and stalls are
    visible. A worker that stops reporting reads as a line that simply ends;
    one with a single sample so far has a legend entry and no points."""
    series = [(r["worker_id"], _rate_points(r.get("recent", []))) for r in records]
    if not any(xs for _, (xs, _) in series):
        return None
    ts = [s["t"] for r in records for s in r.get("recent", [])]
    fig = _time_figure(
        f"{stats.unit.capitalize()} per hour (recent window)", f"{stats.unit} / hour", ts
    )
    for i, (worker_id, (xs, ys)) in enumerate(series):
        _worker_series(fig, i, worker_id, xs, ys, step=False)
    fig.legend.location = "top_left"
    return fig


def _worker_history(record: dict) -> list[list]:
    """A worker's cumulative count over its whole run: [t, units_total] points
    from the slot's first start (at zero) through every cycle since. The
    recent window is merged in: once the history has been thinned, the window
    holds the finer detail of the latest cycles."""
    points = {(t, n) for t, n in record.get("history", [])}
    points |= {(s["t"], s["units_total"]) for s in record.get("recent", [])}
    return [[record["started_at"], 0], *(list(p) for p in sorted(points))]


def _fleet_total(histories: list[list[list]]) -> tuple[list[float], list[int]]:
    """The fleet's cumulative count: at each point of any worker's history,
    the sum of every worker's latest total at or before it."""
    events = sorted((t, w, n) for w, h in enumerate(histories) for t, n in h)
    latest = [0] * len(histories)
    xs, ys = [], []
    for t, w, n in events:
        latest[w] = n
        xs.append(t)
        ys.append(sum(latest))
    return xs, ys


def cumulative(records: list[dict], stats: StatsSpec):
    """Cumulative units per worker over the whole run, with the fleet total
    on top: what the run has delivered and how steadily, which a rate of
    lumpy per-cycle counts (a survey finds 0-6 positions a game) obscures."""
    if not records:
        return None
    histories = [_worker_history(r) for r in records]
    fleet_xs, fleet_ys = _fleet_total(histories)
    fig = _time_figure(f"{stats.unit.capitalize()} over time", f"cumulative {stats.unit}", fleet_xs)
    for i, (record, history) in enumerate(zip(records, histories, strict=True)):
        xs, ys = [t for t, _ in history], [n for _, n in history]
        _worker_series(fig, i, record["worker_id"], xs, ys, step=True)
    fig.step(
        [_datetime(t) for t in fleet_xs], fleet_ys, mode="after",
        color="#333333", line_dash="dashed", line_width=2.5, legend_label="fleet",
    )  # fmt: skip
    fig.legend.location = "top_left"
    return fig


def _phase_colors(n: int) -> list[str]:
    """A dark-to-light single-hue ramp: phases are ordered stages of one
    cycle, not independent series. Drawn from the (n+1)-step palette so the
    near-white lightest step is never used."""
    return list(Blues[max(3, n + 1)])[:n]


def cycle_breakdown(records: list[dict], stats: StatsSpec):
    """Mean seconds per cycle phase, stacked horizontally with one row per
    worker: where each worker's wall time goes, on a common seconds scale
    that stays readable as the fleet grows. A dominant upload share means
    the worker is network-bound rather than CPU-bound."""
    rows = [worker_summary(r, stats) for r in records if _recent(r)]
    if not rows:
        return None
    phases = list(stats.phases)
    workers = [r["worker_id"] for r in rows]
    source = ColumnDataSource(
        {
            "worker": workers,
            **{p: [r["phases"][p] for r in rows] for p in phases},
        }
    )
    fig = figure(
        title="Cycle time by phase",
        y_range=list(reversed(workers)),  # records order, top to bottom
        height=110 + 34 * len(workers),
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    fig.xaxis.axis_label = "seconds / cycle"
    fig.x_range.start = 0
    renderers = fig.hbar_stack(
        phases,
        y="worker",
        height=0.55,
        source=source,
        color=_phase_colors(len(phases)),
        legend_label=[stats.phases[p] for p in phases],
    )
    fig.add_tools(
        HoverTool(renderers=renderers, tooltips=[("worker", "@worker")]
                  + [(stats.phases[p], f"@{p}{{0.0}} s") for p in phases])
    )  # fmt: skip
    fig.legend.location = "top_right"
    return fig


FIGURES = {
    "cumulative": cumulative,
    "rate": rate,
    "cycle_breakdown": cycle_breakdown,
}

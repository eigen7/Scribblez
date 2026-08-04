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

from bokeh.models import ColumnDataSource, HoverTool
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
        "host_arch": record.get("host_arch"),
        "bundle_arch": record.get("bundle_arch"),
        "units_total": record["units_total"],
        "cycles_total": record["cycles_total"],
        "updated_at": record["updated_at"],
        "units_per_hour": units_recent / span * 3600 if span > 0 else None,
        "phases": {p: mean(p) for p in stats.phases},
        "upload_mbps": (upload_bytes / 1e6) / upload_s if upload_s > 0 else None,
    }


def _rate_points(samples: list[dict]) -> tuple[list[datetime], list[float]]:
    """Units/hour between consecutive samples, stamped at each interval's end."""
    steps = [(a, b) for a, b in pairwise(samples) if b["t"] > a["t"]]
    xs = [datetime.fromtimestamp(b["t"], tz=UTC) for _, b in steps]
    ys = [(b["units_total"] - a["units_total"]) / (b["t"] - a["t"]) * 3600 for a, b in steps]
    return xs, ys


def rate(records: list[dict], stats: StatsSpec):
    """Units/hour per worker over its recent samples: the slope the old
    cumulative timeline made the reader eyeball, plotted directly, so
    throughput dips and stalls are visible. A worker that stops reporting
    reads as a line that simply ends."""
    fig = figure(
        title=f"{stats.unit.capitalize()} per hour (recent window)",
        x_axis_type="datetime",
        height=280,
        sizing_mode="stretch_width",
    )
    fig.yaxis.axis_label = f"{stats.unit} / hour"
    fig.y_range.start = 0
    palette = Category10[10]
    plotted = 0
    for i, record in enumerate(records):
        xs, ys = _rate_points(record.get("recent", []))
        if not xs:
            continue
        fig.line(xs, ys, color=palette[i % 10], legend_label=record["worker_id"], line_width=2)
        plotted += 1
    if plotted == 0:
        return None
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
    "rate": rate,
    "cycle_breakdown": cycle_breakdown,
}

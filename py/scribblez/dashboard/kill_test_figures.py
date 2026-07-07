"""Bokeh figures for the kill-test Stats tab.

Built from the per-worker stats records (stats/<worker_id>.json under the
tag's data dir; see the WorkerStats section of py/cloud/worker_entrypoint.py).
Each builder takes the parsed records and returns a Bokeh model or None when
there is nothing to plot; the API serializes with json_item for the React
BokehFigure embed, mirroring the training dashboard's figure path.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10
from bokeh.plotting import figure

# Worker throughput/breakdown figures average over this many trailing cycles,
# so they track current behavior rather than the whole run.
WINDOW = 20


def read_stats(data_dir: Path) -> list[dict]:
    """All worker stats records under the tag, newest-updated first."""
    stats_dir = data_dir / "stats"
    records = []
    for path in sorted(stats_dir.glob("*.json")) if stats_dir.is_dir() else []:
        records.append(json.loads(path.read_text()))
    return sorted(records, key=lambda r: -r.get("updated_at", 0))


def _recent(record: dict) -> list[dict]:
    return record.get("recent", [])[-WINDOW:]


def worker_summary(record: dict) -> dict:
    """The per-worker roll-up the Stats tab tabulates: recent-window rates and
    cycle-phase means, plus the cumulative counters."""
    recent = _recent(record)
    span = recent[-1]["t"] - recent[0]["t"] if len(recent) > 1 else 0.0
    pairs_recent = sum(s["pairs"] for s in recent[1:])  # rate over the span between samples
    upload_bytes = sum(s["bytes"] for s in recent)
    upload_s = sum(s["upload_s"] for s in recent)

    def mean(key: str) -> float:
        return sum(s[key] for s in recent) / len(recent) if recent else 0.0

    return {
        "worker_id": record["worker_id"],
        "kind": record["kind"],
        "threads": record.get("threads"),
        "host_arch": record.get("host_arch"),
        "bundle_arch": record.get("bundle_arch"),
        "pairs_total": record["pairs_total"],
        "cycles_total": record["cycles_total"],
        "updated_at": record["updated_at"],
        "pairs_per_hour": pairs_recent / span * 3600 if span > 0 else None,
        "gen_s": mean("gen_s"),
        "sim_s": mean("sim_s"),
        "upload_s": mean("upload_s"),
        "upload_mbps": (upload_bytes / 1e6) / upload_s if upload_s > 0 else None,
    }


def _workers_figure(title: str, workers: list[str], y_label: str):
    fig = figure(
        title=title,
        x_range=workers,
        height=300,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    fig.yaxis.axis_label = y_label
    fig.xaxis.major_label_orientation = 0.6
    fig.y_range.start = 0
    return fig


def throughput(records: list[dict]):
    """Pairs/hour per worker over its recent cycles."""
    rows = [worker_summary(r) for r in records]
    rows = [r for r in rows if r["pairs_per_hour"] is not None]
    if not rows:
        return None
    workers = [r["worker_id"] for r in rows]
    fig = _workers_figure("Recent throughput", workers, "pairs / hour")
    fig.vbar(x=workers, top=[r["pairs_per_hour"] for r in rows], width=0.7, color="#1f77b4")
    return fig


def cycle_breakdown(records: list[dict]):
    """Mean seconds per cycle phase, stacked per worker: where each worker's
    wall time goes (generate vs sim vs upload). A dominant upload share means
    the worker is network-bound rather than CPU-bound."""
    rows = [worker_summary(r) for r in records if _recent(r)]
    if not rows:
        return None
    phases = ["gen_s", "sim_s", "upload_s"]
    labels = {"gen_s": "self-play", "sim_s": "sim", "upload_s": "upload"}
    source = ColumnDataSource(
        {
            "worker": [r["worker_id"] for r in rows],
            **{p: [r[p] for r in rows] for p in phases},
        }
    )
    fig = _workers_figure("Cycle time breakdown", [r["worker_id"] for r in rows], "seconds / cycle")
    renderers = fig.vbar_stack(
        phases,
        x="worker",
        width=0.7,
        source=source,
        color=Category10[3],
        legend_label=[labels[p] for p in phases],
    )
    fig.add_tools(
        HoverTool(renderers=renderers, tooltips=[("worker", "@worker")]
                  + [(labels[p], f"@{p}{{0.0}} s") for p in phases])
    )  # fmt: skip
    fig.legend.location = "top_left"
    return fig


def pairs_timeline(records: list[dict]):
    """Cumulative pairs produced per worker, from each record's recent-sample
    window (older history ages out of the window)."""
    lines = [(r["worker_id"], r.get("recent", [])) for r in records]
    lines = [(w, s) for w, s in lines if len(s) > 1]
    if not lines:
        return None
    fig = figure(
        title="Pairs over time (recent window)",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    fig.yaxis.axis_label = "cumulative pairs"
    palette = Category10[10]
    for i, (worker, samples) in enumerate(lines):
        xs = [datetime.fromtimestamp(s["t"], tz=UTC) for s in samples]
        ys = [s["pairs_total"] for s in samples]
        fig.line(xs, ys, color=palette[i % 10], legend_label=worker, line_width=2)
    fig.legend.location = "top_left"
    return fig


FIGURES = {
    "throughput": throughput,
    "cycle_breakdown": cycle_breakdown,
    "pairs_timeline": pairs_timeline,
}

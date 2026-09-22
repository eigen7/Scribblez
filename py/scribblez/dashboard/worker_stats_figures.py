"""The generic worker Stats tab's data: per-worker summaries and the
cumulative figure.

Built from the per-worker stats records (stats/<worker_id>.json under the
tag's root; see scribblez/workloads/worker.py) and shaped by the role's
StatsSpec (unit noun + timing phases), so any workload role that publishes
stats gets the same summary tiles/table and figure. The figure builder
returns a Bokeh model, which the API serializes with json_item for the React
BokehFigure embed.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from bokeh.models import Range1d
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


# The figure's worker selector value that plots the fleet total.
FLEET = "fleet"


def _datetime(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=UTC)


def _time_range(ts: list[float]) -> Range1d:
    """An x range spanning every timestamp in `ts`, with a margin. Set
    explicitly because Bokeh's auto range around a lone point collapses to
    zero width, and the datetime axis then labels it in microseconds."""
    lo, hi = min(ts), max(ts)
    pad = max((hi - lo) * 0.03, 60.0)
    return Range1d(_datetime(lo - pad), _datetime(hi + pad))


def _worker_history(record: dict) -> list[list]:
    """A worker's cumulative count over its whole run: [t, units_total] points
    from the slot's first start (at zero) through every cycle since. The
    recent window is merged in: once the history has been thinned, the window
    holds the finer detail of the latest cycles."""
    points = {(t, n) for t, n in record.get("history", [])}
    points |= {(s["t"], s["units_total"]) for s in record.get("recent", [])}
    return [[record["started_at"], 0], *(list(p) for p in sorted(points))]


def _fleet_total(histories: list[list[list]]) -> list[list]:
    """The fleet's cumulative count: at each point of any worker's history,
    the sum of every worker's latest total at or before it."""
    events = sorted((t, w, n) for w, h in enumerate(histories) for t, n in h)
    latest = [0] * len(histories)
    points = []
    for t, w, n in events:
        latest[w] = n
        points.append([t, sum(latest)])
    return points


def _series(records: list[dict], worker: str) -> list[list]:
    """The [t, units_total] points to plot: one worker's history, or the
    fleet total over every record's."""
    if worker == FLEET:
        return _fleet_total([_worker_history(r) for r in records])
    (record,) = (r for r in records if r["worker_id"] == worker)
    return _worker_history(record)


def cumulative(records: list[dict], stats: StatsSpec, worker: str):
    """Cumulative units over the whole run, for one worker or the fleet: what
    the run has delivered and how steadily, which a rate of lumpy per-cycle
    counts (a survey finds 0-6 positions a game) obscures. A step line with a
    marker per cycle, so a run of one cycle still shows."""
    points = _series(records, worker)
    if not points:
        return None
    xs = [_datetime(t) for t, _ in points]
    ys = [n for _, n in points]
    fig = figure(
        title=f"{stats.unit.capitalize()} over time: {worker}",
        x_axis_type="datetime",
        x_range=_time_range([t for t, _ in points]),
        height=280,
        sizing_mode="stretch_width",
    )
    fig.yaxis.axis_label = f"cumulative {stats.unit}"
    fig.y_range.start = 0
    fig.step(xs, ys, mode="after", line_width=2)
    fig.scatter(xs, ys, size=6)
    return fig

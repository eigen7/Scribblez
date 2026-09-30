"""Data for the generic Stats tab: per-worker summaries and the cumulative
figure.

Built from the per-worker stats records (stats/<worker_id>.json under the tag
root, written by scribblez/workloads/worker.py) and shaped by the role's
StatsSpec, so every role that publishes stats gets the same tab. The figure is
a Bokeh model that master_api.py serializes as a json_item.
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

# A worker is stale once its last sample is older than this many cycle times,
# and never sooner than STALE_FLOOR_S: a slow-cycling worker is not dead just
# because a minute passed.
STALE_CYCLES = 5
STALE_FLOOR_S = 120.0


def read_stats(stats_dir: Path) -> list[dict]:
    """All worker stats records under the tag, newest-updated first."""
    records = []
    for path in sorted(stats_dir.glob("*.json")) if stats_dir.is_dir() else []:
        records.append(json.loads(path.read_text()))
    return sorted(records, key=lambda r: -r.get("updated_at", 0))


def _recent(record: dict) -> list[dict]:
    return record.get("recent", [])[-WINDOW:]


def _mean(recent: list[dict], key: str) -> float:
    return sum(s.get(key, 0.0) for s in recent) / len(recent) if recent else 0.0


def worker_summary(record: dict, stats: StatsSpec, now: float) -> dict:
    """One worker's row in the Stats tab: recent-window rate, cycle time and
    phase means, plus its cumulative counters.

    The rate and cycle time come from the samples' wall-clock timestamps, so
    they count everything a cycle waits on, including time no phase tracks
    (reported as `other_s`), and never count a background phase twice."""
    recent = _recent(record)
    span = recent[-1]["t"] - recent[0]["t"] if len(recent) > 1 else 0.0
    # The first sample's units predate the span, so they are left out.
    units_recent = sum(s["units"] for s in recent[1:])
    upload_bytes = sum(s["bytes"] for s in recent)
    upload_s = sum(s.get("upload_s", 0.0) for s in recent)
    phases = {p: _mean(recent, p) for p in stats.phases}
    cycle_s = span / (len(recent) - 1) if span > 0 else None
    foreground = sum(v for p, v in phases.items() if p not in stats.background)
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
        "stale": now - record["updated_at"] > max(STALE_FLOOR_S, STALE_CYCLES * (cycle_s or 0.0)),
        "units_per_hour": units_recent / span * 3600 if span > 0 else None,
        "cycle_s": cycle_s,
        "phases": phases,
        "other_s": max(0.0, cycle_s - foreground) if cycle_s is not None else None,
        "upload_mbps": (upload_bytes / 1e6) / upload_s if upload_s > 0 else None,
    }


def pace(records: list[dict], role: str, stats: StatsSpec, now: float) -> float | None:
    """The fleet rate of `role`'s live workers, in units per hour, or None when
    none has a rate: the number that compares one tag's speed with another's."""
    rows = [worker_summary(r, stats, now) for r in records if r.get("role") == role]
    rates = [w["units_per_hour"] for w in rows if not w["stale"] and w["units_per_hour"]]
    return sum(rates) if rates else None


# The figure's worker selector value that plots the fleet total.
FLEET = "fleet"


def _datetime(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=UTC)


def _time_range(ts: list[float]) -> Range1d:
    """An x range spanning `ts` with a margin. Explicit because Bokeh's auto
    range around a lone point has zero width, which the datetime axis then
    labels in microseconds."""
    lo, hi = min(ts), max(ts)
    pad = max((hi - lo) * 0.03, 60.0)
    return Range1d(_datetime(lo - pad), _datetime(hi + pad))


def _worker_history(record: dict) -> list[list]:
    """A worker's cumulative count as [t, units_total] points, starting at zero
    at its first start. The recent window is merged in because the stored
    history is thinned and the window keeps the latest cycles at full
    detail."""
    points = {(t, n) for t, n in record.get("history", [])}
    points |= {(s["t"], s["units_total"]) for s in record.get("recent", [])}
    return [[record["started_at"], 0], *(list(p) for p in sorted(points))]


def _fleet_total(histories: list[list[list]]) -> list[list]:
    """The fleet's cumulative count: at each point of any worker's history, the
    sum of every worker's latest total at or before it."""
    events = sorted((t, w, n) for w, h in enumerate(histories) for t, n in h)
    latest = [0] * len(histories)
    points = []
    for t, w, n in events:
        latest[w] = n
        points.append([t, sum(latest)])
    return points


def _series(records: list[dict], worker: str) -> list[list]:
    """The points to plot for one worker, or for the fleet."""
    if worker == FLEET:
        return _fleet_total([_worker_history(r) for r in records])
    (record,) = (r for r in records if r["worker_id"] == worker)
    return _worker_history(record)


def cumulative(records: list[dict], stats: StatsSpec, worker: str):
    """Cumulative units over the whole run, for one worker or the fleet. Shows
    what the run has delivered and how steadily, which a rate obscures when
    per-cycle counts are lumpy (a survey finds 0-6 positions a game). Each
    cycle gets a marker, so a run of one cycle still shows."""
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

"""Tests for the Stats tab figures (scribblez.dashboard.worker_stats_figures)."""

from bokeh.models import Range1d
from scribblez.dashboard import worker_stats_figures as figs
from scribblez.workloads.base import StatsSpec

STATS = StatsSpec(unit="positions", phases={"sim_s": "survey"})
T0 = 1_790_000_000.0


def _record(worker_id: str, samples: list[tuple[float, int]]) -> dict:
    """A stats record whose cycles ended at the given (t, units_total) points."""
    return {
        "worker_id": worker_id,
        "started_at": T0,
        "updated_at": samples[-1][0] if samples else T0,
        "units_total": samples[-1][1] if samples else 0,
        "recent": [{"t": t, "units": 0, "units_total": n, "bytes": 0} for t, n in samples],
        "history": [[t, n] for t, n in samples],
    }


def _legend_labels(fig) -> list[str]:
    return [item.label["value"] for item in fig.legend[0].items]


def test_rate_spans_the_sample_window_and_lists_every_worker():
    """One rate point (the second sample of a worker) used to leave Bokeh a
    zero-width datetime range, labelled in microseconds; and a worker with a
    single sample had no legend entry at all."""
    records = [_record("ssh-0", [(T0 + 100, 0)]), _record("ssh-1", [(T0 + 50, 1), (T0 + 650, 4)])]
    fig = figs.rate(records, STATS)
    assert isinstance(fig.x_range, Range1d)
    assert (fig.x_range.end - fig.x_range.start).total_seconds() >= 600
    assert _legend_labels(fig) == ["ssh-0", "ssh-1"]


def test_rate_needs_at_least_one_interval():
    assert figs.rate([_record("ssh-0", [(T0 + 100, 0)])], STATS) is None


def test_fleet_total_sums_each_workers_latest_count():
    a = [[1.0, 1], [3.0, 3]]
    b = [[2.0, 5]]
    assert figs._fleet_total([a, b]) == ([1.0, 2.0, 3.0], [1, 6, 8])


def test_worker_history_merges_the_recent_window():
    record = _record("ssh-0", [(T0 + 600, 2), (T0 + 900, 3)])
    record["history"] = [[T0 + 900, 3]]  # thinned: the window still has the earlier point
    assert figs._worker_history(record) == [[T0, 0], [T0 + 600, 2], [T0 + 900, 3]]


def test_cumulative_starts_each_worker_at_zero_and_adds_the_fleet():
    records = [_record("ssh-0", [(T0 + 600, 2)]), _record("ssh-1", [(T0 + 900, 3)])]
    fig = figs.cumulative(records, STATS)
    assert _legend_labels(fig) == ["ssh-0", "ssh-1", "fleet"]
    fleet = fig.legend[0].items[-1].renderers[0].data_source.data
    assert list(fleet["y"]) == [0, 0, 2, 5]
    assert figs.cumulative([], STATS) is None

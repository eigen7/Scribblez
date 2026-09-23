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


def _ys(fig) -> list[int]:
    return list(fig.renderers[0].data_source.data["y"])


def test_worker_history_merges_the_recent_window():
    record = _record("ssh-0", [(T0 + 600, 2), (T0 + 900, 3)])
    record["history"] = [[T0 + 900, 3]]  # thinned: the window still has the earlier point
    assert figs._worker_history(record) == [[T0, 0], [T0 + 600, 2], [T0 + 900, 3]]


def test_fleet_total_sums_each_workers_latest_count():
    a = [[1.0, 1], [3.0, 3]]
    b = [[2.0, 5]]
    assert figs._fleet_total([a, b]) == [[1.0, 1], [2.0, 6], [3.0, 8]]


def test_cumulative_plots_one_worker_or_the_fleet_from_zero():
    records = [_record("ssh-0", [(T0 + 600, 2)]), _record("ssh-1", [(T0 + 900, 3)])]
    assert _ys(figs.cumulative(records, STATS, "ssh-1")) == [0, 3]
    assert _ys(figs.cumulative(records, STATS, figs.FLEET)) == [0, 0, 2, 5]
    assert figs.cumulative([], STATS, figs.FLEET) is None


def test_cumulative_spans_a_single_cycle_visibly():
    """A lone point would otherwise leave Bokeh a zero-width datetime range,
    labelled in microseconds."""
    fig = figs.cumulative([_record("ssh-0", [(T0 + 600, 2)])], STATS, "ssh-0")
    assert isinstance(fig.x_range, Range1d)
    assert (fig.x_range.end - fig.x_range.start).total_seconds() >= 600

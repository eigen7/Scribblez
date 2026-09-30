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


def _timed_record(ts: list[float], gen_s: float, upload_s: float) -> dict:
    """A generator record whose cycles ended at `ts`, each with the given phases."""
    samples = [
        {"t": t, "units": 1000, "units_total": 1000 * (i + 1), "bytes": 0,
         "gen_s": gen_s, "upload_s": upload_s}
        for i, t in enumerate(ts)
    ]  # fmt: skip
    return {
        "worker_id": "ssh-1", "kind": "ssh", "units_total": 1000 * len(ts),
        "cycles_total": len(ts), "updated_at": ts[-1], "recent": samples,
    }  # fmt: skip


GEN_STATS = StatsSpec(
    unit="games",
    phases={"gen_s": "self-play", "upload_s": "deliver"},
    background=frozenset({"upload_s"}),
)


def test_cycle_time_is_wall_clock_and_other_excludes_background_phases():
    """Cycles 4 s apart that spend 2.5 s in self-play: the 1.5 s delivery
    overlaps the next cycle, so the unaccounted 1.5 s is `other`."""
    record = _timed_record([T0, T0 + 4, T0 + 8, T0 + 12], gen_s=2.5, upload_s=1.5)
    row = figs.worker_summary(record, GEN_STATS, now=T0 + 13)
    assert row["cycle_s"] == 4.0
    assert row["phases"] == {"gen_s": 2.5, "upload_s": 1.5}
    assert row["other_s"] == 1.5
    assert row["units_per_hour"] == 3000 / 12 * 3600
    assert row["stale"] is False


def test_other_never_goes_negative():
    """Phase means and the wall-clock span cover slightly different samples,
    so a phase can exceed the cycle time by a hair."""
    record = _timed_record([T0, T0 + 2, T0 + 4], gen_s=2.1, upload_s=0.0)
    assert figs.worker_summary(record, GEN_STATS, now=T0 + 5)["other_s"] == 0.0


def test_a_single_sample_has_no_cycle_time():
    record = _timed_record([T0], gen_s=2.5, upload_s=1.5)
    row = figs.worker_summary(record, GEN_STATS, now=T0 + 1)
    assert row["cycle_s"] is None and row["other_s"] is None


def test_staleness_scales_with_cycle_time_above_a_floor():
    fast = _timed_record([T0, T0 + 4], gen_s=4.0, upload_s=0.0)
    assert figs.worker_summary(fast, GEN_STATS, now=T0 + 4 + 119)["stale"] is False
    assert figs.worker_summary(fast, GEN_STATS, now=T0 + 4 + 121)["stale"] is True
    slow = _timed_record([T0, T0 + 100], gen_s=100.0, upload_s=0.0)
    assert figs.worker_summary(slow, GEN_STATS, now=T0 + 100 + 499)["stale"] is False
    assert figs.worker_summary(slow, GEN_STATS, now=T0 + 100 + 501)["stale"] is True


def test_pace_sums_the_roles_live_workers_only():
    a = _timed_record([T0, T0 + 4], gen_s=4.0, upload_s=0.0)  # 900K games/hr
    b = {**_timed_record([T0, T0 + 4], gen_s=4.0, upload_s=0.0), "worker_id": "ssh-2"}
    dead = {**_timed_record([T0 - 900, T0 - 896], gen_s=4.0, upload_s=0.0), "worker_id": "ssh-3"}
    trainer = {**a, "worker_id": "ssh-0", "role": "train"}
    for r in (a, b, dead):
        r["role"] = "generate"
    now = T0 + 5
    assert figs.pace([a, b, dead, trainer], "generate", GEN_STATS, now) == 2 * 900_000
    assert figs.pace([dead], "generate", GEN_STATS, now) is None

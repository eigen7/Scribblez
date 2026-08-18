"""Tests for the per-worker stats record (scribblez.workloads.worker)."""

import json

from cloud.sinks import LocalSink
from scribblez import workloads
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, stats_rel_path


def _ctx(tmp_path, **overrides):
    spec = workloads.get("kill_test")
    fields = dict(
        spec=spec,
        role=spec.role("generate"),
        tag="t",
        params=spec.params_cls(),
        worker_id="ssh-0",
        threads=4,
        max_cycles=0,
        sink=LocalSink(tmp_path),
        kind="ssh",
    )
    return WorkerContext(**(fields | overrides))


def _published(tmp_path) -> dict:
    return json.loads((tmp_path / stats_rel_path("ssh-0")).read_text())


def test_cycle_publishes_counters_and_a_sample(tmp_path):
    stats = WorkerStats(_ctx(tmp_path))
    stats.cycle_done({"gen_s": 1.5}, units=1000, nbytes=42)
    record = _published(tmp_path)
    assert (record["units_total"], record["cycles_total"]) == (1000, 1)
    assert record["kind"] == "ssh"  # the slot's kind, not the sink's
    assert [s["units"] for s in record["recent"]] == [1000]


def test_a_restarted_worker_resumes_its_slot_counters(tmp_path):
    """The pacing gate stops and restarts generators constantly; totals that
    reset each time would read as work undone."""
    first = WorkerStats(_ctx(tmp_path))
    first.cycle_done({"gen_s": 1.5}, units=1000, nbytes=42)
    first.cycle_done({"gen_s": 1.5}, units=1000, nbytes=42)
    started_at = _published(tmp_path)["started_at"]

    second = WorkerStats(_ctx(tmp_path))
    second.cycle_done({"gen_s": 1.5}, units=1000, nbytes=42)
    record = _published(tmp_path)
    assert (record["units_total"], record["cycles_total"]) == (3000, 3)
    assert record["started_at"] == started_at  # the slot's first start, not this run's
    # The rate window measures now, so it holds only this run's samples.
    assert [s["units_total"] for s in record["recent"]] == [3000]


def test_a_first_run_starts_from_zero(tmp_path):
    WorkerStats(_ctx(tmp_path, worker_id="local-0")).cycle_done({}, units=5, nbytes=0)
    assert json.loads((tmp_path / stats_rel_path("local-0")).read_text())["units_total"] == 5

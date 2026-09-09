"""Tests for the generate role's off-generation-path chunk delivery
(scribblez.workloads.selfplay_gen)."""

import threading
import time

from scribblez import workloads
from scribblez.workloads import selfplay_gen
from scribblez.workloads.base import WorkerContext


class RecordingSink:
    """A sink whose `deliver` records order and, on request, blocks a given
    call number until released -- standing in for LocalSink/R2Sink without
    touching a filesystem tree or the network."""

    kind = "local"

    def __init__(self, block_call: int | None = None, fail_call: int | None = None):
        self.delivered: list[tuple[str, str]] = []
        self._block_call = block_call
        self._fail_call = fail_call
        self._n = 0
        self.call_started = threading.Event()
        self.release_call = threading.Event()

    def deliver(self, src, data_rel) -> int:
        self._n += 1
        if self._n == self._block_call:
            self.call_started.set()
            self.release_call.wait(timeout=5)
        if self._n == self._fail_call:
            raise AssertionError(f"upload of {src.name} failed: simulated failure")
        self.delivered.append((src.name, data_rel))
        src.unlink()
        return len(data_rel)

    def push_json(self, rel_path, obj):
        pass

    def read_json(self, rel_path):
        return None


def _ctx(tmp_path, sink, max_cycles=0, worker_id="local-0"):
    spec = workloads.get("position_eval")
    return WorkerContext(
        spec=spec,
        role=spec.role("generate"),
        tag="t",
        params=spec.params_cls(),
        worker_id=worker_id,
        threads=2,
        max_cycles=max_cycles,
        sink=sink,
        mount_root=tmp_path,
    )


def _fake_run_games(calls: list):
    """A run_games stand-in: records the chunk dir it was called with and
    writes one small .slog file into it, as the real binary would."""

    def run_games(out_dir, **kwargs):
        calls.append(out_dir)
        # Named after the (per-cycle) chunk dir, so each cycle's delivery is
        # distinguishable -- a real chunk gets its name from play_game's
        # timestamp, unique the same way.
        (out_dir / f"{out_dir.name}.slog").write_bytes(b"fake-slog")
        return 0

    return run_games


def test_deliveries_happen_in_order(tmp_path, monkeypatch):
    sink = RecordingSink()
    calls = []
    monkeypatch.setattr(selfplay_gen, "run_games", _fake_run_games(calls))

    rc = selfplay_gen.run_generate(_ctx(tmp_path, sink, max_cycles=4))

    assert rc == 0
    assert len(calls) == 4
    # Every cycle's chunk landed in its own subdirectory, so none collided.
    assert len({c.name for c in calls}) == 4
    assert [d[1] for d in sink.delivered] == [f"staging/{c.name}-local-0.slog" for c in calls]


def test_next_cycle_starts_before_the_previous_delivery_completes(tmp_path, monkeypatch):
    sink = RecordingSink(block_call=1)
    calls = []
    monkeypatch.setattr(selfplay_gen, "run_games", _fake_run_games(calls))

    result = {}

    def run():
        result["rc"] = selfplay_gen.run_generate(_ctx(tmp_path, sink, max_cycles=2))

    runner_thread = threading.Thread(target=run)
    runner_thread.start()
    try:
        assert sink.call_started.wait(timeout=5), "first delivery never started"
        deadline = time.monotonic() + 5
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        # Cycle 2 generated while cycle 1's delivery is still blocked: nothing
        # has been delivered yet even though both chunks are generated.
        assert len(calls) == 2
        assert sink.delivered == []
    finally:
        sink.release_call.set()
        runner_thread.join(timeout=5)

    assert not runner_thread.is_alive()
    assert result["rc"] == 0
    assert len(sink.delivered) == 2


def test_a_failing_delivery_fails_the_runner(tmp_path, monkeypatch):
    sink = RecordingSink(fail_call=1)
    calls = []
    monkeypatch.setattr(selfplay_gen, "run_games", _fake_run_games(calls))

    try:
        selfplay_gen.run_generate(_ctx(tmp_path, sink, max_cycles=2))
    except AssertionError as e:
        assert "simulated failure" in str(e)
    else:
        raise AssertionError("expected the delivery failure to propagate")


def test_stop_drains_what_is_pending(tmp_path, monkeypatch):
    """A WorkerStopped mid-loop (the SIGTERM path) still delivers whatever was
    already queued before the exception was raised."""
    sink = RecordingSink()
    calls = []

    def flaky_run_games(out_dir, **kwargs):
        calls.append(out_dir)
        (out_dir / "game.slog").write_bytes(b"fake-slog")
        if len(calls) == 2:
            raise selfplay_gen.WorkerStopped
        return 0

    monkeypatch.setattr(selfplay_gen, "run_games", flaky_run_games)

    rc = selfplay_gen.run_generate(_ctx(tmp_path, sink, max_cycles=0))

    assert rc == 0
    assert len(calls) == 2
    # Cycle 1's chunk was queued and delivered before the stop; cycle 2's
    # in-flight chunk (WorkerStopped raised out of run_games itself) never was.
    assert len(sink.delivered) == 1


def test_max_cycles_still_bounds_the_loop(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(selfplay_gen, "run_games", _fake_run_games(calls))

    rc = selfplay_gen.run_generate(_ctx(tmp_path, RecordingSink(), max_cycles=3))

    assert rc == 0
    assert len(calls) == 3

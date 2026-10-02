"""The trainer's artifacts through its sink (cloud/sinks.py fetch_data_dir /
fetch_file / deliver_output, and position_eval/trainer.py's use of them): a
trainer touches nothing it does not already have on its machine."""

import json

import pytest
from cloud import sinks
from cloud.sinks import LocalSink
from scribblez.generational import lifecycle
from scribblez.paths import POSITION_EVAL, TagPaths


@pytest.fixture
def paths(tmp_path) -> TagPaths:
    p = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    p.root.mkdir(parents=True)
    return p


def test_the_local_sink_finds_artifacts_where_they_are(paths):
    sink = LocalSink(paths.root)
    gen = paths.generation_dir(2)
    assert not sink.fetch_data_dir("generations/gen_000002", gen)
    gen.mkdir(parents=True)
    assert sink.fetch_data_dir("generations/gen_000002", gen)
    assert not sink.fetch_file("train_state.json", paths.train_state_path)
    paths.train_state_path.write_text("{}")
    assert sink.fetch_file("train_state.json", paths.train_state_path)
    sink.deliver_output(paths.train_state_path, "train_state.json", keep=True)
    assert paths.train_state_path.exists()  # nothing to deliver, nothing removed
    with pytest.raises(AssertionError):
        sink.fetch_file("train_state.json", paths.root / "elsewhere.json")


class _FakeSink:
    """A sink whose bucket holds complete generations and, optionally, a
    checkpoint; records what was asked of it."""

    kind = "ssh"

    def __init__(self, generations=(), checkpoint=False):
        self.generations = set(generations)
        self.checkpoint = checkpoint
        self.fetched = []

    def fetch_data_dir(self, data_rel, dest):
        self.fetched.append(data_rel)
        index = int(data_rel.rsplit("_", 1)[1])
        if index not in self.generations:
            return False
        dest.mkdir(parents=True, exist_ok=True)
        lifecycle.write_manifest(dest, {"index": index, "status": lifecycle.COMPLETE})
        return True

    def fetch_file(self, rel, dest):
        """The bucket's state in the pre-pair layout, if it has one."""
        self.fetched.append(rel)
        if not self.checkpoint or rel not in ("checkpoints/model.pt", "train_state.json"):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        state = {"generation_index": 7, "rows_trained": 700}
        dest.write_text(json.dumps(state) if rel.endswith(".json") else "pt")
        return True

    def list_dirs(self, rel):
        return []  # no state pairs


def test_wait_pulls_the_generation_through_the_sink(paths, monkeypatch):
    pytest.importorskip("torch")
    from scribblez.position_eval import trainer

    monkeypatch.setattr(trainer, "POLL_SECONDS", 0)
    sink = _FakeSink()
    ticks = iter(range(3))

    def sleep(_):
        if next(ticks) == 1:
            sink.generations.add(4)  # published meanwhile

    monkeypatch.setattr(trainer.time, "sleep", sleep)
    trainer.wait_for_generation(paths, 4, sink)
    assert lifecycle.is_complete(paths.generation_dir(4))
    assert sink.fetched.count("generations/gen_000004") == 3
    # Already complete locally: the sink is not consulted.
    sink.fetched.clear()
    trainer.wait_for_generation(paths, 4, sink)
    assert sink.fetched == []


def test_a_failed_data_home_ends_the_wait(paths, monkeypatch):
    """Nothing would ever complete the generation, so the trainer must fail
    rather than wait forever with its generators parked."""
    pytest.importorskip("torch")
    from scribblez.position_eval import trainer

    class _DeadHome:
        def check(self):
            raise RuntimeError("data home died")

    monkeypatch.setattr(trainer.time, "sleep", lambda _: pytest.fail("waited on a dead home"))
    with pytest.raises(RuntimeError, match="data home died"):
        trainer.wait_for_generation(paths, 4, _FakeSink(), home=_DeadHome())


def test_a_fresh_machine_takes_the_window(paths):
    pytest.importorskip("torch")
    from scribblez.position_eval import trainer

    sink = _FakeSink(generations={4, 5, 6})
    trainer.ensure_window(paths, sink, cursor=7, window=4)
    # Generations 3..6 were asked for; 3 was never published (evicted) and
    # the window is just shorter for it.
    assert [f for f in sink.fetched if f.startswith("generations")] == [
        f"generations/gen_{i:06d}" for i in (3, 4, 5, 6)
    ]
    assert lifecycle.window_dirs(paths, 6, 4) == [paths.generation_dir(i) for i in (4, 5, 6)]


def test_the_local_sink_follows_the_worker_mount_root(tmp_path, monkeypatch):
    from scribblez import workloads

    spec = workloads.get("position_eval")
    sink = sinks.make_sink(spec, "t", tmp_path)
    sink.push_json("records/run.json", {})
    assert (tmp_path / "tags" / "position_eval" / "t" / "records" / "run.json").exists()


def test_the_local_sink_removes_an_output_and_has_nothing_to_pull(paths):
    sink = LocalSink(paths.root)
    store = paths.data_dir / "slogs"
    store.mkdir(parents=True)
    (store / "a.mset").touch()
    (store / "a.slog").touch()
    sink.fetch_data_files("slogs", store)  # the store is its own
    assert (store / "a.mset").exists()
    assert sink.count_data_files("slogs", ".mset") == 1
    assert sink.count_data_files("slogs", ".sobs") == 0
    assert sink.count_data_files("nowhere", ".mset") == 0
    sink.remove_output("data/slogs/a.mset")
    sink.remove_output("data/slogs/a.mset")  # absent is success
    assert not (store / "a.mset").exists()

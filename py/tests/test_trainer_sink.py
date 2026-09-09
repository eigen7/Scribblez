"""The trainer's artifacts through its sink (cloud/sinks.py fetch_data_dir /
fetch_file / deliver_output, and position_eval/trainer.py's use of them): a
trainer on the controller's machine touches nothing it does not already
have; one speaking to the bucket pulls generations and restores from it,
and delivers its outputs back."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cloud import sinks
from cloud.sinks import LocalSink, R2Sink
from scribblez.generational import lifecycle
from scribblez.paths import POSITION_EVAL, TagPaths

R2 = SimpleNamespace(bucket="b")


class _Rclone:
    """rclone over a dict of objects; copy/copyto land real files."""

    def __init__(self, objects=()):
        self.objects = set(objects)  # bucket keys (without the "r2:b/" head)
        self.calls = []

    @staticmethod
    def _key(path):
        return path.split("/", 1)[1]

    def __call__(self, r2, *args, capture=False, input_text=None):
        self.calls.append(args)
        op = args[0]
        if op == "lsf":
            key = self._key(args[1])
            present = key in self.objects or any(k.startswith(key + "/") for k in self.objects)
            return SimpleNamespace(returncode=0, stdout="x\n" if present else "", stderr="")
        if op == "copy":  # bucket prefix -> local dir, whole
            prefix, dest = self._key(args[-2]), Path(args[-1])
            dest.mkdir(parents=True, exist_ok=True)
            for k in self.objects:
                if k.startswith(prefix + "/"):
                    (dest / k[len(prefix) + 1 :]).write_text(k)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if op == "copyto":
            src, dst = args[1], args[2]
            if src.startswith("r2:"):
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_text(self._key(src))
            else:
                self.objects.add(self._key(dst))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(args)


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


def test_the_r2_sink_pulls_a_generation_only_once_its_manifest_is_there(paths, monkeypatch):
    rc = _Rclone({"position_eval/t/generations/gen_000002/a.slog"})
    monkeypatch.setattr(sinks, "rclone", rc)
    sink = R2Sink(R2, "position_eval", "t")
    gen = paths.generation_dir(2)
    assert not sink.fetch_data_dir("generations/gen_000002", gen)  # chunks but no manifest yet
    assert not gen.exists()
    rc.objects.add("position_eval/t/generations/gen_000002/manifest.json")
    assert sink.fetch_data_dir("generations/gen_000002", gen)
    assert sorted(p.name for p in gen.iterdir()) == ["a.slog", "manifest.json"]
    assert rc.calls[-1][:2] == ("copy", "--size-only")


def test_the_r2_sink_fetches_and_delivers_root_files(paths, monkeypatch):
    rc = _Rclone()
    monkeypatch.setattr(sinks, "rclone", rc)
    sink = R2Sink(R2, "position_eval", "t")
    assert not sink.fetch_file("checkpoints/model.pt", paths.rolling_checkpoint)
    export = paths.onnx_dir / "model_epoch_0003.onnx"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"onnx")
    sink.deliver_output(export, "models/model_epoch_0003.onnx")
    assert not export.exists()  # the bucket is where exports live; the pod disk is scratch
    paths.checkpoints_dir.mkdir()
    paths.rolling_checkpoint.write_bytes(b"pt")
    sink.deliver_output(paths.rolling_checkpoint, "checkpoints/model.pt", keep=True)
    assert paths.rolling_checkpoint.exists()  # a resume needs it
    assert {
        "position_eval/t/models/model_epoch_0003.onnx",
        "position_eval/t/checkpoints/model.pt",
    } <= rc.objects
    other = TagPaths("t", POSITION_EVAL, mount_root=paths.mount_root / "other")
    assert sink.fetch_file("checkpoints/model.pt", other.rolling_checkpoint)
    assert other.rolling_checkpoint.read_text() == "position_eval/t/checkpoints/model.pt"


# --- the trainer's use of them ------------------------------------------------


class _FakeSink:
    """A sink whose bucket holds complete generations and, optionally, a
    checkpoint; records what was asked of it."""

    kind = "cloud"

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
        self.fetched.append(rel)
        if not self.checkpoint:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"generation_index": 7}) if rel.endswith(".json") else "pt")
        return True


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


def test_a_fresh_machine_restores_and_takes_the_window(paths):
    pytest.importorskip("torch")
    from scribblez.position_eval import trainer

    sink = _FakeSink(generations={4, 5, 6}, checkpoint=True)
    trainer.restore_from_sink(paths, sink)
    assert paths.rolling_checkpoint.read_text() == "pt"
    assert json.loads(paths.train_state_path.read_text()) == {"generation_index": 7}
    trainer.ensure_window(paths, sink, cursor=7, window=4)
    # Generations 3..6 were asked for; 3 was never published (evicted) and
    # the window is just shorter for it.
    assert [f for f in sink.fetched if f.startswith("generations")] == [
        f"generations/gen_{i:06d}" for i in (3, 4, 5, 6)
    ]
    assert lifecycle.window_dirs(paths, 6, 4) == [paths.generation_dir(i) for i in (4, 5, 6)]
    # A machine with its own checkpoint keeps it.
    sink.fetched.clear()
    trainer.restore_from_sink(paths, sink)
    assert sink.fetched == []


def test_the_local_sink_follows_the_worker_mount_root(tmp_path, monkeypatch):
    from scribblez import workloads

    monkeypatch.setenv("SCZ_SINK", "local")
    spec = workloads.get("position_eval")
    sink = sinks.make_sink(spec, "t", tmp_path)
    sink.push_json("records/run.json", {})
    assert (tmp_path / "tags" / "position_eval" / "t" / "records" / "run.json").exists()

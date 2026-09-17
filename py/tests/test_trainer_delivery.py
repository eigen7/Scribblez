"""The trainer's delivery thread (position_eval/trainer.py OutputDeliverer,
_deliver_generation): outputs leave off the training critical path, in
order, bounded, with failures surfacing on the training thread."""

import threading
import time
from pathlib import Path

import pytest
from cloud.sinks import LocalSink
from scribblez.generational.records import TrainRecorder
from scribblez.paths import TagPaths
from scribblez.position_eval import trainer


class _RecordingSink:
    kind = "cloud"

    def __init__(self):
        self.calls: list[tuple] = []

    def deliver_output(self, src: Path, rel_path: str, *, keep: bool = False):
        self.calls.append(("deliver", rel_path, src.name, keep))
        if not keep:
            src.unlink()

    def push_json(self, rel_path: str, obj: dict):
        self.calls.append(("json", rel_path, obj["generation"], len(obj["control_events"])))

    def push_file(self, src: Path, rel_path: str):
        self.calls.append(("file", rel_path))


def test_steps_run_in_submission_order_on_another_thread():
    d = trainer.OutputDeliverer()
    seen = []
    for i in range(3):
        d.submit(f"step {i}", lambda i=i: seen.append((i, threading.current_thread().name)))
    d.drain()
    assert [i for i, _ in seen] == [0, 1, 2]
    assert all(name != threading.main_thread().name for _, name in seen)
    assert [what for what, _ in d.collect()] == ["step 0", "step 1", "step 2"]


def test_submitting_blocks_once_the_queue_is_full():
    """A bucket that has fallen MAX_PENDING_DELIVERIES generations behind is
    a problem to stop for, not to keep piling onto."""
    gate = threading.Event()
    d = trainer.OutputDeliverer(max_pending=1)
    d.submit("slow", gate.wait)  # occupies the thread
    d.submit("queued", lambda: None)  # fills the one slot
    unblocked = []

    def third():
        d.submit("third", lambda: None)
        unblocked.append(time.monotonic())

    t = threading.Thread(target=third)
    t.start()
    t.join(timeout=0.3)
    assert t.is_alive() and not unblocked  # blocked on the full queue
    gate.set()
    t.join(timeout=5)
    assert unblocked
    d.drain()


def test_a_failed_step_is_raised_on_the_training_thread():
    d = trainer.OutputDeliverer()

    def boom():
        raise AssertionError("upload of x failed")

    d.submit("generation 3", boom)
    with pytest.raises(RuntimeError, match="delivering generation 3 failed: upload of x failed"):
        d.drain()
    e = trainer.OutputDeliverer()
    e.submit("generation 3", boom)
    time.sleep(0.2)
    with pytest.raises(RuntimeError, match="generation 3"):
        e.submit("generation 4", lambda: None)


def test_a_generations_deliveries_end_with_its_record(tmp_path):
    paths = TagPaths("t", "position_eval", mount_root=tmp_path)
    paths.onnx_dir.mkdir(parents=True)
    paths.rolling_checkpoint.parent.mkdir(parents=True)
    (paths.onnx_dir / "shared.bin").write_bytes(b"blob")
    paths.onnx_path(4).write_bytes(b"onnx")
    paths.rolling_checkpoint.write_bytes(b"ckpt-4")
    paths.train_state_path.write_text('{"generation_index": 5}')
    ckpt_snap = trainer._snapshot(paths.rolling_checkpoint, 4)
    state_snap = trainer._snapshot(paths.train_state_path, 4)
    paths.rolling_checkpoint.write_bytes(b"ckpt-5")  # the next generation rewrites it
    sink = _RecordingSink()
    recorder = TrainRecorder(sink)
    recorder.control_event(100, "lr", 0.1)
    staged = recorder.stage_generation(4, 1000, {"loss": 0.5})
    recorder.control_event(200, "lr", 0.2)  # after staging: the next generation's
    trainer._deliver_generation(sink, paths, 4, ckpt_snap, state_snap, recorder, staged)
    assert sink.calls == [
        ("deliver", "models/shared.bin", "shared.bin", True),
        ("deliver", "models/model_epoch_0004.onnx", "model_epoch_0004.onnx", False),
        ("deliver", "checkpoints/model.pt", "model.pt.gen4", False),
        ("deliver", "train_state.json", "train_state.json.gen4", False),
        ("json", "records/gen_000004.json", 4, 1),
    ]
    assert not ckpt_snap.exists() and not state_snap.exists()
    assert paths.rolling_checkpoint.read_bytes() == b"ckpt-5"
    assert recorder.stage_generation(5, 2000, {}).record["control_events"][0]["value"] == 0.2


def test_the_local_sink_drops_a_snapshot_and_keeps_the_original(tmp_path):
    root = tmp_path / "t"
    (root / "checkpoints").mkdir(parents=True)
    ckpt = root / "checkpoints" / "model.pt"
    ckpt.write_bytes(b"x")
    snap = trainer._snapshot(ckpt, 1)
    LocalSink(root).deliver_output(snap, "checkpoints/model.pt")
    assert not snap.exists() and ckpt.read_bytes() == b"x"
    LocalSink(root).deliver_output(ckpt, "checkpoints/model.pt", keep=True)
    assert ckpt.exists()

"""RoleSpec.inputs' worker half: where a runner reads an out-of-tag input
from (workloads.base.resolve_input)."""

import threading
from types import SimpleNamespace

import pytest
from scribblez.workloads import base


def _ctx(tmp_path, kind="ssh"):
    return SimpleNamespace(kind=kind, tag_paths=lambda: SimpleNamespace(root=tmp_path / "tag"))


def test_the_source_wins_where_it_exists(tmp_path):
    src = tmp_path / "teacher.onnx"
    src.write_bytes(b"x")
    assert base.resolve_input(_ctx(tmp_path), "inputs/t.onnx", src) == src


def test_a_copy_pushed_into_the_container_is_taken(tmp_path):
    staged = tmp_path / "tag" / "inputs" / "t.onnx"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"x")
    assert base.resolve_input(_ctx(tmp_path), "inputs/t.onnx", tmp_path / "no") == staged


def test_a_remote_slot_waits_for_the_push(tmp_path, monkeypatch):
    """The controller pushes a container's inputs right after creating it,
    which may land after the runner first looks."""
    monkeypatch.setattr(base, "INPUT_POLL_SECONDS", 0.01)
    staged = tmp_path / "tag" / "inputs" / "t.onnx"

    def push():
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"x")

    timer = threading.Timer(0.05, push)
    timer.start()
    assert base.resolve_input(_ctx(tmp_path), "inputs/t.onnx", tmp_path / "no") == staged
    timer.join()


def test_a_remote_slot_gives_up_and_a_local_one_does_not_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "INPUT_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(base, "INPUT_POLL_SECONDS", 0.01)
    with pytest.raises(FileNotFoundError, match="inputs/t.onnx"):
        base.resolve_input(_ctx(tmp_path), "inputs/t.onnx", tmp_path / "no")
    monkeypatch.setattr(base, "INPUT_WAIT_SECONDS", 600)  # a wait would hang the test
    with pytest.raises(FileNotFoundError):
        base.resolve_input(_ctx(tmp_path, kind="local"), "inputs/t.onnx", tmp_path / "no")

"""RoleSpec.inputs' worker half: where a runner reads an out-of-tag input
from (workloads.base.resolve_input)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from scribblez.workloads import base


class _Sink:
    def __init__(self, staged: bytes | None):
        self.staged = staged
        self.fetches = 0

    def fetch_file(self, rel, dest: Path) -> bool:
        self.fetches += 1
        if self.staged is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.staged)
        return True


def _ctx(tmp_path, sink, kind="ssh"):
    return SimpleNamespace(
        kind=kind, sink=sink, tag_paths=lambda: SimpleNamespace(root=tmp_path / "tag")
    )


def test_the_source_wins_where_it_exists(tmp_path):
    src = tmp_path / "teacher.onnx"
    src.write_bytes(b"x")
    sink = _Sink(b"y")
    assert base.resolve_input(_ctx(tmp_path, sink), "inputs/t.onnx", src) == src
    assert sink.fetches == 0


def test_a_copy_pushed_into_the_container_is_taken_without_the_sink(tmp_path):
    staged = tmp_path / "tag" / "inputs" / "t.onnx"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"x")
    sink = _Sink(None)
    assert base.resolve_input(_ctx(tmp_path, sink), "inputs/t.onnx", tmp_path / "no") == staged
    assert sink.fetches == 0


def test_a_bucket_copy_is_fetched_under_the_tag_root(tmp_path):
    sink = _Sink(b"x")
    got = base.resolve_input(_ctx(tmp_path, sink), "inputs/t.onnx", tmp_path / "no")
    assert got == tmp_path / "tag" / "inputs" / "t.onnx" and got.read_bytes() == b"x"


def test_a_remote_slot_waits_for_staging_and_a_local_one_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "INPUT_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(base, "INPUT_POLL_SECONDS", 0.01)
    sink = _Sink(None)
    with pytest.raises(FileNotFoundError, match="inputs/t.onnx"):
        base.resolve_input(_ctx(tmp_path, sink), "inputs/t.onnx", tmp_path / "no")
    assert sink.fetches > 1  # polled
    local = _Sink(None)
    with pytest.raises(FileNotFoundError):
        base.resolve_input(_ctx(tmp_path, local, kind="local"), "inputs/t.onnx", tmp_path / "no")
    assert local.fetches == 0  # nothing stages for it: no wait, no sink

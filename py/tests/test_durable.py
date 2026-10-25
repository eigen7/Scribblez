"""cloud/durable.py: a rename into place is made only after its data is on
disk, and is itself flushed."""

import os
from pathlib import Path

from cloud import durable


def _record_calls(monkeypatch) -> list[tuple[str, str]]:
    calls = []
    real_open, real_fsync, real_replace = os.open, os.fsync, os.replace
    names = {}

    def opened(path, flags):
        fd = real_open(path, flags)
        names[fd] = Path(path).name
        return fd

    def fsynced(fd):
        calls.append(("fsync", names[fd]))
        real_fsync(fd)

    def replaced(src, dest):
        calls.append(("replace", Path(dest).name))
        real_replace(src, dest)

    monkeypatch.setattr(durable.os, "open", opened)
    monkeypatch.setattr(durable.os, "fsync", fsynced)
    monkeypatch.setattr(durable.os, "replace", replaced)
    return calls


def test_a_file_is_flushed_before_the_rename_and_its_directory_after(tmp_path, monkeypatch):
    src, dest = tmp_path / "a.tmp", tmp_path / "out" / "a.pool"
    dest.parent.mkdir()
    src.write_bytes(b"pool")
    calls = _record_calls(monkeypatch)
    durable.replace(src, dest)
    assert calls == [("fsync", "a.tmp"), ("replace", "a.pool"), ("fsync", "out")]
    assert dest.read_bytes() == b"pool"


def test_a_directorys_every_file_is_flushed_before_the_rename(tmp_path, monkeypatch):
    src = tmp_path / "stage"
    (src / "sub").mkdir(parents=True)
    (src / "x").write_bytes(b"x")
    (src / "sub" / "y").write_bytes(b"y")
    calls = _record_calls(monkeypatch)
    durable.replace(src, tmp_path / "gen_000001")
    flushed = {name for op, name in calls[:-2] if op == "fsync"}
    assert flushed == {"stage", "sub", "x", "y"}
    assert calls[-2:] == [("replace", "gen_000001"), ("fsync", tmp_path.name)]
    assert (tmp_path / "gen_000001" / "sub" / "y").read_bytes() == b"y"

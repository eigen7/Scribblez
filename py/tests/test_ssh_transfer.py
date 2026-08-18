"""Tests for reading an ssh worker's results out of its container.

The container commands are plain POSIX sh and coreutils, so the fake machine
here runs them for real against a directory standing in for the container's
filesystem -- everything but the `ssh ... docker exec` wrapper is exercised.
"""

import subprocess

import pytest
from cloud.ssh_transfer import INCOMING_DIR, pull_results

DATA_DIRS = ["data/staging"]


class _FakeMachine:
    """Runs container commands against `root`, a stand-in for the container."""

    def __init__(self, root):
        self.root = root
        self.execs = []

    def read_from_container(self, container: str, command: list[str]) -> bytes:
        assert command[:2] == ["sh", "-c"]
        return subprocess.run(
            ["sh", "-c", command[2]], cwd=self.root, capture_output=True, check=True
        ).stdout

    def exec_in_container(self, container: str, command: list[str]):
        self.execs.append(command)
        subprocess.run(command, cwd=self.root, check=True)


def _container(tmp_path, **files):
    root = tmp_path / "container"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pull(tmp_path, remote_root, local_root):
    machine = _FakeMachine(tmp_path)
    names = pull_results(
        machine,
        "c",
        remote_root=str(remote_root),
        local_root=local_root,
        data_dirs=DATA_DIRS,
    )
    return machine, names


def test_pull_moves_delivered_data_and_copies_records(tmp_path):
    remote = _container(
        tmp_path,
        **{
            "data/staging/c1-ssh-0.slog": "chunk one",
            "data/staging/c2-ssh-0.slog": "chunk two",
            "stats/ssh-0.json": '{"units_total": 2000}',
            "params/ssh-0.json": "{}",
        },
    )
    local = tmp_path / "local"
    local.mkdir()
    machine, names = _pull(tmp_path, remote, local)

    assert sorted(names) == [
        "data/staging/c1-ssh-0.slog",
        "data/staging/c2-ssh-0.slog",
        "params/ssh-0.json",
        "stats/ssh-0.json",
    ]
    assert (local / "data/staging/c1-ssh-0.slog").read_text() == "chunk one"
    assert (local / "stats/ssh-0.json").read_text() == '{"units_total": 2000}'

    # Delivered data is gone from the container; the records it keeps writing
    # to -- and reads its counters back from on restart -- stay.
    assert list((remote / "data/staging").iterdir()) == []
    assert (remote / "stats/ssh-0.json").exists()
    assert machine.execs and machine.execs[0][0] == "rm"


def test_pull_leaves_nothing_staged_behind(tmp_path):
    """A chunk is renamed into place from a scratch dir on the same
    filesystem, so it appears in staging whole or not at all."""
    remote = _container(tmp_path, **{"data/staging/c1.slog": "x"})
    local = tmp_path / "local"
    local.mkdir()
    _pull(tmp_path, remote, local)
    incoming = local / INCOMING_DIR
    assert not any(p.is_file() for p in incoming.rglob("*"))


def test_pull_of_a_worker_that_has_produced_nothing(tmp_path):
    """A container whose directories do not exist yet must read as empty, not
    as a failure: tar refuses to build an empty archive."""
    remote = _container(tmp_path)
    local = tmp_path / "local"
    local.mkdir()
    machine, names = _pull(tmp_path, remote, local)
    assert names == []
    assert machine.execs == []  # nothing to delete


def test_pull_of_a_container_that_is_gone(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    machine, names = _pull(tmp_path, tmp_path / "no-such-root", local)
    assert names == []


def test_a_repulled_chunk_overwrites_rather_than_duplicating(tmp_path):
    """The delete is a separate step after the extraction succeeds, so a
    transfer that dies mid-stream loses nothing and simply re-pulls."""
    remote = _container(tmp_path, **{"data/staging/c1.slog": "second copy"})
    local = tmp_path / "local"
    (local / "data/staging").mkdir(parents=True)
    (local / "data/staging/c1.slog").write_text("first copy")
    _pull(tmp_path, remote, local)
    assert (local / "data/staging/c1.slog").read_text() == "second copy"


def test_a_failed_read_leaves_the_container_untouched(tmp_path):
    """Nothing is deleted until it is safely on disk here."""

    class _Broken(_FakeMachine):
        def read_from_container(self, container, command):
            raise RuntimeError("ssh died mid-stream")

    remote = _container(tmp_path, **{"data/staging/c1.slog": "x"})
    machine = _Broken(tmp_path)
    with pytest.raises(RuntimeError):
        pull_results(
            machine, "c", remote_root=str(remote), local_root=tmp_path / "local",
            data_dirs=DATA_DIRS,
        )  # fmt: skip
    assert (remote / "data/staging/c1.slog").exists()
    assert machine.execs == []

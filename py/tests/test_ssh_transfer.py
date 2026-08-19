"""Tests for reading an ssh worker's results out of its container.

The container commands are plain POSIX sh and coreutils, so the fake machine
here runs them for real against a directory standing in for the container's
filesystem -- everything but the `ssh ... docker exec` wrapper is exercised.
"""

import subprocess

import pytest
from cloud.ssh_transfer import (
    BATCH,
    INCOMING_DIR,
    collect_command,
    pull_results,
    sweep_stopped,
)

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


def _pull(tmp_path, remote_root, local_root, *, machine=None, batch=BATCH):
    machine = machine or _FakeMachine(tmp_path)
    result = pull_results(
        machine,
        "c",
        remote_root=str(remote_root),
        local_root=local_root,
        data_dirs=DATA_DIRS,
        batch=batch,
    )
    return machine, result


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
    machine, result = _pull(tmp_path, remote, local)

    assert sorted(result.pulled) == [
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
    machine, result = _pull(tmp_path, remote, local)
    assert (result.pulled, result.remaining) == ([], 0)
    assert machine.execs == []  # nothing to delete


def test_pull_of_a_container_that_is_gone(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    machine, result = _pull(tmp_path, tmp_path / "no-such-root", local)
    assert (result.pulled, result.remaining) == ([], 0)


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


def _chunks(tmp_path, count: int):
    return _container(
        tmp_path, **{f"data/staging/{i:04d}-ssh-0.slog": f"chunk {i}" for i in range(count)}
    )


def test_a_pull_takes_a_bounded_batch_and_reports_the_rest(tmp_path):
    """The pull used to take everything waiting, so its cost grew with the
    backlog it was draining -- and once one overran its timeout, none ever
    finished again."""
    remote = _chunks(tmp_path, 40)
    local = tmp_path / "local"
    local.mkdir()
    machine, result = _pull(tmp_path, remote, local, batch=16)
    assert len(result.pulled) == 16
    assert result.remaining == 24
    assert len(list((remote / "data/staging").iterdir())) == 24


def test_repeated_pulls_drain_a_backlog_oldest_first(tmp_path):
    """The drain rate is a floor: a batch per pass, whatever the backlog."""
    remote = _chunks(tmp_path, 40)
    local = tmp_path / "local"
    local.mkdir()
    machine = _FakeMachine(tmp_path)
    pulls = [_pull(tmp_path, remote, local, machine=machine, batch=16)[1] for _ in range(3)]
    first, second, third = pulls

    assert [r.remaining for r in (first, second, third)] == [24, 8, 0]
    assert first.pulled[0] == "data/staging/0000-ssh-0.slog"  # oldest name first
    assert len(list((local / "data/staging").iterdir())) == 40
    assert list((remote / "data/staging").iterdir()) == []


def test_the_container_side_bounds_its_own_transfer():
    """A tar that outlives its ssh client keeps reading the backlog it was
    asked for, and one per pass compounds; the container kills it instead."""
    script = collect_command("/tag", ["data/staging/a.slog"], 45)[2]
    assert "timeout 45 tar" in script


def test_records_come_back_even_with_no_data_waiting(tmp_path):
    """Stats are how a worker reports it is alive; a quiet generator must not
    look stale."""
    remote = _container(tmp_path, **{"stats/ssh-0.json": '{"units_total": 7}'})
    local = tmp_path / "local"
    local.mkdir()
    _, result = _pull(tmp_path, remote, local)
    assert result.pulled == ["stats/ssh-0.json"]
    assert (local / "stats/ssh-0.json").read_text() == '{"units_total": 7}'


def test_a_sweep_reads_a_stopped_container_and_places_files_where_they_belong(tmp_path):
    """docker cp names members relative to the copied directory's parent, so
    "staging/c1.slog" has to land as "data/staging/c1.slog"."""
    import io
    import tarfile

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tar:
        info = tarfile.TarInfo("staging/c1-ssh-0.slog")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"flush"))

    class _Stopped:
        def copy_from_container(self, container, path):
            assert path.endswith("/data/staging")
            return payload.getvalue()

    local = tmp_path / "local"
    local.mkdir()
    names = sweep_stopped(
        _Stopped(), "c", remote_root="/tag", local_root=local, data_dirs=DATA_DIRS
    )
    assert names == ["data/staging/c1-ssh-0.slog"]
    assert (local / "data/staging/c1-ssh-0.slog").read_text() == "flush"


def test_a_sweep_of_a_container_with_nothing_in_it_is_empty(tmp_path):
    class _Empty:
        def copy_from_container(self, container, path):
            return b""  # the path is not there at all

    local = tmp_path / "local"
    local.mkdir()
    assert (
        sweep_stopped(_Empty(), "c", remote_root="/tag", local_root=local, data_dirs=DATA_DIRS)
        == []
    )


def test_an_unparseable_listing_is_reported_as_unknown(tmp_path):
    """The total decides whether a container may be destroyed, so a count
    nobody can vouch for must not read as empty."""
    from cloud.ssh_transfer import parse_listing

    assert parse_listing(b"") == ([], 0)  # no tag root there yet: a real zero
    assert parse_listing(b"data/staging/a.slog\n") == ([], None)  # sentinel missing
    assert parse_listing(b"data/staging/a.slog\nTOTAL 9\n") == (["data/staging/a.slog"], 9)

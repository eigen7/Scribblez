"""Tests for moving files between the controller and an ssh worker's container.

The container commands are plain POSIX sh and coreutils, so the fake machine
here runs them for real against a directory standing in for the container's
filesystem -- everything but the `ssh ... docker exec` wrapper is exercised.
"""

import subprocess

import pytest
from cloud import ssh_transfer
from cloud.ssh_machine import SshMachineError
from cloud.ssh_transfer import (
    BATCH,
    INCOMING_DIR,
    collect_command,
    list_dir,
    pull_results,
    push_file,
    remove_file,
    sweep_stopped,
)

DATA_DIRS = ["data/staging"]


class _FakeMachine:
    """Runs container commands against `root`, a stand-in for the container."""

    def __init__(self, root, push_bytes: int | None = None):
        self.root = root
        self.execs = []
        self.push_bytes = push_bytes  # None: the whole file arrives

    def read_from_container(self, container: str, command: list[str], timeout=None) -> bytes:
        assert command[:2] == ["sh", "-c"]
        return subprocess.run(
            ["sh", "-c", command[2]], cwd=self.root, capture_output=True, check=True
        ).stdout

    def exec_in_container(self, container: str, command: list[str]):
        self.execs.append(command)
        subprocess.run(command, cwd=self.root, check=True)

    def write_to_container(self, container: str, command: list[str], src):
        assert command[:2] == ["sh", "-c"]
        with open(src, "rb") as f:
            data = f.read()[: self.push_bytes]  # a link that dies mid-push sends less
        res = subprocess.run(["sh", "-c", command[2]], cwd=self.root, input=data)
        if res.returncode != 0:
            raise SshMachineError("push failed")


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
        def read_from_container(self, container, command, timeout=None):
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
    """An unbounded pull's cost grows with the backlog it drains, and once one
    overruns its timeout, none ever finishes again."""
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
        def copy_from_container(self, container, path, dest):
            assert path.endswith("/data/staging")
            dest.write_bytes(payload.getvalue())
            return True

    local = tmp_path / "local"
    local.mkdir()
    names = sweep_stopped(
        _Stopped(), "c", remote_root="/tag", local_root=local, data_dirs=DATA_DIRS
    )
    assert names == ["data/staging/c1-ssh-0.slog"]
    assert (local / "data/staging/c1-ssh-0.slog").read_text() == "flush"
    # The archive it streamed through is not left behind.
    assert not list((local / INCOMING_DIR).glob("sweep-*"))


def test_a_sweep_of_a_container_with_nothing_in_it_is_empty(tmp_path):
    class _Empty:
        def copy_from_container(self, container, path, dest):
            return False  # the path is not there at all

    local = tmp_path / "local"
    local.mkdir()
    assert (
        sweep_stopped(_Empty(), "c", remote_root="/tag", local_root=local, data_dirs=DATA_DIRS)
        == []
    )


def test_a_sweep_that_fails_midway_leaves_no_archive_behind(tmp_path):
    """The stream lands in a file so it need not be held in memory; a failed
    sweep must not leave that file in the tag either."""

    class _Broken:
        def copy_from_container(self, container, path, dest):
            dest.write_bytes(b"partial")
            raise SshMachineError("connection reset")

    local = tmp_path / "local"
    local.mkdir()
    with pytest.raises(SshMachineError):
        sweep_stopped(_Broken(), "c", remote_root="/tag", local_root=local, data_dirs=DATA_DIRS)
    assert not list((local / INCOMING_DIR).glob("sweep-*"))


def test_an_unparseable_listing_is_reported_as_unknown(tmp_path):
    """The total decides whether a container may be destroyed, so a count
    nobody can vouch for must not read as empty."""
    from cloud.ssh_transfer import parse_listing

    assert parse_listing(b"").total == 0  # no tag root there yet: a real zero
    assert parse_listing(b"data/staging/a.slog\n").total is None  # sentinel missing
    listing = parse_listing(b"data/staging/a.slog\nTOTAL 9\nBYTES 12\n")
    assert (listing.names, listing.total, listing.nbytes) == (["data/staging/a.slog"], 9, 12)


def test_a_sweep_clears_a_spool_left_by_a_process_that_died(tmp_path):
    """The spool is removed in a finally, which a killed process does not run.
    Nothing else cleans .incoming, so the next sweep does."""
    local = tmp_path / "local"
    (local / INCOMING_DIR).mkdir(parents=True)
    orphan = local / INCOMING_DIR / "sweep-staging.tar"
    orphan.write_bytes(b"x" * 1000)

    class _Empty:
        def copy_from_container(self, container, path, dest):
            return False

    sweep_stopped(_Empty(), "c", remote_root="/tag", local_root=local, data_dirs=DATA_DIRS)
    assert not orphan.exists()


def test_a_push_lands_atomically_where_the_worker_looks(tmp_path):
    container = _container(tmp_path)
    machine = _FakeMachine(container)
    remote_root = str(container / "tag")
    src = tmp_path / "model_epoch_0010.onnx"
    src.write_bytes(b"weights")
    inbox = "data/match_inbox/ssh-0"

    assert list_dir(machine, "c", remote_root=remote_root, rel=inbox) == []
    push_file(machine, "c", remote_root=remote_root, rel_dest=f"{inbox}/{src.name}", src=src)

    landed = container / "tag" / inbox / src.name
    assert landed.read_bytes() == b"weights"
    # Nothing half-written is left under the temporary name the push uses.
    assert [p.name for p in landed.parent.iterdir()] == [src.name]
    assert list_dir(machine, "c", remote_root=remote_root, rel=inbox) == [src.name]


def test_a_push_cut_off_midway_never_lands(tmp_path):
    """`cat` cannot tell a truncated stream from a whole one -- it exits 0 on
    the EOF a dropped link produces just as happily. A short model landing
    under the name the worker polls for would wedge the slot: it reads as a
    match in flight, so nothing replaces it, and the engine dies on it as fast
    as the container can be restarted."""
    container = _container(tmp_path)
    machine = _FakeMachine(container, push_bytes=4096)
    remote_root = str(container / "tag")
    src = tmp_path / "model_epoch_0010.onnx"
    src.write_bytes(b"w" * 100_000)
    inbox = "data/match_inbox/ssh-0"

    with pytest.raises(SshMachineError):
        push_file(machine, "c", remote_root=remote_root, rel_dest=f"{inbox}/{src.name}", src=src)
    assert list_dir(machine, "c", remote_root=remote_root, rel=inbox) == []
    # What it did leave is invisible to the listing, and the next push
    # overwrites it.
    machine.push_bytes = None
    push_file(machine, "c", remote_root=remote_root, rel_dest=f"{inbox}/{src.name}", src=src)
    assert (container / "tag" / inbox / src.name).read_bytes() == src.read_bytes()
    assert [p.name for p in (container / "tag" / inbox).iterdir()] == [src.name]


def test_removing_a_file_from_a_container(tmp_path):
    container = _container(tmp_path)
    remote_root = str(container / "tag")
    inbox = "data/match_inbox/ssh-0"
    played = container / "tag" / inbox / "model_epoch_0010.onnx.done"
    played.parent.mkdir(parents=True)
    played.write_text("x")

    machine = _FakeMachine(container)
    remove_file(machine, "c", remote_root=remote_root, rel=f"{inbox}/{played.name}")
    assert list_dir(machine, "c", remote_root=remote_root, rel=inbox) == []
    # Absent is success: the mark may have gone with a replaced container.
    remove_file(machine, "c", remote_root=remote_root, rel=f"{inbox}/{played.name}")


# ---- a trainer's outputs: byte-bounded batches, state pairs, root records ------


def _pull_outputs(tmp_path, remote, local, *, batch_bytes=None, record_files=()):
    extra = {} if batch_bytes is None else {"batch_bytes": batch_bytes}
    return pull_results(
        _FakeMachine(tmp_path), "c", remote_root=str(remote), local_root=local,
        data_dirs=["models", "records"], pair_dirs={"state": "train_state.json"},
        record_files=record_files, **extra,
    )  # fmt: skip


def test_a_large_entry_goes_alone_and_the_batch_stays_a_prefix(tmp_path):
    """Past the first entry the batch stops at BATCH_BYTES, so a checkpoint-sized
    file never rides with a backlog behind it, and nothing later in the order
    jumps ahead of what was left (exports before the records announcing them)."""
    remote = _container(
        tmp_path,
        **{
            "models/model_epoch_0001.onnx": "x" * 100,
            "models/model_epoch_0002.onnx": "x" * 10,
            "records/gen_000001.json": "{}",
        },
    )
    local = tmp_path / "local"
    local.mkdir()
    first = _pull_outputs(tmp_path, remote, local, batch_bytes=50)
    assert first.pulled == ["models/model_epoch_0001.onnx"] and first.remaining == 2
    second = _pull_outputs(tmp_path, remote, local, batch_bytes=50)
    assert second.pulled == ["models/model_epoch_0002.onnx", "records/gen_000001.json"]
    assert second.remaining == 0


def test_a_state_pair_is_taken_whole_and_only_once_its_cursor_is_there(tmp_path):
    remote = _container(
        tmp_path,
        **{
            "state/gen_000003/model.pt": "w3",
            "state/gen_000003/train_state.json": '{"rows_trained": 3}',
            "state/gen_000004/model.pt": "w4",  # its cursor is still being written
            "scheduler_state.json": '{"gate": null}',
        },
    )
    local = tmp_path / "local"
    local.mkdir()
    result = _pull_outputs(tmp_path, remote, local, record_files=("scheduler_state.json",))
    assert sorted(result.pulled) == [
        "scheduler_state.json",
        "state/gen_000003/model.pt",
        "state/gen_000003/train_state.json",
    ]
    assert result.remaining == 0  # gen 4 is not a pair yet
    assert not (remote / "state/gen_000003").exists()  # moved, whole
    assert (remote / "state/gen_000004/model.pt").exists()
    assert (remote / "scheduler_state.json").exists()  # a record: copied, never removed


def test_a_file_still_being_written_is_left_in_the_container(tmp_path):
    """A trainer writes each export and record beside its final name with a
    `.tmp` suffix; a pull that moved one would tear it and pull the file out
    from under the trainer's rename."""
    remote = _container(
        tmp_path,
        **{
            "models/model_epoch_0005.onnx.tmp": "half",
            "records/gen_000005.json.tmp": "{",
            "records/gen_000004.json": "{}",
        },
    )
    local = tmp_path / "local"
    local.mkdir()
    result = _pull_outputs(tmp_path, remote, local)
    assert result.pulled == ["records/gen_000004.json"] and result.remaining == 0
    assert (remote / "models/model_epoch_0005.onnx.tmp").exists()
    assert (remote / "records/gen_000005.json.tmp").exists()


def test_a_sweep_names_a_top_level_pair_by_its_tag_relative_path(tmp_path):
    """docker cp names a top-level directory's members from the directory
    itself, so the names come back as "state/...", which is how the caller
    recognizes the pair to install."""
    import io
    import tarfile

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tar:
        for name, data in (("model.pt", b"w9"), ("train_state.json", b"{}")):
            info = tarfile.TarInfo(f"state/gen_000009/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    class _Stopped:
        def copy_from_container(self, container, path, dest):
            if not path.endswith("/state"):
                return False
            dest.write_bytes(payload.getvalue())
            return True

    local = tmp_path / "local"
    local.mkdir()
    names = sweep_stopped(
        _Stopped(), "c", remote_root="/tag", local_root=local, data_dirs=["models"],
        pair_dirs={"state": "train_state.json"},
    )  # fmt: skip
    assert sorted(names) == ["state/gen_000009/model.pt", "state/gen_000009/train_state.json"]
    assert (local / "state/gen_000009/model.pt").read_bytes() == b"w9"


def test_a_pulls_time_limit_scales_with_its_bytes(tmp_path, monkeypatch):
    """A lone checkpoint over a home link takes minutes, more than the fixed
    floor: the in-container tar and the ssh read both get a limit from the
    batch's size, the read a margin longer so the tar is the one that stops."""
    assert ssh_transfer.transfer_seconds(0) == ssh_transfer.COLLECT_TIMEOUT_SECONDS
    big = 300 * ssh_transfer.MIN_RATE
    assert ssh_transfer.transfer_seconds(big) == 300

    monkeypatch.setattr(ssh_transfer, "MIN_RATE", 1)  # so 100 bytes take 100 s
    remote = _container(tmp_path, **{"models/model_epoch_0001.onnx": "x" * 100})
    local = tmp_path / "local"
    local.mkdir()
    reads = []

    class _Timed(_FakeMachine):
        def read_from_container(self, container, command, timeout=None):
            reads.append((command[2], timeout))
            return super().read_from_container(container, command, timeout)

    pull_results(
        _Timed(tmp_path), "c", remote_root=str(remote), local_root=local,
        data_dirs=["models"],
    )  # fmt: skip
    collect, timeout = next((c, t) for c, t in reads if " tar " in c)
    assert "timeout 100 tar" in collect
    assert timeout == 100 + ssh_transfer.READ_MARGIN_SECONDS

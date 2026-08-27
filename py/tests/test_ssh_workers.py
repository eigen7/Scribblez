"""Unit tests for the ssh worker kind's pure logic: probe classification, the
display-state mapping, ssh/env-file construction, and worker env assembly."""

import subprocess

import pytest
from cloud import ssh_machine
from cloud.credentials import (
    CloudCredentials,
    R2Credentials,
    RegistryConfig,
    RunpodCredentials,
)
from cloud.ssh_machine import (
    SshMachine,
    SshMachineError,
    classify_probe,
    env_file,
)
from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard.tasks import TaskRecord, WorkerRecord
from scribblez.dashboard.workers import _container_name, _next_worker_id, _ssh_state
from scripts.cloud_fleet import bundle_worker_env


def test_classify_probe():
    assert classify_probe(0, "running\n", "") == "running"
    assert classify_probe(0, "paused\n", "") == "paused"
    # created / exited / dead all mean nothing of it is executing.
    assert classify_probe(0, "exited\n", "") == "stopped"
    assert classify_probe(0, "created\n", "") == "stopped"
    assert classify_probe(1, "", "Error: No such object: scz-x") == "missing"
    # Docker 29 lowercased the phrasing ("error: no such object"); the match is
    # case-insensitive so a missing container is not misread as unreachable.
    assert classify_probe(1, "", "error: no such object: scz-x") == "missing"
    # ssh's own failure (exit 255) is unreachable even if the message happens
    # to mention objects; so is any docker failure that is not a definite
    # missing container (e.g. the daemon being down).
    assert classify_probe(255, "", "No such object") == "unreachable"
    assert classify_probe(255, "", "Connection timed out") == "unreachable"
    assert classify_probe(1, "", "Cannot connect to the Docker daemon") == "unreachable"


def test_ssh_state_mapping():
    assert _ssh_state("running", "running", gated=True) == "waiting"
    # A gate parks an ssh container by pausing it; that is still "waiting".
    assert _ssh_state("running", "paused", gated=True) == "waiting"
    assert _ssh_state("running", "unknown", gated=False) == "checking"
    assert _ssh_state("paused", "paused", gated=False) == "stopping"
    assert _ssh_state("running", "running", gated=False) == "running"
    # A container that ran and died is the alarm `exited` carries; one that
    # does not exist yet is a slot on its way up, and saying "exited" for the
    # minutes an image pull takes reports a failure that has not happened.
    assert _ssh_state("running", "stopped", gated=False) == "exited"
    assert _ssh_state("running", "missing", gated=False) == "starting"
    assert _ssh_state("running", "unreachable", gated=False) == "unreachable"
    assert _ssh_state("paused", "running", gated=False) == "stopping"
    assert _ssh_state("paused", "stopped", gated=False) == "paused"
    assert _ssh_state("paused", "missing", gated=False) == "paused"
    assert _ssh_state("paused", "unreachable", gated=False) == "unreachable"


def test_ssh_argv_quotes_remote_command():
    argv = SshMachine("dev@laptop2").argv(["docker", "inspect", "-f", "{{.State.Running}}", "c"])
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert argv[-2] == "dev@laptop2"
    # The remote side runs a shell: the whole command is one argument with
    # shell metacharacters quoted.
    assert argv[-1] == "docker inspect -f '{{.State.Running}}' c"


def test_env_file_format():
    assert env_file({"A": "1", "B": "x y"}) == "A=1\nB=x y\n"
    with pytest.raises(AssertionError):
        env_file({"A": "1\n2"})


def _task(*worker_ids: str) -> TaskRecord:
    return TaskRecord(
        workload="kill_test",
        tag="t",
        params={},
        created_at=0.0,
        workers=[
            WorkerRecord(worker_id=w, role="generate", kind="ssh", desired_state="running")
            for w in worker_ids
        ],
    )


def test_next_worker_id_skips_taken():
    assert _next_worker_id(_task(), "ssh") == "ssh-0"
    assert _next_worker_id(_task("ssh-0", "local-0"), "ssh") == "ssh-1"
    assert _next_worker_id(_task("ssh-0", "ssh-2"), "ssh") == "ssh-1"


def test_container_name_qualified_by_workload_and_tag():
    spec = workloads.get("kill_test")
    assert _container_name(spec, "run1", "ssh-0") == "scz-kill_test-run1-ssh-0"


def test_bundle_worker_env_composition():
    creds = CloudCredentials(
        runpod=RunpodCredentials(api_key="k", container_registry_auth_id="a"),
        registry=RegistryConfig(worker_image="docker.io/u/scribblez-worker"),
        r2=R2Credentials(account_id="acct", access_key_id="ak", secret_access_key="sk", bucket="b"),
    )
    spec = workloads.get("kill_test")
    params = params_mod.validate(spec.params_cls, {})
    env = bundle_worker_env(
        creds, spec, "run1", params, role="generate", bundle_id="bid", worker_id="ssh-0",
        kind="ssh",
    )  # fmt: skip
    assert env["R2_ACCESS_KEY_ID"] == "ak"
    assert env["SCZ_WORKLOAD"] == "kill_test"
    assert env["SCZ_ROLE"] == "generate"
    assert env["SCZ_TAG"] == "run1"
    assert env["SCZ_BUNDLE"] == "bid"
    assert env["SCZ_WORKER_ID"] == "ssh-0"
    assert env["SCZ_WORKER_KIND"] == "ssh"


class _FakeCompleted:
    def __init__(self, returncode, stdout=b"", stderr=b""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _copy(monkeypatch, tmp_path, completed) -> tuple[bool, bytes]:
    """SshMachine.copy_from_container against a canned `docker cp` result.
    Returns what it reported and what it spooled."""

    def fake_run(argv, stdout=None, **kwargs):
        if completed.returncode == 0:
            stdout.write(completed.stdout)
        return completed

    monkeypatch.setattr(ssh_machine.subprocess, "run", fake_run)
    dest = tmp_path / "sweep.tar"
    wrote = SshMachine("user@h").copy_from_container("c", "/tag/data/staging", dest)
    return wrote, dest.read_bytes()


def test_copy_from_container_spools_the_stream(monkeypatch, tmp_path):
    wrote, spooled = _copy(monkeypatch, tmp_path, _FakeCompleted(0, stdout=b"tarbytes"))
    assert (wrote, spooled) == (True, b"tarbytes")


def test_an_empty_stream_reports_nothing_written(monkeypatch, tmp_path):
    assert _copy(monkeypatch, tmp_path, _FakeCompleted(0)) == (False, b"")


def test_a_path_the_container_never_made_is_empty_not_an_error(monkeypatch, tmp_path):
    """Verbatim from Docker 29 on the fleet. A worker that died before
    producing anything never made its output directory, and the sweep that
    precedes replacing it must not read that as a failure -- doing so would
    block the replacement forever."""
    missing = (
        b"Error response from daemon: Could not find the file /tag/data/staging in container c\n"
    )
    assert _copy(monkeypatch, tmp_path, _FakeCompleted(1, stderr=missing))[0] is False
    older = b"Error: No such container:path: c:/tag/data/staging\n"  # before Docker reworded it
    assert _copy(monkeypatch, tmp_path, _FakeCompleted(1, stderr=older))[0] is False


def test_any_other_copy_failure_raises(monkeypatch, tmp_path):
    """The caller sweeps a container in order to destroy it, so a read that
    failed for an unrecognised reason has to stop that."""
    for stderr in (b"Error response from daemon: No such container: c\n", b"ssh: refused\n"):
        with pytest.raises(SshMachineError):
            _copy(monkeypatch, tmp_path, _FakeCompleted(1, stderr=stderr))


def _write(monkeypatch, tmp_path, completed):
    """SshMachine.write_to_container against a canned `docker exec` result."""

    def fake_run(argv, **kwargs):
        if completed is None:
            raise subprocess.TimeoutExpired(argv, 1)
        kwargs["stdin"].read()  # the real call streams the file
        return completed

    monkeypatch.setattr(ssh_machine.subprocess, "run", fake_run)
    src = tmp_path / "model_epoch_0010.onnx"
    src.write_bytes(b"weights")
    SshMachine("user@h").write_to_container("c", ["sh", "-c", "cat > x"], src)


def test_write_to_container_streams_the_file(monkeypatch, tmp_path):
    _write(monkeypatch, tmp_path, _FakeCompleted(0))


def test_a_failed_write_raises(monkeypatch, tmp_path):
    """A push that did not land must not read as one that did: the controller
    would then wait forever for a match its worker was never given."""
    with pytest.raises(SshMachineError):
        _write(monkeypatch, tmp_path, _FakeCompleted(1, stderr=b"no such container: c\n"))


def test_a_write_that_times_out_raises(monkeypatch, tmp_path):
    with pytest.raises(SshMachineError):
        _write(monkeypatch, tmp_path, None)


def test_a_paused_container_between_gates_is_not_reported_as_exited():
    """A gate parks an ssh container by pausing it and releases it a pass
    before anything unpauses it. In that window the slot is healthy and about
    to resume -- calling it "exited", beside an empty exit reason, sent an
    operator looking for a crash that had not happened."""
    assert _ssh_state("running", "paused", gated=True) == "waiting"
    assert _ssh_state("running", "paused", gated=False) == "starting"
    assert _ssh_state("running", "stopped", gated=False) == "exited"  # this one really did

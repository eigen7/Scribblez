"""Unit tests for the ssh worker kind's pure logic: probe classification, the
display-state mapping, ssh/env-file construction, and worker env assembly."""

import pytest
from cloud.credentials import (
    CloudCredentials,
    R2Credentials,
    RegistryConfig,
    RunpodCredentials,
)
from cloud.ssh_machine import SshMachine, classify_probe, env_file
from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard.tasks import TaskRecord, WorkerRecord
from scribblez.dashboard.workers import _container_name, _next_worker_id, _ssh_state
from scripts.cloud_fleet import bundle_worker_env


def test_classify_probe():
    assert classify_probe(0, "true\n", "") == "running"
    assert classify_probe(0, "false\n", "") == "stopped"
    assert classify_probe(1, "", "Error: No such object: scz-x") == "missing"
    # ssh's own failure (exit 255) is unreachable even if the message happens
    # to mention objects; so is any docker failure that is not a definite
    # missing container (e.g. the daemon being down).
    assert classify_probe(255, "", "No such object") == "unreachable"
    assert classify_probe(255, "", "Connection timed out") == "unreachable"
    assert classify_probe(1, "", "Cannot connect to the Docker daemon") == "unreachable"


def test_ssh_state_mapping():
    assert _ssh_state("running", "running", gated=True) == "waiting"
    assert _ssh_state("running", "running", gated=False) == "running"
    assert _ssh_state("running", "stopped", gated=False) == "exited"
    assert _ssh_state("running", "missing", gated=False) == "exited"
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
        creds, spec, "run1", params, role="generate", bundle_id="bid", worker_id="ssh-0"
    )
    assert env["R2_ACCESS_KEY_ID"] == "ak"
    assert env["SCZ_WORKLOAD"] == "kill_test"
    assert env["SCZ_ROLE"] == "generate"
    assert env["SCZ_TAG"] == "run1"
    assert env["SCZ_BUNDLE"] == "bid"
    assert env["SCZ_WORKER_ID"] == "ssh-0"

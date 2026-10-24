"""Tests for the worker entrypoint's startup guards (py/cloud/worker_entrypoint.py)."""

from types import SimpleNamespace

import pytest
from cloud import worker_entrypoint
from scribblez import workloads
from scribblez.workloads.worker import WorkerFatal


def test_params_understood_accepts_the_launcher_environment():
    spec = workloads.get("kill_test")
    env = spec.worker_env("t", spec.params_cls(), "generate") | {"SCZ_WORKER_ID": "w0"}
    worker_entrypoint.check_params_understood(spec, env)


def test_params_understood_refuses_a_parameter_the_bundle_predates():
    spec = workloads.get("kill_test")
    env = {"SCZ_TAG": "t", "SCZ_FUTURE_KNOB": "1", "SCZ_BUNDLE_ID": "bid"}
    with pytest.raises(AssertionError, match="SCZ_FUTURE_KNOB"):
        worker_entrypoint.check_params_understood(spec, env)


def test_the_refusal_names_the_bundle_to_replace():
    spec = workloads.get("kill_test")
    env = {"SCZ_FUTURE_KNOB": "1", "SCZ_BUNDLE_ID": "bid"}
    with pytest.raises(AssertionError, match="bid"):
        worker_entrypoint.check_params_understood(spec, env)


def test_every_worker_env_var_is_documented():
    """The reserved names are the entrypoint's contract with its launchers; an
    undocumented one would surface as a stale-bundle failure at startup."""
    for name in worker_entrypoint.WORKER_ENV_VARS:
        assert name in worker_entrypoint.__doc__


def test_a_sigterm_ended_run_exits_interrupted_whatever_the_runner_returned(monkeypatch):
    """The dashboard reads exit 0 as the role's terminal condition reached (a
    finished slot); a run SIGTERM cut short, which runners drain and return 0
    from, must not read as that."""
    monkeypatch.setattr(worker_entrypoint, "_interrupted", False)
    with pytest.raises(worker_entrypoint.WorkerStopped):
        worker_entrypoint._on_sigterm(15, None)
    assert worker_entrypoint._interrupted
    monkeypatch.setattr(worker_entrypoint, "_interrupted", False)


def test_a_fatal_runner_exits_fatal_with_its_reason_as_the_last_line(monkeypatch, capsys):
    """A WorkerFatal from the runner (a trainer gone non-finite) exits with
    EXIT_FATAL, which the dashboard reads as "do not restart", and prints the
    reason last, where the dashboard reads it."""

    def runner(ctx):
        raise WorkerFatal("non-finite slot loss at batch 3")

    monkeypatch.setattr(worker_entrypoint, "_interrupted", False)
    monkeypatch.setattr(worker_entrypoint.signal, "signal", lambda *a: None)
    monkeypatch.setattr(worker_entrypoint, "make_sink", lambda *a: SimpleNamespace(push_json=print))
    spec = workloads.get("kill_test")
    # The role's runner raises; its deps hook does nothing.
    monkeypatch.setattr(
        worker_entrypoint.workloads,
        "resolve",
        lambda name: runner if name == spec.roles[0].runner else (lambda params: None),
    )
    for k, v in spec.worker_env("t", spec.params_cls(), spec.roles[0].name).items():
        monkeypatch.setenv(k, v)
    assert worker_entrypoint.main() == worker_entrypoint.EXIT_FATAL
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last == "fatal: non-finite slot loss at batch 3"

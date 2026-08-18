"""Tests for the worker entrypoint's startup guards (py/cloud/worker_entrypoint.py)."""

import pytest
from cloud import worker_entrypoint
from scribblez import workloads


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

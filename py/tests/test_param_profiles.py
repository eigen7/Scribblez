"""Parameter profiles (scribblez/params.py, WorkloadSpec.profiles): named value
sets layered between the dataclass defaults and what the operator sets.

Pins the resolution order at every consumer -- CLI flags, task creation -- the
declaration-time validation of a profile, and the provenance a task keeps.
"""

import argparse
from dataclasses import dataclass

import pytest
from scribblez import params as params_mod
from scribblez.dashboard import tasks
from scribblez.params import param
from scribblez.workloads import WorkloadSpec
from scribblez.workloads.position_eval import PROFILES, TRUNK_CONV, TRUNK_TRANSFORMER


@dataclass(frozen=True)
class DemoParams:
    depth: int = param(4, "an int")
    rate: float = param(0.5, "a float")
    arch: str = param("small", "a str", choices=("small", "big"))
    clip: bool = param(False, "a bool")


PROFILE_SET = {
    "big": {"arch": "big", "depth": 12, "clip": True},
    "small": {"arch": "small"},
}


def _spec(**kw) -> WorkloadSpec:
    return WorkloadSpec(name="demo", title="Demo", params_cls=DemoParams, roles=(), **kw)


def _parser(spec: WorkloadSpec) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    spec.add_cli_arguments(p)
    return p


def test_cli_layers_flags_over_profile_over_defaults():
    spec = _spec(profiles=PROFILE_SET, default_profile="big")
    p = _parser(spec)
    # nothing given: the default profile's values, defaults where it is silent
    assert spec.params_from_args(p.parse_args([])) == DemoParams(12, 0.5, "big", True)
    # a profile chosen, one of its values overridden, one it is silent on set
    got = spec.params_from_args(p.parse_args(["--profile", "small", "--depth", "2", "--rate", "9"]))
    assert got == DemoParams(2, 9.0, "small", False)
    # a bool the profile turns on can be turned back off from the flag
    assert spec.params_from_args(p.parse_args(["--no-clip"])).clip is False


def test_cli_without_profiles_keeps_the_dataclass_defaults():
    """A workload with no profiles gets no --profile flag, and unspecified flags
    (now SUPPRESSed rather than defaulted) still resolve to the defaults."""
    spec = _spec()
    p = _parser(spec)
    assert spec.params_from_args(p.parse_args([])) == DemoParams()
    assert spec.params_from_args(p.parse_args(["--depth", "7"])) == DemoParams(depth=7)
    with pytest.raises(SystemExit):
        p.parse_args(["--profile", "big"])


def test_cli_help_carries_the_defaults_and_profile_overrides():
    p = _parser(_spec(profiles=PROFILE_SET, default_profile="big"))
    text = p.format_help()
    assert "(default: 4; profile big=12)" in text  # depth: the base default and who overrides it
    assert "(default: 0.5)" in text  # rate: no profile touches it
    assert "--profile {big,small}" in text


def test_validate_layers_base_under_raw():
    got = params_mod.validate(DemoParams, {"rate": 2}, base={"depth": 9, "rate": 1.0})
    assert got == DemoParams(depth=9, rate=2.0)
    with pytest.raises(params_mod.ParamsError):  # a bad base value is still a bad value
        params_mod.validate(DemoParams, {}, base={"arch": "huge"})


def test_a_profile_must_validate_at_declaration():
    with pytest.raises(AssertionError, match="profile 'bad'"):
        _spec(profiles={"bad": {"arch": "huge"}}, default_profile="bad")
    with pytest.raises(AssertionError, match="default_profile"):
        _spec(profiles=PROFILE_SET, default_profile="nope")
    with pytest.raises(AssertionError, match="without profiles"):
        _spec(default_profile="big")


def test_resolve_and_diff():
    spec = _spec(profiles=PROFILE_SET, default_profile="big")
    name, params = spec.resolve_params(None, {"rate": 3})
    assert name == "big" and params == DemoParams(12, 3.0, "big", True)
    name, params = spec.resolve_params("small", {})
    assert name == "small" and params == DemoParams(arch="small")
    with pytest.raises(AssertionError, match="no profile"):
        spec.resolve_params("huge", {})
    assert spec.profile_defaults("big") == {"depth": 12, "rate": 0.5, "arch": "big", "clip": True}
    diff = spec.profile_diff("big", {"depth": 12, "rate": 3.0, "arch": "big", "clip": False})
    assert diff == [
        {"name": "rate", "profile": 0.5, "task": 3.0},
        {"name": "clip", "profile": True, "task": False},
    ]


def test_create_task_resolves_and_records_the_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "task_path", lambda spec, tag: tmp_path / f"{tag}.task.json")
    spec = _spec(profiles=PROFILE_SET, default_profile="big")
    task = tasks.create_task(spec, "t1", {"rate": 2})
    assert task.profile == "big" and task.params == {
        "depth": 12,
        "rate": 2.0,
        "arch": "big",
        "clip": True,
    }
    task = tasks.create_task(spec, "t2", {}, profile="small")
    assert task.profile == "small" and task.params["arch"] == "small" and task.params["depth"] == 4
    # the record round-trips, profile included
    assert tasks.load_task(spec, "t2").profile == "small"
    with pytest.raises(AssertionError, match="no profile"):
        tasks.create_task(spec, "t3", {}, profile="huge")


def test_position_eval_profiles_are_one_per_trunk():
    """The registered recipes: each sets its trunk, the transformer's turns on
    gradient clipping (the conv one trains as its runs always have), and the
    default is the transformer."""
    from scribblez import workloads

    spec = workloads.get("position_eval")
    assert (
        set(PROFILES) == {TRUNK_TRANSFORMER, TRUNK_CONV}
        and spec.default_profile == TRUNK_TRANSFORMER
    )
    tf = spec.profile_defaults(TRUNK_TRANSFORMER)
    cnn = spec.profile_defaults(TRUNK_CONV)
    assert tf["trunk"] == TRUNK_TRANSFORMER and cnn["trunk"] == TRUNK_CONV
    assert tf["grad_clip"] > 0 and cnn["grad_clip"] == 0
    assert tf["batch_size"] == cnn["batch_size"]  # a knob no profile names agrees

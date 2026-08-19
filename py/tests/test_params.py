"""Unit tests for the dataclass-driven parameter schemas."""

import argparse
from dataclasses import dataclass

import pytest
from scribblez import params as params_mod
from scribblez.params import param


@dataclass(frozen=True)
class DemoParams:
    count: int = param(3, "an int")
    rate: float = param(0.5, "a float")
    name: str = param("abc", "a str")
    flag: bool = param(False, "a bool")


def test_schema_kinds():
    kinds = {f.name: f.kind for f in params_mod.schema(DemoParams)}
    assert kinds == {"count": "int", "rate": "float", "name": "str", "flag": "bool"}


def test_env_roundtrip():
    p = DemoParams(count=7, rate=1.25, name="x y", flag=True)
    env = params_mod.to_env(p)
    assert env == {"SCZ_COUNT": "7", "SCZ_RATE": "1.25", "SCZ_NAME": "x y", "SCZ_FLAG": "1"}
    assert params_mod.from_env(DemoParams, env) == p


def test_env_defaults_for_absent_vars():
    assert params_mod.from_env(DemoParams, {}) == DemoParams()


def test_args_roundtrip():
    parser = argparse.ArgumentParser()
    params_mod.add_arguments(parser, DemoParams)
    args = parser.parse_args(["--count", "9", "--rate", "2.5", "--name", "zz", "--flag"])
    assert params_mod.from_args(DemoParams, args) == DemoParams(9, 2.5, "zz", True)


def test_validate_coerces_and_reports():
    p = params_mod.validate(DemoParams, {"count": 2, "rate": 3, "name": "n", "flag": True})
    assert p == DemoParams(2, 3.0, "n", True)
    with pytest.raises(params_mod.ParamsError) as e:
        params_mod.validate(
            DemoParams, {"count": "x", "rate": "y", "name": 5, "flag": 1, "bogus": 0}
        )
    msg = "; ".join(str(a) for a in e.value.args)
    assert "unknown parameter 'bogus'" in msg
    assert "count" in msg and "rate" in msg and "name" in msg and "flag" in msg


def test_registry_params_all_schema_valid():
    """Every registered workload's params dataclass passes schema derivation."""
    from scribblez import workloads

    for spec in workloads.WORKLOADS.values():
        fields = params_mod.schema(spec.params_cls)
        assert fields, spec.name
        # The env encoding round-trips the defaults.
        p = spec.params_cls()
        assert params_mod.from_env(spec.params_cls, params_mod.to_env(p)) == p


def test_unknown_env_flags_variables_outside_the_schema():
    env = {"SCZ_COUNT": "1", "SCZ_ELSEWHERE": "x", "SCZ_TAG": "t", "PATH": "/bin"}
    assert params_mod.unknown_env(DemoParams, env) == ["SCZ_ELSEWHERE", "SCZ_TAG"]
    assert params_mod.unknown_env(DemoParams, env, allowed=("SCZ_TAG",)) == ["SCZ_ELSEWHERE"]


def test_unknown_env_accepts_a_fully_known_environment():
    env = params_mod.to_env(DemoParams()) | {"SCZ_TAG": "t"}
    assert params_mod.unknown_env(DemoParams, env, allowed=("SCZ_TAG",)) == []


@dataclass(frozen=True)
class ClosedParams:
    """A field whose value set is closed, alongside an open one."""

    mode: str = param("a", "a closed str", choices=("a", "b"))
    rate: float = param(1.0, "an open float")


def test_schema_carries_choices():
    fields = {f.name: f for f in params_mod.schema(ClosedParams)}
    assert fields["mode"].choices == ("a", "b")
    assert fields["rate"].choices is None
    # public_schema is what the dashboard form reads.
    public = {f["name"]: f["choices"] for f in params_mod.public_schema(ClosedParams)}
    assert public["mode"] == ("a", "b")
    assert public["rate"] is None


def test_validate_rejects_a_value_outside_the_set():
    """A misspelt choice is refused where it is entered, rather than passing
    validation and failing later in whatever consumes it."""
    with pytest.raises(params_mod.ParamsError) as e:
        params_mod.validate(ClosedParams, {"mode": "c"})
    assert "expected one of ['a', 'b']" in e.value.args[0]


def test_validate_accepts_a_value_in_the_set():
    assert params_mod.validate(ClosedParams, {"mode": "b", "rate": 2.0}).mode == "b"


def test_a_closed_field_still_type_checks_first():
    with pytest.raises(params_mod.ParamsError) as e:
        params_mod.validate(ClosedParams, {"mode": 7})
    assert "expected a string" in e.value.args[0]


def test_argparse_enforces_the_set():
    parser = argparse.ArgumentParser()
    params_mod.add_arguments(parser, ClosedParams)
    assert parser.parse_args(["--mode", "b"]).mode == "b"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "c"])


def test_a_default_outside_its_own_choices_is_a_declaration_error():
    @dataclass(frozen=True)
    class Bad:
        mode: str = param("z", "bad default", choices=("a",))

    with pytest.raises(AssertionError, match="not among its choices"):
        params_mod.schema(Bad)

"""Dataclass-driven parameter schemas.

A workload's parameters are declared once, as a frozen dataclass whose fields
carry help text via `param(...)`, and every consumer derives its interface from
that single declaration:

  - argparse flags for CLIs             (add_arguments / from_args)
  - environment variables for workers   (to_env / from_env)
  - a JSON schema for the dashboard's   (public_schema)
    auto-generated config form
  - validation of user-supplied values  (validate)

Supported field kinds: int, float, str, and bool (bool fields must default to
False so the argparse mapping to store_true flags stays faithful).

A field may also declare `choices`, closing its value set and naming, per
choice, the defaults that choice implies for other fields. Validation then
rejects anything outside the set -- so a bad value is refused where it is
entered rather than surfacing later as a worker crash -- and the dashboard
form renders a selector that re-seeds the dependent fields when the choice
changes.
"""

import dataclasses
import os

ENV_PREFIX = "SCZ_"

_KINDS = ("int", "float", "str", "bool")


def param(default, help: str, choices: dict[str, dict] | None = None) -> dataclasses.Field:
    """Declare one parameter field: `x: int = param(200, "what x means")`.

    `choices` closes the value set, mapping each allowed value to the defaults
    it implies for other fields (an empty dict when it implies none)."""
    return dataclasses.field(default=default, metadata={"help": help, "choices": choices})


@dataclasses.dataclass(frozen=True)
class ParamField:
    name: str
    kind: str  # one of _KINDS
    default: object
    help: str
    choices: dict[str, dict] | None = None  # allowed value -> defaults it implies elsewhere


class ParamsError(Exception):
    """User-supplied parameter values failed validation; args are the messages."""


def schema(params_cls: type) -> list[ParamField]:
    out = []
    for f in dataclasses.fields(params_cls):
        kind = f.type if isinstance(f.type, str) else f.type.__name__
        assert kind in _KINDS, f"{params_cls.__name__}.{f.name}: unsupported kind {kind}"
        if kind == "bool":
            assert f.default is False, f"{f.name}: bool params must default to False"
        choices = f.metadata.get("choices")
        if choices is not None:
            assert kind != "bool", f"{f.name}: a bool field is already a closed set"
            assert f.default in choices, f"{f.name}: default {f.default!r} is not among its choices"
        out.append(ParamField(f.name, kind, f.default, f.metadata.get("help", ""), choices))
    return out


def public_schema(params_cls: type) -> list[dict]:
    """The schema as JSON-ready dicts (drives the dashboard's config form)."""
    return [dataclasses.asdict(f) for f in schema(params_cls)]


def add_arguments(parser, params_cls: type):
    """Register one argparse option per field (--snake-case-name)."""
    for f in schema(params_cls):
        flag = "--" + f.name.replace("_", "-")
        if f.kind == "bool":
            parser.add_argument(flag, action="store_true", help=f.help)
        else:
            py_type = {"int": int, "float": float, "str": str}[f.kind]
            choices = sorted(f.choices) if f.choices else None
            parser.add_argument(flag, type=py_type, default=f.default, help=f.help, choices=choices)


def from_args(params_cls: type, args):
    return params_cls(**{f.name: getattr(args, f.name) for f in schema(params_cls)})


def to_env(params) -> dict[str, str]:
    """The SCZ_* environment variables encoding `params` for a worker."""
    out = {}
    for f in schema(type(params)):
        value = getattr(params, f.name)
        out[ENV_PREFIX + f.name.upper()] = (
            ("1" if value else "0") if f.kind == "bool" else str(value)
        )
    return out


def unknown_env(params_cls: type, env=os.environ, *, allowed=()) -> list[str]:
    """The SCZ_* variables in `env` naming neither a parameter of `params_cls`
    nor one of `allowed` (the launcher's own worker-level knobs).

    from_env reads the schema, not the environment, so a parameter the schema
    predates is silently ignored rather than misapplied -- which lets a worker
    on stale code produce data that looks valid and is not. Callers that know
    the full set of variables they were launched with use this to refuse
    instead.
    """
    known = {ENV_PREFIX + f.name.upper() for f in schema(params_cls)} | set(allowed)
    return sorted(k for k in env if k.startswith(ENV_PREFIX) and k not in known)


def from_env(params_cls: type, env=os.environ):
    """Build params from SCZ_* variables; absent variables keep their defaults."""
    kwargs = {}
    for f in schema(params_cls):
        raw = env.get(ENV_PREFIX + f.name.upper())
        if raw is None:
            continue
        parse = {"int": int, "float": float, "str": str, "bool": lambda r: r == "1"}[f.kind]
        kwargs[f.name] = parse(raw)
    return params_cls(**kwargs)


def _coerce(f: ParamField, value):
    """Coerce a JSON-ish value to the field's kind, raising ValueError on mismatch."""
    if f.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{f.name}: expected a boolean, got {value!r}")
        return value
    if f.kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"{f.name}: expected a string, got {value!r}")
        return value
    if f.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{f.name}: expected a number, got {value!r}")
        return float(value)
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{f.name}: expected an integer, got {value!r}") from None
    return value


def _check_choice(f: ParamField, value):
    """Reject a value outside a closed field's set, naming what was allowed."""
    if f.choices is not None and value not in f.choices:
        raise ValueError(f"{f.name}: expected one of {sorted(f.choices)}, got {value!r}")
    return value


def validate(params_cls: type, raw: dict):
    """Build params from a JSON-ish dict, raising ParamsError naming every
    unknown field and type mismatch; absent fields keep their defaults."""
    fields = {f.name: f for f in schema(params_cls)}
    errors = [f"unknown parameter '{k}'" for k in raw if k not in fields]
    kwargs = {}
    for name, f in fields.items():
        if name not in raw:
            continue
        try:
            kwargs[name] = _check_choice(f, _coerce(f, raw[name]))
        except ValueError as e:
            errors.append(str(e))
    if errors:
        raise ParamsError(*errors)
    return params_cls(**kwargs)

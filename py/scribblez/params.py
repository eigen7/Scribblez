"""Dataclass-driven parameter schemas.

A workload's parameters are declared once, as a frozen dataclass whose fields
carry help text via `param(...)`, and every consumer derives its interface from
that single declaration:

  - argparse flags for CLIs             (add_arguments / from_args)
  - environment variables for workers   (to_env / from_env)
  - a JSON schema for the dashboard's   (public_schema)
    auto-generated config form
  - validation of user-supplied values  (validate)

Supported field kinds: int, float, str, and bool. A bool field maps to an
argparse BooleanOptionalAction, so both `--flag` and `--no-flag` exist whatever
its default.

A field may declare `choices`, closing its value set. Validation and argparse
both enforce the set, so a bad value is refused where it is entered rather than
crashing a worker later, and the dashboard form renders a selector.

A workload may also define *profiles* (WorkloadSpec.profiles): named partial
value sets, such as a per-trunk training recipe, that the new-tag form and the
CLI's --profile start from. Precedence, lowest first: dataclass defaults, the
profile's values, values the operator set explicitly. A tag freezes the result
and records the profile name. Profiles are resolved here rather than in the
form so the CLI and dashboard cannot drift apart.
"""

import argparse
import dataclasses
import os

ENV_PREFIX = "SCZ_"

_KINDS = ("int", "float", "str", "bool")


def param(default, help: str, choices=None) -> dataclasses.Field:
    """Declare one parameter field: `x: int = param(200, "what x means")`.

    `choices`, when given, is the closed set of values the field accepts."""
    return dataclasses.field(
        default=default,
        metadata={"help": help, "choices": tuple(choices) if choices is not None else None},
    )


@dataclasses.dataclass(frozen=True)
class ParamField:
    name: str
    kind: str  # one of _KINDS
    default: object
    help: str
    choices: tuple | None = None  # the closed set of accepted values, if the field has one


class ParamsError(Exception):
    """User-supplied parameter values failed validation; args are the messages."""


def schema(params_cls: type) -> list[ParamField]:
    out = []
    for f in dataclasses.fields(params_cls):
        kind = f.type if isinstance(f.type, str) else f.type.__name__
        assert kind in _KINDS, f"{params_cls.__name__}.{f.name}: unsupported kind {kind}"
        choices = f.metadata.get("choices")
        if choices is not None:
            assert kind != "bool", f"{f.name}: a bool field is already a closed set"
            assert f.default in choices, f"{f.name}: default {f.default!r} is not among its choices"
            assert len(set(choices)) == len(choices), f"{f.name}: duplicate choices"
        out.append(ParamField(f.name, kind, f.default, f.metadata.get("help", ""), choices))
    return out


def public_schema(params_cls: type) -> list[dict]:
    """The schema as JSON-ready dicts (drives the dashboard's config form)."""
    return [dataclasses.asdict(f) for f in schema(params_cls)]


def _flag_help(f: ParamField, profiles: dict | None) -> str:
    """Help text naming the dataclass default and each profile's override,
    since the flag itself carries no default (see add_arguments)."""
    overrides = [
        f"{name}={values[f.name]}" for name, values in (profiles or {}).items() if f.name in values
    ]
    note = f"; profile {', '.join(overrides)}" if overrides else ""
    return f"{f.help} (default: {f.default}{note})"


def add_arguments(
    parser, params_cls: type, profiles: dict | None = None, default_profile: str = ""
):
    """Register one --kebab-case option per field, plus `--profile` when the
    workload has profiles.

    The flags have no argparse default (SUPPRESS) so that from_args can tell a
    flag the user typed from one left unset, and layer only typed flags over
    the profile."""
    if profiles:
        parser.add_argument(
            "--profile",
            choices=list(profiles),
            default=default_profile,
            help="parameter profile: named defaults the flags below override",
        )
    for f in schema(params_cls):
        flag = "--" + f.name.replace("_", "-")
        help_text = _flag_help(f, profiles)
        if f.kind == "bool":
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
                help=help_text,
            )
        else:
            py_type = {"int": int, "float": float, "str": str}[f.kind]
            choices = list(f.choices) if f.choices else None
            parser.add_argument(
                flag, type=py_type, default=argparse.SUPPRESS, help=help_text, choices=choices
            )


def from_args(params_cls: type, args, profiles: dict | None = None):
    """Params from parsed args: the dataclass defaults, under the chosen
    profile's values (args.profile, when add_arguments registered `profiles`),
    under every flag actually given."""
    chosen = getattr(args, "profile", "") if profiles else ""
    values = dict(profiles[chosen]) if chosen else {}
    for f in schema(params_cls):
        if hasattr(args, f.name):
            values[f.name] = getattr(args, f.name)
    return params_cls(**values)


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

    from_env silently ignores variables its schema does not know, so a worker
    running stale code would produce plausible-looking but wrong data. Callers
    use this to refuse to start instead.
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
        raise ValueError(f"{f.name}: expected one of {list(f.choices)}, got {value!r}")
    return value


def validate(params_cls: type, raw: dict, base: dict | None = None):
    """Build params from a JSON-ish dict, raising ParamsError naming every
    unknown field and type mismatch. A field `raw` omits takes `base`'s value
    (a profile's), and one both omit keeps its default."""
    fields = {f.name: f for f in schema(params_cls)}
    merged = {**(base or {}), **raw}
    errors = [f"unknown parameter '{k}'" for k in merged if k not in fields]
    kwargs = {}
    for name, f in fields.items():
        if name not in merged:
            continue
        try:
            kwargs[name] = _check_choice(f, _coerce(f, merged[name]))
        except ValueError as e:
            errors.append(str(e))
    if errors:
        raise ParamsError(*errors)
    return params_cls(**kwargs)

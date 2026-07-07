"""Dataclass-driven parameter schemas.

A workload's parameters are declared once, as a frozen dataclass whose fields
carry help text via `param(...)`, and every consumer derives its interface from
that single declaration:

  - argparse flags for CLIs             (add_arguments / from_args)
  - environment variables for workers   (to_env / from_env)
  - a JSON schema for the dashboard's   (public_schema)
    auto-generated config form
  - validation of user-supplied values  (validate)

Supported field kinds: int and bool (bool fields must default to False so the
argparse mapping to store_true flags stays faithful).
"""

import dataclasses
import os

ENV_PREFIX = "SCZ_"


def param(default, help: str) -> dataclasses.Field:
    """Declare one parameter field: `x: int = param(200, "what x means")`."""
    return dataclasses.field(default=default, metadata={"help": help})


@dataclasses.dataclass(frozen=True)
class ParamField:
    name: str
    kind: str  # "int" | "bool"
    default: object
    help: str


class ParamsError(Exception):
    """User-supplied parameter values failed validation; args are the messages."""


def schema(params_cls: type) -> list[ParamField]:
    out = []
    for f in dataclasses.fields(params_cls):
        kind = f.type if isinstance(f.type, str) else f.type.__name__
        assert kind in ("int", "bool"), f"{params_cls.__name__}.{f.name}: unsupported kind {kind}"
        if kind == "bool":
            assert f.default is False, f"{f.name}: bool params must default to False"
        out.append(ParamField(f.name, kind, f.default, f.metadata.get("help", "")))
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
            parser.add_argument(flag, type=int, default=f.default, help=f.help)


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


def from_env(params_cls: type, env=os.environ):
    """Build params from SCZ_* variables; absent variables keep their defaults."""
    kwargs = {}
    for f in schema(params_cls):
        raw = env.get(ENV_PREFIX + f.name.upper())
        if raw is None:
            continue
        kwargs[f.name] = raw == "1" if f.kind == "bool" else int(raw)
    return params_cls(**kwargs)


def validate(params_cls: type, raw: dict):
    """Build params from a JSON-ish dict, raising ParamsError naming every
    unknown field and type mismatch; absent fields keep their defaults."""
    fields = {f.name: f for f in schema(params_cls)}
    errors = [f"unknown parameter '{k}'" for k in raw if k not in fields]
    kwargs = {}
    for name, f in fields.items():
        if name not in raw:
            continue
        value = raw[name]
        if f.kind == "bool":
            if not isinstance(value, bool):
                errors.append(f"{name}: expected a boolean, got {value!r}")
                continue
        elif isinstance(value, bool) or not isinstance(value, int):
            try:
                value = int(value)
            except (TypeError, ValueError):
                errors.append(f"{name}: expected an integer, got {value!r}")
                continue
        kwargs[name] = value
    if errors:
        raise ParamsError(*errors)
    return params_cls(**kwargs)

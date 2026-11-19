"""The known positions (deep_bot/known_positions.py) as a role's out-of-tag
inputs (RoleSpec.inputs): a local slot reads them in place, a remote one from
the copies the controller stages under its tag."""

from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.deep_bot.known_positions import Position
from scribblez.workloads.base import WorkerContext, resolve_input

# Where a remote slot finds them, under its tag root.
INPUT_DIR = "inputs/known-positions"


def _rel(f: Path) -> str:
    return f"{INPUT_DIR}/{f.name}"


def inputs() -> dict[str, Path]:
    """Every file a reader of the positions needs, keyed as the slot finds it."""
    return {_rel(f): f for f in known_positions.files()}


def load(ctx: WorkerContext) -> list[Position]:
    """The positions where the slot reads them. The manifest comes first,
    since only it names the GCGs: a remote slot has no repo copy to list them
    from."""
    source = known_positions.DIRECTORY / known_positions.MANIFEST
    manifest = resolve_input(ctx, _rel(source), source)
    positions = known_positions.load(manifest.parent)
    for p in positions:
        resolve_input(ctx, _rel(p.gcg), known_positions.DIRECTORY / p.gcg.name)
    return positions

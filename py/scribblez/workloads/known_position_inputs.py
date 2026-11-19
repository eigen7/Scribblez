"""The known positions (deep_bot/known_positions.py) as a role's out-of-tag
inputs (RoleSpec.inputs): a local slot reads them in place, a remote one from
the copies the controller stages under its tag."""

from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.deep_bot.known_positions import Position
from scribblez.workloads.base import WorkerContext, resolve_input

# Where a remote slot finds them, under its tag root.
INPUT_DIR = "inputs/known-positions"


def inputs() -> dict[str, Path]:
    """Every file a reader of the positions needs, keyed as the slot finds it."""
    return {f"{INPUT_DIR}/{f.name}": f for f in known_positions.files()}


def load(ctx: WorkerContext) -> list[Position]:
    """The positions where the slot reads them."""
    staged = [resolve_input(ctx, rel, source) for rel, source in inputs().items()]
    return known_positions.load(staged[-1].parent)  # the manifest, last

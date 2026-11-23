"""A multi-model trainer's units (units.py) as one rolling checkpoint: the
models and optimizers by kind, each unit's config with the versions a seed
must match (units.check_seed), and the cursor, whose `<kind>_rows` fields are
the units' rows-clocks. A later tag seeds from the checkpoint as it would
from a standalone workload's (units.seed_part).
"""

from dataclasses import asdict

from torch import nn

from scribblez.deep_bot import units
from scribblez.generational import checkpoint
from scribblez.paths import TagPaths
from scribblez.train_common import timed_print


class OptimizerSet:
    """The units' optimizers as the rolling checkpoint saves and loads them."""

    def __init__(self, by_kind: dict):
        self._units = by_kind

    def state_dict(self) -> dict:
        return {kind: unit.optimizer.state_dict() for kind, unit in self._units.items()}

    def load_state_dict(self, saved: dict):
        for kind, unit in self._units.items():
            unit.optimizer.load_state_dict(saved[kind])


def models(by_kind: dict) -> nn.ModuleDict:
    """The units' models as the rolling checkpoint saves and loads them."""
    return nn.ModuleDict({kind: unit.model for kind, unit in by_kind.items()})


def unit_rows(by_kind: dict, state) -> dict[str, int]:
    """Each unit's rows-clock in the cursor `state`."""
    return {kind: getattr(state, f"{kind}_rows") for kind in by_kind}


def prefixed(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def checkpoint_config(params, by_kind: dict) -> dict:
    """The tag's params, and each unit's params with its versions."""
    return {
        "params": asdict(params),
        "units": {
            kind: asdict(unit.params) | units.unit_versions(kind, unit)
            for kind, unit in by_kind.items()
        },
    }


def save(paths: TagPaths, by_kind: dict, params, state):
    """The rolling checkpoint of the units' models and optimizers at the
    cursor `state`."""
    config = checkpoint_config(params, by_kind)
    checkpoint.save(paths, models(by_kind), OptimizerSet(by_kind), state, config)


def resume(by_kind: dict, paths: TagPaths, device, state_cls: type):
    """Load the rolling checkpoint into the units, if there is one, and
    return its cursor, a `state_cls`; else the units stay at their seeds."""
    ckpt, state = checkpoint.load(paths, device, state_cls)
    if ckpt is None:
        return state
    models(by_kind).load_state_dict(ckpt["model_state_dict"])
    OptimizerSet(by_kind).load_state_dict(ckpt["optimizer_state_dict"])
    timed_print(f"Resuming at generation {state.generation_index}")
    return state

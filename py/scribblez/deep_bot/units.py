"""A multi-model trainer's units (deep_bot_generation's two), built from the
tag's seeds (docs/plans/subtree_bot_generations.md, "Seeds and warm start").

A seed is a rolling checkpoint copied to seeds/<kind>.pt under the tag
(seeds.py): a standalone workload's, or a multi-model trainer's, which holds
every unit's model and config under its kind (unit_set.checkpoint_config).
The recorded config gives the unit's params, so the unit's architecture is
the seed's by construction, with the windows and the rows-clocks re-sized
from the tag's params: `<kind>_window`, `<kind>_turns_per_game`, `<kind>_lr`,
`games_per_generation`, `warmup_generations` and `averaging_generations`. The
models start from the seeds' weights with fresh optimizers.
"""

from dataclasses import fields, replace
from pathlib import Path

import torch

from scribblez.deep_bot import LEAF, LEAVE
from scribblez.deep_bot.seeds import seed_path
from scribblez.deep_bot_leaf_eval.unit import LeafEvalUnit
from scribblez.leave_inference.unit import LeaveInferenceUnit
from scribblez.paths import TagPaths
from scribblez.workloads.deep_bot_leaf_eval import DeepBotLeafEvalParams
from scribblez.workloads.leave_inference import LeaveInferenceParams

# A game's eligible turns, for sizing a rows-clock when a unit trains them all.
TURNS_PER_GAME = 20

UNIT_CLASSES = {LEAVE: LeaveInferenceUnit, LEAF: LeafEvalUnit}
PARAMS_CLASSES = {LEAVE: LeaveInferenceParams, LEAF: DeepBotLeafEvalParams}


def window(params, kind: str) -> int:
    """The generations `kind` trains over."""
    return getattr(params, f"{kind}_window")


def rows_per_generation(params, kind: str) -> int:
    """About how many rows `kind` trains each generation, the unit of its
    rows-clock: every game of its window, at its turns per game."""
    per_game = getattr(params, f"{kind}_turns_per_game") or TURNS_PER_GAME
    return params.games_per_generation * window(params, kind) * per_game


def unit_params(kind: str, seed_config: dict, params):
    """`kind`'s unit params: the seed's, with this tag's window, turns per
    game, learning rate and rows-clocks."""
    cls = PARAMS_CLASSES[kind]
    seed = cls(**{f.name: seed_config[f.name] for f in fields(cls) if f.name in seed_config})
    rows = rows_per_generation(params, kind)
    changes = {
        "window": window(params, kind),
        "lr": getattr(params, f"{kind}_lr"),
        "lr_warmup_rows": params.warmup_generations * rows,
        "averaging_window_rows": params.averaging_generations * rows,
        "max_rows": -1,
        "turns_per_game": getattr(params, f"{kind}_turns_per_game"),
    }
    return replace(seed, **changes)


def check_seed(unit, seed_config: dict, path: Path):
    """Refuse a seed written under other formats or leave values than this
    code's and this machine's."""
    current = unit.versions()
    for key in current:
        if seed_config.get(key) != current[key]:
            raise RuntimeError(
                f"{path} has {key}={seed_config.get(key)}, but this run has {current[key]}; "
                "seed the tag from a tag trained with this code"
            )


def seed_part(seed: dict, kind: str) -> tuple[dict, dict]:
    """`kind`'s config and weights in a loaded seed: all of a standalone
    workload's checkpoint, or the part of a multi-model trainer's under
    `kind`."""
    configs = seed["config"].get("units")
    if configs is None:
        return seed["config"], seed["model_state_dict"]
    prefix = f"{kind}."
    weights = {
        key.removeprefix(prefix): value
        for key, value in seed["model_state_dict"].items()
        if key.startswith(prefix)
    }
    return configs[kind], weights


def build_unit(kind: str, paths: TagPaths, params, device):
    """`kind`'s unit at its seed's weights, with a fresh optimizer."""
    path = seed_path(paths, kind)
    config, weights = seed_part(torch.load(path, map_location=device, weights_only=False), kind)
    unit = UNIT_CLASSES[kind](unit_params(kind, config, params), device)
    check_seed(unit, config, path)
    unit.model.load_state_dict(weights)
    return unit


def build_units(paths: TagPaths, params, device, kinds: tuple[str, ...]) -> dict:
    """The units of `kinds` at their seeds, by kind."""
    return {kind: build_unit(kind, paths, params, device) for kind in kinds}

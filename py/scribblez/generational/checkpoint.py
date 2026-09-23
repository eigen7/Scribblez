"""The rolling checkpoint: what a trainer resumes from after a restart.

A single `model.pt` under the tag holds the model and optimizer state plus the
training cursor (GenerationalState), so a restarted trainer continues exactly
where it left off. Every trainer uses it, generational or not.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields

import torch

from ..paths import TagPaths


@dataclass
class GenerationalState:
    """The training cursor persisted across restarts.

    rows_trained: cumulative rows trained; the rows-clock that the LR schedule
        runs on and the dashboard plots against.
    generation_index: the next generation (or pass) to train. Each is trained
        exactly once, so this is also the index its metrics and ONNX export
        will be written under.

    A trainer needing more state subclasses this and passes the subclass to
    resume() and peek_state(); the checkpoint persists whatever fields the
    class declares.
    """

    rows_trained: int = 0
    generation_index: int = 0


def save(paths: TagPaths, model, optimizer, state: GenerationalState, config: dict):
    """Persist the rolling checkpoint. `config` is the run's frozen task params,
    recorded for later inspection."""
    path = paths.rolling_checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside and rename over: a crash or a copy taken mid-write must
    # never tear the one file a resume depends on.
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            **asdict(state),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        tmp,
    )
    os.replace(tmp, path)


def _load(paths: TagPaths, device, state_cls: type) -> tuple[dict | None, GenerationalState]:
    """The rolling checkpoint's raw dict (None when there is none yet) and its
    cursor as `state_cls`. A field missing from the checkpoint keeps its
    default."""
    path = paths.rolling_checkpoint
    if not path.exists():
        return None, state_cls()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    names = {f.name for f in fields(state_cls)}
    return ckpt, state_cls(**{k: v for k, v in ckpt.items() if k in names})


def peek_state(paths: TagPaths, state_cls: type = GenerationalState) -> GenerationalState:
    """The rolling checkpoint's cursor alone, for a run deciding whether there
    is work left before it builds a model. A zero cursor when no checkpoint
    exists yet."""
    return _load(paths, "cpu", state_cls)[1]


def resume(
    paths: TagPaths, model, optimizer, device, state_cls: type = GenerationalState
) -> GenerationalState:
    """Load the rolling checkpoint into `model` and `optimizer` and return its
    cursor as `state_cls`. A zero cursor when no checkpoint exists yet."""
    ckpt, state = _load(paths, device, state_cls)
    if ckpt is None:
        return state
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(
        f"Resuming from {paths.rolling_checkpoint.name}: generation {state.generation_index}, "
        f"{state.rows_trained} rows trained"
    )
    return state

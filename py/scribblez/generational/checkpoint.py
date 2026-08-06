"""Rolling checkpoint for the generational trainer -- the restart authority.

A single rolling `model.pt` under the tag holds everything needed to resume: the
model and optimizer state plus the generational cursor. Restarting the script
loads this and continues exactly where it left off.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch

from ..paths import TagPaths


@dataclass
class GenerationalState:
    """The generational cursor persisted across restarts.

    rows_trained: cumulative rows (positions) trained -- the rows-clock that
        keys the dashboard x-axis.
    generation_index: the next generation to train. Each generation is trained
        exactly once, so this is also the metrics / ONNX index its checkpoint
        will be written under.

    A trainer needing more cursor than this subclasses it and names the subclass
    to resume(); the checkpoint persists whatever fields the class declares, so
    the extra state rides along without every other trainer carrying it.
    """

    rows_trained: int = 0
    generation_index: int = 0


def save(paths: TagPaths, model, optimizer, state: GenerationalState, config: dict):
    """Persist the rolling checkpoint (model + optimizer + generational cursor).
    `config` is the run's frozen task params, recorded for later inspection."""
    path = paths.rolling_checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **asdict(state),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def resume(
    paths: TagPaths, model, optimizer, device, state_cls: type = GenerationalState
) -> GenerationalState:
    """Load the rolling checkpoint into `model`/`optimizer` and return the cursor,
    as `state_cls` (a GenerationalState or a subclass of it carrying more).
    Returns a fresh zero cursor when no checkpoint exists yet; a field the
    checkpoint predates keeps its default."""
    path = paths.rolling_checkpoint
    if not path.exists():
        return state_cls()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    names = {f.name for f in fields(state_cls)}
    state = state_cls(**{k: v for k, v in ckpt.items() if k in names})
    print(
        f"Resuming from {path.name}: generation {state.generation_index}, "
        f"{state.rows_trained} rows trained"
    )
    return state

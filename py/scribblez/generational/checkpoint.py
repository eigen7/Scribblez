"""Rolling checkpoint for the generational trainer -- the restart authority.

A single rolling `model.pt` under the tag holds everything needed to resume: the
model and optimizer state plus the generational cursor (cumulative rows trained,
which generation is being trained, how far into that generation's epochs, and a
monotonic checkpoint index for the dashboard metrics key). Restarting the script
loads this and continues exactly where it left off.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..paths import TagPaths


@dataclass
class GenerationalState:
    """The generational cursor persisted across restarts.

    rows_trained: cumulative rows (positions) trained -- the rows-clock that keys
        the dashboard x-axis and drives the warmup learning rate.
    generation_index: the generation currently being trained (newest in the
        training window).
    epoch_in_generation: number of epochs already completed over that generation,
        so a resume continues at this epoch rather than repeating them.
    checkpoint_index: monotonic count of checkpoints written, used as the integer
        key of the dashboard `metrics` table.
    """

    rows_trained: int = 0
    generation_index: int = 0
    epoch_in_generation: int = 0
    checkpoint_index: int = 0


def save(paths: TagPaths, model, optimizer, state: GenerationalState, config: dict):
    """Persist the rolling checkpoint (model + optimizer + generational cursor).
    `config` is the run's frozen task params, recorded for later inspection."""
    path = paths.rolling_checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rows_trained": state.rows_trained,
            "generation_index": state.generation_index,
            "epoch_in_generation": state.epoch_in_generation,
            "checkpoint_index": state.checkpoint_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def resume(paths: TagPaths, model, optimizer, device) -> GenerationalState:
    """Load the rolling checkpoint into `model`/`optimizer` and return the cursor.
    Returns a fresh zero cursor when no checkpoint exists yet."""
    path = paths.rolling_checkpoint
    if not path.exists():
        return GenerationalState()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    state = GenerationalState(
        rows_trained=int(ckpt["rows_trained"]),
        generation_index=int(ckpt["generation_index"]),
        epoch_in_generation=int(ckpt["epoch_in_generation"]),
        checkpoint_index=int(ckpt["checkpoint_index"]),
    )
    print(
        f"Resuming from {path.name}: generation {state.generation_index}, "
        f"epoch {state.epoch_in_generation}, {state.rows_trained} rows trained"
    )
    return state

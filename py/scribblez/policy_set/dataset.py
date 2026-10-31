"""Batches of labeled pools for the policy-set model's trainers.

A batch is P pools with all their candidates, flattened as the move set
model's batches are (move_set_eval/dataset.py): the M candidates concatenated
position by position, each with a pos_id into [0, P). Every model input comes
from ffi.policy_set_positions, which rebuilds the positions by replay in the
hidden-leaves arm and encodes them with the engine code PolicyBot plays with
(engine training/policy_set_inputs.h): per position the board row and the
opponent-rack input (from the pool's record of the belief), per candidate the
move features and the equity input. A batch also carries the pool's labels,
and its rollout values as a grid (rollout_grid).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scribblez.ffi import format_layout, policy_set_positions, task_input_shapes
from scribblez.move_set_eval import moves as move_enc
from scribblez.policy_set import pools

# The opponent-rack input's width (engine policy_set::kOppRackFloats).
OPP_RACK_FLOATS = format_layout()["constants"]["policy_set"]["opp_rack_floats"]


@dataclass
class PoolItem:
    """One labeled pool and the .slog its position replays from."""

    slog: Path
    pool: pools.Pool


def read_items(directories: list[Path]) -> list[PoolItem]:
    """Every pool of the .pool files in `directories`, in name order."""
    items = []
    for d in directories:
        for path in sorted(d.glob(f"*{pools.SUFFIX}")):
            slog = path.with_suffix(".slog")
            items += [PoolItem(slog, p) for p in pools.read_pools(path)[1]]
    return items


class PoolBatcher:
    """Builds batches from PoolItems, with the board layout of the session's
    (hidden-leaves) arm."""

    def __init__(self):
        dims = {s.name: s.dims for s in task_input_shapes()}
        self.spatial_shape = tuple(dims["input_spatial"])
        self.scalar_width = dims["input_scalar"][0]
        self.spatial_floats = int(np.prod(self.spatial_shape))

    def build(
        self, items: list[PoolItem], transposes: np.ndarray | None = None
    ) -> dict[str, torch.Tensor]:
        """The batch of `items`; with `transposes` (len(items),) bool, the
        positions where it is set as their transposed positions (the board's
        diagonal symmetry), the candidates' tiles and cross-checks in the
        same frame. Labels are unchanged: the transposed position is the same
        game."""
        slogs = sorted({it.slog for it in items})
        file_id = {s: i for i, s in enumerate(slogs)}
        inputs = move_enc.batch_by_file(
            policy_set_positions,
            slogs,
            [(file_id[it.slog], it.pool.game, it.pool.turn) for it in items],
            [it.pool.moves for it in items],
            {"opp_racks": np.array([it.pool.opp_rack for it in items]), "transposes": transposes},
            {"rows", "opp_rack"},
        )
        rows = inputs["rows"]
        spatial = rows[:, : self.spatial_floats].reshape(len(items), *self.spatial_shape)
        scalar = np.ascontiguousarray(
            rows[:, self.spatial_floats : self.spatial_floats + self.scalar_width]
        )
        pos_id = np.concatenate(
            [np.full(len(it.pool.moves), j, dtype=np.int64) for j, it in enumerate(items)]
        )
        labels = {
            name: torch.from_numpy(np.concatenate([getattr(it.pool, name) for it in items]))
            for name in ("q", "win_prob", "equity_rank")
        }
        labels["stakes"] = torch.tensor(
            [pools.stakes(it.pool) for it in items], dtype=torch.float32
        )
        return {
            "input_spatial": torch.from_numpy(spatial),
            "input_scalar": torch.from_numpy(scalar),
            "opp_rack": torch.from_numpy(inputs["opp_rack"]),
            "move_letters": torch.from_numpy(inputs["letters"]),
            "move_blanks": torch.from_numpy(inputs["blanks"]),
            "move_squares": torch.from_numpy(inputs["squares"]),
            "move_tile_mask": torch.from_numpy(inputs["tile_mask"]),
            "move_scalars": torch.from_numpy(inputs["scalars"]),
            "move_cross_cells": torch.from_numpy(inputs["cells"]),
            "move_cross_letters": torch.from_numpy(inputs["letters_cross"]),
            "move_pos_id": torch.from_numpy(pos_id),
            "move_equity": torch.from_numpy(inputs["equity"]),
            **labels,
            **rollout_grid([it.pool for it in items]),
        }


def rollout_grid(pool_list: list[pools.Pool]) -> dict[str, torch.Tensor]:
    """The candidates' rollout values, position by position, as a grid:
    "rollout_values" (M, R), R the most rollouts any candidate ran, row c
    candidate c's values then zeros; and "rollouts" (M,), how many are its."""
    n = np.concatenate([p.rollouts for p in pool_list]).astype(np.int64)
    values = np.concatenate([p.rollout_values for p in pool_list])
    grid = np.zeros((len(n), int(n.max(initial=0))), dtype=np.float32)
    row = np.repeat(np.arange(len(n)), n)
    grid[row, np.arange(len(values)) - (np.cumsum(n) - n)[row]] = values
    return {"rollout_values": torch.from_numpy(grid), "rollouts": torch.from_numpy(n)}


def batch_order(n: int, batch_size: int, seed: int) -> list[np.ndarray]:
    """A shuffled split of range(n) into batches of `batch_size`, the last
    possibly short."""
    order = np.random.default_rng(seed).permutation(n)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]

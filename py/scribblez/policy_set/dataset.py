"""Batches of labeled pools for the policy-set model's trainers.

A batch is P pools with all their candidates, flattened as the move set
model's batches are (move_set_eval/dataset.py): the positions' board inputs,
rebuilt by replay through ffi.move_set_positions in the hidden-leaves arm,
then the M candidates concatenated position by position, each with a pos_id
into [0, P). Per position it also carries the opponent-rack input (the pool's
OppRackSummary) and, per candidate, the equity input (relative_equity) and the
pool's labels.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scribblez.ffi import task_input_shapes
from scribblez.move_set_eval import moves as move_enc
from scribblez.policy_set import pools

# The opponent-rack input: the 27 presence probabilities, whether the leave
# was modelled, and its value's mean and sd in units of VALUE_SCALE points.
OPP_RACK_FLOATS = 30
VALUE_SCALE = 10.0
# The equity input: each candidate's static equity less its pool's best, in
# units of VALUE_SCALE points, floored at EQUITY_FLOOR (100 points below).
EQUITY_FLOOR = -10.0


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


def opp_rack_input(pool: pools.Pool) -> np.ndarray:
    """The pool's OppRackSummary as the model reads it: (OPP_RACK_FLOATS,)."""
    return np.concatenate(
        [
            pool.opp_rack_presence,
            [
                float(pool.opp_leave_modelled),
                pool.opp_leave_value_mean / VALUE_SCALE,
                pool.opp_leave_value_sd / VALUE_SCALE,
            ],
        ]
    ).astype(np.float32)


class PoolBatcher:
    """Builds batches from PoolItems, with the board layout of the session's
    (hidden-leaves) arm."""

    def __init__(self):
        dims = {s.name: s.dims for s in task_input_shapes()}
        self.spatial_shape = tuple(dims["input_spatial"])
        self.scalar_width = dims["input_scalar"][0]
        self.spatial_floats = int(np.prod(self.spatial_shape))
        self.sd_index, self.sd_scale = move_enc.score_diff_input_layout()

    def build(self, items: list[PoolItem]) -> dict[str, torch.Tensor]:
        slogs = sorted({it.slog for it in items})
        file_id = {s: i for i, s in enumerate(slogs)}
        inputs = move_enc.batch_move_set_positions(
            slogs,
            [(file_id[it.slog], it.pool.game, it.pool.turn) for it in items],
            [it.pool.moves for it in items],
            equities=True,
        )
        rows = inputs["rows"]
        spatial = rows[:, : self.spatial_floats].reshape(len(items), *self.spatial_shape)
        scalar = np.ascontiguousarray(
            rows[:, self.spatial_floats : self.spatial_floats + self.scalar_width]
        )
        pos_id = np.concatenate(
            [np.full(len(it.pool.moves), j, dtype=np.int64) for j, it in enumerate(items)]
        )
        pre_diff = np.rint(scalar[:, self.sd_index] * self.sd_scale).astype(np.int32)
        enc = move_enc.encode_moves(
            np.concatenate([it.pool.moves for it in items]), pre_diff[pos_id]
        )
        labels = {
            name: torch.from_numpy(np.concatenate([getattr(it.pool, name) for it in items]))
            for name in ("q", "win_prob", "equity_rank")
        }
        return {
            "input_spatial": torch.from_numpy(spatial),
            "input_scalar": torch.from_numpy(scalar),
            "opp_rack": torch.from_numpy(np.stack([opp_rack_input(it.pool) for it in items])),
            "move_letters": torch.from_numpy(enc["letters"]),
            "move_blanks": torch.from_numpy(enc["blanks"]),
            "move_squares": torch.from_numpy(enc["squares"]),
            "move_tile_mask": torch.from_numpy(enc["tile_mask"]),
            "move_scalars": torch.from_numpy(enc["scalars"]),
            "move_cross_cells": torch.from_numpy(inputs["cells"]),
            "move_cross_letters": torch.from_numpy(inputs["letters"]),
            "move_pos_id": torch.from_numpy(pos_id),
            "move_equity": torch.from_numpy(
                relative_equity(inputs["equities"], pos_id, len(items))
            ),
            **labels,
        }


def relative_equity(equities: np.ndarray, pos_id: np.ndarray, num_positions: int) -> np.ndarray:
    """The equity input: each candidate's static equity less the best of its
    position's, in VALUE_SCALE points, floored at EQUITY_FLOOR."""
    best = np.full(num_positions, -np.inf, dtype=np.float32)
    np.maximum.at(best, pos_id, equities)
    return np.maximum((equities - best[pos_id]) / VALUE_SCALE, EQUITY_FLOOR).astype(np.float32)


def batch_order(n: int, batch_size: int, seed: int) -> list[np.ndarray]:
    """A shuffled split of range(n) into batches of `batch_size`, the last
    possibly short."""
    order = np.random.default_rng(seed).permutation(n)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]

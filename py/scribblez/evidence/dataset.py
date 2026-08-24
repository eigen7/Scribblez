"""Training dataset for the evidence-conditioned pass and the proves-best head:
trajectory .sobs sidecars paired with pre-move inputs reconstructed by replay.

Each position is identified by (game_index, turn_index) in a .slog file and
carries, in a companion trajectory .sobs, its simmed candidates in trajectory
order -- anchor, proposer picks, uniform tail -- with each one's raw CRN sim
observations. A training row is (position, evidence prefix, held-out simmed
candidate): the prefix is what the model conditions on, and the held-out
candidate's own sim outcome is the target -- its win value for the value
heads, and its CRN-paired gain over the prefix's best-so-far for the
proves-best head. No teacher label is read here -- docs/roadmap.md item 5
says why sim outcomes and not the teacher; the trainer's unfrozen mode reads
the same games' .mset sidecars through MsetDataset, as distillation rows for
the plain pass, never as targets for these.

Per epoch each position contributes ONE prefix, drawn uniformly from its valid
prefix sizes (0 .. last proposer pick; the tail is never evidence), so a pass
touches every position once and prefixes are covered across passes; the
held-out candidates are the simmed ones outside the prefix, tail included.
Prefix 0 rows are what keeps the empty-evidence pass anchored, and where the
gain target is the value itself. Only the simmed candidates are scored --
those are the rows with sim labels -- so the candidate set per position is
its trajectory (a handful of moves), never the full legal set.

Board inputs come from decode_rows(post_move=False), the standard replay
invariant (docs/architecture.md); the caller configures the FFI session's
information condition to the corpus's before iterating (adopt_information_
condition), as MsetDataset's caller does. A corpus is one proposer and one
condition throughout; mixing is refused.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from scribblez.dataset import row_layout
from scribblez.ffi import decode_rows, set_opp_leave_input
from scribblez.move_set_eval import moves as move_enc
from scribblez.sim_evidence.sobs import (
    SOBS_FLAG_OPEN_LEAVES,
    SOBS_FLAG_TRAJECTORY,
    SobsPosition,
    read_sobs,
    read_sobs_flags,
    read_sobs_proposer_hash,
)
from scribblez.workloads import pair_store


def complete_pairs(store: str | Path) -> list[Path]:
    """The .sobs files in `store` whose companion .slog exists, sorted
    (pair_store.complete_pairs)."""
    return pair_store.complete_pairs(store, ".sobs")


def adopt_information_condition(sobs_files: Iterable[str | Path]):
    """Point the FFI session's opponent-leave input arm at the condition the
    trajectories were simmed under, before any dataset is built (the arm is
    baked into the process-wide session at its creation)."""
    first = next(iter(sobs_files))
    set_opp_leave_input(bool(read_sobs_flags(first) & SOBS_FLAG_OPEN_LEAVES))


def trajectory_positions(sobs_path: str | Path) -> set[tuple[int, int]]:
    """The (game_index, turn_index) positions a trajectory .sobs holds -- the
    unfrozen trainer's selection of the same stem's .mset labels (the .mset
    labels many more positions per game than were simmed; the distillation
    anchor reads the simmed ones, which keeps it the size of the trajectory
    side and puts it on the positions the sim loss pulls on)."""
    return {(p.game_index, p.turn_index) for p in read_sobs(sobs_path)}


def sim_targets(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(K,) observation records -> (wld freq (K,3), delta [mean, std] (K,2),
    win value (K,) = P(win) + P(draw)/2), all float32."""
    n = np.maximum(obs["n"].astype(np.float64), 1.0)
    wld = np.stack([obs["wins"] / n, obs["draws"] / n, obs["losses"] / n], axis=1)
    mean = obs["delta_sum"] / n
    std = np.sqrt(np.maximum(obs["delta_sq_sum"] / n - mean**2, 0.0))
    delta = np.stack([mean, std], axis=1)
    value = wld[:, 0] + 0.5 * wld[:, 1]
    return wld.astype(np.float32), delta.astype(np.float32), value.astype(np.float32)


def gain_targets(value: np.ndarray, prefix: int) -> np.ndarray:
    """The proves-best target per candidate at evidence prefix `prefix`:
    max(0, v - best-so-far), best-so-far the maximum sim value over the
    prefix (the floor, 0, at prefix 0 -- where the target is the value
    itself). CRN-paired by construction: every candidate of a position shares
    its seed base."""
    best = float(value[:prefix].max()) if prefix > 0 else 0.0
    return np.maximum(value - best, 0.0).astype(np.float32)


class _TrajPosition:
    """One trajectory position: where to reconstruct its input, and its sims."""

    __slots__ = ("file_id", "sobs", "wld", "delta", "value")

    def __init__(self, file_id: int, sobs: SobsPosition):
        self.file_id = file_id
        self.sobs = sobs
        self.wld, self.delta, self.value = sim_targets(sobs.obs)


class TrajectoryDataset:
    """Streams flattened per-position batches from .slog/.sobs pairs,
    reconstructing pre-move board inputs by replay and drawing one evidence
    prefix per position per epoch."""

    def __init__(self, sobs_files: Iterable[str | Path]):
        sobs_files = [Path(f) for f in sobs_files]
        if not sobs_files:
            raise FileNotFoundError("empty sobs_files list")
        self._slogs: list[Path] = []
        self._files: list[Path] = []
        self._positions: list[_TrajPosition] = []
        self.proposer_hash: str | None = None
        self._flags: int | None = None
        self.absorb(sobs_files)
        input_shapes, _ = row_layout()
        self._spatial_shape = tuple(input_shapes[0].dims)
        self._scalar_width = int(input_shapes[1].dims[0])
        self._spatial_floats = int(np.prod(self._spatial_shape))
        self._sd_index, self._sd_scale = move_enc.score_diff_input_layout()

    @property
    def files(self) -> list[Path]:
        return list(self._files)

    @property
    def flags(self) -> int:
        return self._flags

    def absorb(self, sobs_files: Iterable[str | Path]) -> int:
        """Ingest `sobs_files` on top of what is held (a .sobs is immutable
        once delivered), holding each to the corpus's proposer and flags.
        Returns the positions added."""
        before = len(self._positions)
        for path in (Path(f) for f in sobs_files):
            flags, proposer = read_sobs_flags(path), read_sobs_proposer_hash(path)
            if not flags & SOBS_FLAG_TRAJECTORY:
                raise ValueError(f"{path} is not a trajectory .sobs")
            if self.proposer_hash is None:
                self.proposer_hash, self._flags = proposer, flags
            if proposer != self.proposer_hash:
                raise ValueError(f"corpus mixes proposers: {self.proposer_hash}, {proposer}")
            if flags != self._flags:
                raise ValueError(f"corpus mixes header flags: {self._flags}, {flags}")
            file_id = len(self._slogs)
            self._files.append(path)
            self._slogs.append(path.with_suffix(".slog"))
            self._positions.extend(_TrajPosition(file_id, p) for p in read_sobs(path))
        return len(self._positions) - before

    @property
    def num_positions(self) -> int:
        return len(self._positions)

    @property
    def num_candidates(self) -> int:
        return sum(len(p.sobs.moves) for p in self._positions)

    @property
    def open_leaves(self) -> bool:
        return bool(self._flags & SOBS_FLAG_OPEN_LEAVES)

    @property
    def spatial_planes(self) -> int:
        return self._spatial_shape[0]

    @property
    def scalar_size(self) -> int:
        return self._scalar_width

    @property
    def max_trajectory(self) -> int:
        """The longest trajectory held -- an upper bound on any evidence set."""
        return max((len(p.sobs.moves) for p in self._positions), default=0)

    def iter_batches(self, positions_per_batch: int, seed: int = 0, epoch_index: int = 0):
        """Yield one epoch of batch dicts (see _build_batch). Positions are
        shuffled globally and each draws its prefix, both deterministically
        for a given (seed, epoch)."""
        rng = np.random.default_rng(seed + epoch_index)
        order = rng.permutation(len(self._positions))
        prefixes = [rng.choice(self._positions[i].sobs.evidence_prefix_sizes()) for i in order]
        for start in range(0, len(order), positions_per_batch):
            idx = order[start : start + positions_per_batch]
            batch = [self._positions[i] for i in idx]
            yield self._build_batch(batch, prefixes[start : start + positions_per_batch])

    def _board_inputs(self, batch: list[_TrajPosition]) -> tuple[np.ndarray, np.ndarray]:
        """Pre-move board inputs, one decode_rows call per source file."""
        p = len(batch)
        spatial = np.empty((p, *self._spatial_shape), dtype=np.float32)
        scalar = np.empty((p, self._scalar_width), dtype=np.float32)
        by_file: dict[int, list[int]] = defaultdict(list)
        for local_p, pos in enumerate(batch):
            by_file[pos.file_id].append(local_p)
        for file_id, locals_ in by_file.items():
            games = np.array([batch[j].sobs.game_index for j in locals_], dtype=np.int64)
            turns = np.array([batch[j].sobs.turn_index for j in locals_], dtype=np.int64)
            rows = decode_rows(self._slogs[file_id], games, turns, post_move=False)
            spatial[locals_] = rows[:, : self._spatial_floats].reshape(-1, *self._spatial_shape)
            scalar[locals_] = rows[:, self._spatial_floats :][:, : self._scalar_width]
        return spatial, scalar

    def _build_batch(self, batch: list[_TrajPosition], prefixes: list[int]) -> dict:
        """One batch: P positions' board inputs; their trajectories' M
        candidates flattened (no padding, position blocks contiguous) as move
        inputs with `move_pos_id`; per-candidate sim targets `sim_wld` (M,3),
        `sim_delta` (M,2), `sim_value` (M,), `target_gain` (M,); `held_out`
        (M,) bool marking the candidates outside their position's prefix (the
        rows that carry loss); `slot` (M,) each candidate's index within its
        trajectory; and, for the evidence builder, `prefix_sizes` (P,),
        `pre_move_diff` (P,) and `positions` (the P SobsPositions)."""
        spatial, scalar = self._board_inputs(batch)
        all_moves = np.concatenate([pos.sobs.moves for pos in batch])
        counts = [len(pos.sobs.moves) for pos in batch]
        pos_id = np.repeat(np.arange(len(batch), dtype=np.int64), counts)
        slot = np.concatenate([np.arange(k, dtype=np.int64) for k in counts])
        pre_diff_points = np.rint(scalar[:, self._sd_index] * self._sd_scale).astype(np.int32)
        enc = move_enc.encode_moves(all_moves, pre_diff_points[pos_id])
        prefix_arr = np.asarray(prefixes, dtype=np.int64)
        pairs = zip(batch, prefixes, strict=True)
        gain = np.concatenate([gain_targets(pos.value, p) for pos, p in pairs])
        return {
            "input_spatial": torch.from_numpy(spatial),
            "input_scalar": torch.from_numpy(scalar),
            "move_letters": torch.from_numpy(enc["letters"]),
            "move_blanks": torch.from_numpy(enc["blanks"]),
            "move_squares": torch.from_numpy(enc["squares"]),
            "move_tile_mask": torch.from_numpy(enc["tile_mask"]),
            "move_scalars": torch.from_numpy(enc["scalars"]),
            "move_pos_id": torch.from_numpy(pos_id),
            "sim_wld": torch.from_numpy(np.concatenate([pos.wld for pos in batch])),
            "sim_delta": torch.from_numpy(np.concatenate([pos.delta for pos in batch])),
            "sim_value": torch.from_numpy(np.concatenate([pos.value for pos in batch])),
            "target_gain": torch.from_numpy(gain),
            "held_out": torch.from_numpy(slot >= prefix_arr[pos_id]),
            "slot": torch.from_numpy(slot),
            "prefix_sizes": torch.from_numpy(prefix_arr),
            "pre_move_diff": pre_diff_points,
            "positions": [pos.sobs for pos in batch],
        }

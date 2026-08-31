"""Training dataset for the evidence-conditioned pass and the proves-best head:
trajectory .sobs sidecars paired with pre-move inputs reconstructed by replay.

Each position is identified by (game_index, turn_index) in a .slog file and
carries, in a companion trajectory .sobs, its simmed candidates in trajectory
order -- anchor, on-policy proposer picks, off-policy draws -- with each one's
raw CRN sim observations. A training row is (position, evidence subset,
held-out simmed candidate): the subset is what the model conditions on, and the
held-out candidate's own sim outcome is the target -- its win value for the
value heads, and its CRN-paired gain over the subset's best-so-far for the
proves-best head. No teacher label is read here -- docs/roadmap.md item 5
says why sim outcomes and not the teacher; the trainer's unfrozen mode reads
the same games' .mset sidecars through MsetDataset, as distillation rows for
the plain pass, never as targets for these.

The evidence set is order-free -- the fusion stage is permutation-invariant and
the gain label is a max over the set -- so a row's evidence is an arbitrary
*subset* of a pool, not a leading prefix (docs/roadmap.md items 4-5). Per epoch
each position contributes `subsets_per_pool` drawn subsets (assemble_subset):
each is the empty set, or the anchor plus a random subset of the on-policy
picks, capped at the deployment evidence width -- the sets the deployed loop
actually walks. Off-policy draws are labels-only and never enter a subset. The
held-out candidates a row scores are every simmed candidate outside its subset,
off-policy included. The empty subset keeps the evidence-free pass anchored, and
is where the gain target is the value itself. Only the simmed candidates are
scored -- those are the rows with sim labels -- so the candidate set per position
is its trajectory (a handful of moves), never the full legal set.

A single explicit membership tensor (`in_evidence`, one bool per flattened
candidate) drives every seam that reads the subset -- the gain baseline, the
held-out mask, and both halves of each evidence token -- all in the same
enumeration order, so the observed and predicted halves cannot silently drift
apart.

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
    ROLE_ANCHOR,
    ROLE_ON_POLICY,
    SOBS_FLAG_OPEN_LEAVES,
    SOBS_FLAG_TRAJECTORY,
    SobsPosition,
    read_sobs,
    read_sobs_flags,
    read_sobs_leaf,
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


def gain_targets(value: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """The proves-best target per candidate given the evidence `subset` (a (K,)
    bool membership mask over the position's candidates): max(0, v -
    best-so-far), best-so-far the maximum sim value over the subset (the floor,
    0, for the empty subset -- where the target is the value itself). A max over
    the subset's members, not a leading prefix. CRN-paired by construction:
    every candidate of a position shares its seed base."""
    best = float(value[subset].max()) if subset.any() else 0.0
    return np.maximum(value - best, 0.0).astype(np.float32)


def assemble_subset(
    rng: np.random.Generator,
    pos: SobsPosition,
    max_evidence_width: int | None = None,
    empty_fraction: float | None = None,
) -> np.ndarray:
    """Draw one evidence subset for a trajectory position as a (K,) bool
    membership mask over its K simmed candidates.

    A subset is either empty (the prefix-0 rows that keep the plain pass
    calibrated) or the anchor plus a random subset of the on-policy picks --
    deployment holds only those -- of total size up to min(num_evidence,
    max_evidence_width). Off-policy draws are labels-only and never members.

    The size is drawn uniformly, then a uniform subset of that size, so the
    composition is any subset rather than a leading run. By default the size is
    uniform over {0..cap} -- the empty subset at ~1/(cap+1), matching the old
    uniform prefix draw and deployment's per-turn size sweep. Passing
    `empty_fraction` instead fixes P(empty subset) and draws a uniform non-empty
    size otherwise: an empty subset holds every candidate out, so its fraction
    sets the rows-clocked LR horizon and is pinned per run."""
    cap = pos.num_evidence
    if max_evidence_width is not None:
        cap = min(cap, max_evidence_width)
    mask = np.zeros(len(pos.roles), dtype=bool)
    if cap == 0:
        return mask
    if empty_fraction is None:
        size = int(rng.integers(cap + 1))
    else:
        size = 0 if rng.random() < empty_fraction else int(rng.integers(1, cap + 1))
    if size == 0:
        return mask
    eligible = np.arange(pos.num_evidence)
    anchor = eligible[pos.roles[eligible] == ROLE_ANCHOR]
    on_policy = eligible[pos.roles[eligible] == ROLE_ON_POLICY]
    mask[anchor] = True
    mask[rng.choice(on_policy, size=size - len(anchor), replace=False)] = True
    return mask


def _compact_index(mask: np.ndarray) -> np.ndarray:
    """(K,) bool subset -> (K,) int giving each member its 0-based rank among the
    subset's members in slot order (0 for non-members, which are never read).
    This is the padded evidence slot the token scatters to, so an arbitrary
    subset packs compactly the way the deployment builder does."""
    idx = np.zeros(len(mask), dtype=np.int64)
    idx[mask] = np.arange(int(mask.sum()))
    return idx


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
        self._leaf: tuple[str, int] | None = None
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
            leaf = read_sobs_leaf(path)
            if not flags & SOBS_FLAG_TRAJECTORY:
                raise ValueError(f"{path} is not a trajectory .sobs")
            if self.proposer_hash is None:
                self.proposer_hash, self._flags, self._leaf = proposer, flags, leaf
            if proposer != self.proposer_hash:
                raise ValueError(f"corpus mixes proposers: {self.proposer_hash}, {proposer}")
            if flags != self._flags:
                raise ValueError(f"corpus mixes header flags: {self._flags}, {flags}")
            if leaf != self._leaf:
                raise ValueError(f"corpus mixes leaf models/horizons: {self._leaf}, {leaf}")
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

    def iter_batches(
        self,
        positions_per_batch: int,
        seed: int = 0,
        epoch_index: int = 0,
        *,
        subsets_per_pool: int = 1,
        max_evidence_width: int | None = None,
        empty_fraction: float | None = None,
    ):
        """Yield one epoch of batch dicts (see _build_batch). Each position
        contributes `subsets_per_pool` drawn evidence subsets (assemble_subset,
        capped at `max_evidence_width`, empty at `empty_fraction`); the resulting
        (position, subset) units are shuffled globally and batched, all
        deterministically for a given (seed, epoch). `subsets_per_pool` and
        `empty_fraction` both move the epoch's held-out-row count, so they move
        the rows-clocked LR horizon -- pin them for a run."""
        rng = np.random.default_rng(seed + epoch_index)
        units = [
            (pos, assemble_subset(rng, pos.sobs, max_evidence_width, empty_fraction))
            for pos in self._positions
            for _ in range(subsets_per_pool)
        ]
        order = rng.permutation(len(units))
        for start in range(0, len(order), positions_per_batch):
            yield self._build_batch([units[j] for j in order[start : start + positions_per_batch]])

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

    def _build_batch(self, units: list[tuple[_TrajPosition, np.ndarray]]) -> dict:
        """One batch of P (position, subset) units. `units` pairs a position
        with one drawn evidence subset -- a (K,) bool membership mask over its
        K simmed candidates (assemble_subset); the same position may recur under
        different subsets.

        The batch flattens the units' M candidates (no padding, unit blocks
        contiguous) as move inputs with `move_pos_id`; per-candidate sim targets
        `sim_wld` (M,3), `sim_delta` (M,2), `sim_value` (M,), `target_gain` (M,);
        the membership `in_evidence` (M,) bool and its compact per-unit slot
        `ev_index` (M,); `held_out` (M,) bool marking the candidates outside
        their unit's subset (the rows that carry loss, off-policy always among
        them); `evidence_size` (P,) the members per unit; `slot` (M,) each
        candidate's trajectory index; the raw flattened records `all_moves`,
        `all_obs` (the observed half of every evidence token, gathered by
        `in_evidence`); and `pre_move_diff` (P,), `positions` (the P
        SobsPositions)."""
        positions = [pos for pos, _ in units]
        masks = [mask for _, mask in units]
        spatial, scalar = self._board_inputs(positions)
        all_moves = np.concatenate([pos.sobs.moves for pos in positions])
        all_obs = np.concatenate([pos.sobs.obs for pos in positions])
        counts = [len(pos.sobs.moves) for pos in positions]
        pos_id = np.repeat(np.arange(len(units), dtype=np.int64), counts)
        slot = np.concatenate([np.arange(k, dtype=np.int64) for k in counts])
        pre_diff_points = np.rint(scalar[:, self._sd_index] * self._sd_scale).astype(np.int32)
        enc = move_enc.encode_moves(all_moves, pre_diff_points[pos_id])
        in_evidence = np.concatenate(masks)
        ev_index = np.concatenate([_compact_index(mask) for mask in masks])
        evidence_size = np.array([int(mask.sum()) for mask in masks], dtype=np.int64)
        gain = np.concatenate([gain_targets(pos.value, mask) for pos, mask in units])
        return {
            "input_spatial": torch.from_numpy(spatial),
            "input_scalar": torch.from_numpy(scalar),
            "move_letters": torch.from_numpy(enc["letters"]),
            "move_blanks": torch.from_numpy(enc["blanks"]),
            "move_squares": torch.from_numpy(enc["squares"]),
            "move_tile_mask": torch.from_numpy(enc["tile_mask"]),
            "move_scalars": torch.from_numpy(enc["scalars"]),
            "move_pos_id": torch.from_numpy(pos_id),
            "sim_wld": torch.from_numpy(np.concatenate([pos.wld for pos in positions])),
            "sim_delta": torch.from_numpy(np.concatenate([pos.delta for pos in positions])),
            "sim_value": torch.from_numpy(np.concatenate([pos.value for pos in positions])),
            "target_gain": torch.from_numpy(gain),
            "in_evidence": torch.from_numpy(in_evidence),
            "ev_index": torch.from_numpy(ev_index),
            "held_out": torch.from_numpy(~in_evidence),
            "evidence_size": torch.from_numpy(evidence_size),
            "slot": torch.from_numpy(slot),
            "all_moves": all_moves,
            "all_obs": all_obs,
            "pre_move_diff": pre_diff_points,
            "positions": [pos.sobs for pos in positions],
        }

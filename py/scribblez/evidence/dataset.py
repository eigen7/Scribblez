"""Training dataset for the evidence-conditioned pass and the proves-best head:
trajectory .sobs sidecars paired with pre-move board inputs rebuilt by replay.

A position is a (game_index, turn_index) in a .slog file. Its trajectory .sobs
holds the candidates simmed there, in trajectory order (the anchor, the
proposer's on-policy picks, then off-policy draws), each with its raw sim
observations. All candidates of a position share rollout seeds (common random
numbers, CRN), so their outcomes are directly comparable.

A training row is (position, evidence subset, held-out candidate). The model
conditions on the subset, and the target is the held-out candidate's own sim
outcome: its win value for the value heads, and its gain over the subset's
best-so-far for the proves-best head. No teacher label is read here;
docs/roadmap.md item 5 explains why.

Evidence is an unordered set (the fusion stage is permutation-invariant and the
gain label is a max), so a row's evidence is any subset, not a leading prefix.
Each epoch, every position contributes `subsets_per_pool` subsets drawn by
assemble_subset, shaped like the sets the deployed loop sees. Every simmed
candidate outside the subset is held out and scored, off-policy draws
included. Only simmed candidates have sim labels, so a position's candidate set
is its short trajectory, never the full legal set.

One membership tensor, `in_evidence`, drives everything that reads the subset
(the gain baseline, the held-out mask, and both halves of each evidence token)
in one enumeration order, so the observed and predicted halves of a token
cannot drift apart.

Board inputs follow the replay-reconstruction invariant (docs/architecture.md).
Call adopt_information_condition before building a dataset. A corpus must have
one proposer, one information condition and one leaf model throughout.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from scribblez.dataset import row_layout
from scribblez.ffi import decode_rows, set_opp_leave_input
from scribblez.move_set_eval import moves as move_enc
from scribblez.sim_evidence.sobs import (
    COUNT_HEADS,
    RECORD_DTYPE,
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
    """The .sobs files in `store` whose companion .slog exists, sorted."""
    return pair_store.complete_pairs(store, ".sobs")


def adopt_information_condition(sobs_files: Iterable[str | Path]):
    """Set the FFI session's opponent-leave input to the information condition
    the trajectories were simmed under. Call before building any dataset: the
    setting is fixed when the process-wide session is created."""
    first = next(iter(sobs_files))
    set_opp_leave_input(bool(read_sobs_flags(first) & SOBS_FLAG_OPEN_LEAVES))


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
    """The proves-best target per candidate: max(0, value - best-so-far),
    where best-so-far is the highest sim value in `subset` (a (K,) bool mask),
    or 0 for an empty subset. The difference is CRN-paired, since a
    position's candidates share rollout seeds. evidence_fusion.best_so_far is
    the model-side twin of this baseline."""
    best = float(value[subset].max()) if subset.any() else 0.0
    return np.maximum(value - best, 0.0).astype(np.float32)


def assemble_subset(
    rng: np.random.Generator,
    pos: SobsPosition,
    max_evidence_width: int | None = None,
    empty_fraction: float | None = None,
) -> np.ndarray:
    """Draw one evidence subset for a trajectory position, as a (K,) bool
    mask over its simmed candidates.

    A subset is either empty (rows that keep the plain pass calibrated) or the
    anchor plus random on-policy picks, since deployment evidence holds only
    those; off-policy draws are never members. Its size is at most
    min(num_evidence, max_evidence_width).

    By default the size is uniform over {0..cap}, like deployment's per-turn
    sweep of set sizes. `empty_fraction` instead fixes P(empty) and draws a
    uniform non-empty size otherwise. An empty subset holds out every
    candidate, so this fraction changes the rows per epoch and with it the
    rows-clocked LR horizon; keep it fixed within a run."""
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
    """(K,) bool subset -> (K,) padded evidence slot for each member: its rank
    among the members, so any subset packs to the front as the deployment
    builder packs it. Non-members get 0 and are never read."""
    idx = np.zeros(len(mask), dtype=np.int64)
    idx[mask] = np.arange(int(mask.sum()))
    return idx


_OBS_DTYPE = RECORD_DTYPE["obs"]
_OBS_SCALAR_FIELDS = tuple(name for name in _OBS_DTYPE.names if name not in COUNT_HEADS)


class _PackedObs:
    """A position's (K,) observation records with the footprint histograms
    held sparse, rebuilt exactly per batch by densify().

    A dense record is ~35 KB but nearly empty (~200 rollouts touch at most 200
    of the 2927 classes per head). Held dense, a trajectory corpus of the size
    trained so far needs ~58 GiB of RAM, more than a 62 GB trainer host can
    spare."""

    __slots__ = ("k", "scalars", "hists")

    def __init__(self, obs: np.ndarray):
        self.k = len(obs)
        self.scalars = {name: obs[name].copy() for name in _OBS_SCALAR_FIELDS}
        self.hists = {}
        for name in COUNT_HEADS:
            dense = obs[name]
            rows, cls = np.nonzero(dense)
            self.hists[name] = (rows.astype(np.int32), cls.astype(np.int32), dense[rows, cls])

    def densify(self) -> np.ndarray:
        """The original (K,) structured records, rebuilt exactly."""
        out = np.zeros(self.k, dtype=_OBS_DTYPE)
        for name, col in self.scalars.items():
            out[name] = col
        for name, (rows, cls, vals) in self.hists.items():
            out[name][rows, cls] = vals
        return out


class _TrajPosition:
    """One trajectory position: where to reconstruct its input, and its sims.
    The observation records live packed in `obs`; `sobs` keeps every other
    SobsPosition field with its own `obs` emptied."""

    __slots__ = ("file_id", "sobs", "obs", "wld", "delta", "value")

    def __init__(self, file_id: int, sobs: SobsPosition):
        self.file_id = file_id
        self.wld, self.delta, self.value = sim_targets(sobs.obs)
        self.obs = _PackedObs(sobs.obs)
        # moves/roles are views into the position's dense record buffer; keeping
        # the views would keep the whole buffer alive. Copy them so it is freed.
        self.sobs = dataclasses.replace(
            sobs,
            moves=sobs.moves.copy(),
            roles=sobs.roles.copy(),
            obs=np.empty(0, dtype=_OBS_DTYPE),
        )


class TrajectoryDataset:
    """Streams flattened batches of (position, evidence subset) units from
    .slog/.sobs pairs."""

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
        """Ingest more .sobs files, returning the number of positions added.
        Each must match the corpus's proposer, header flags and leaf model."""
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
        """The longest trajectory held, an upper bound on any evidence set."""
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
        """Yield one epoch of batch dicts (see _build_batch).

        Each position contributes `subsets_per_pool` subsets (assemble_subset);
        the (position, subset) units are shuffled and batched
        deterministically for a given seed + epoch_index. `subsets_per_pool`
        and `empty_fraction` change the held-out rows per epoch and so the
        rows-clocked LR horizon; keep them fixed within a run."""
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
        """One batch of P (position, subset) units; a position may recur under
        different subsets. The units' M candidates are flattened, each unit's
        block contiguous, with no padding.

        Keys beyond the board and move inputs (MsetDataset's names):
          sim_wld (M, 3), sim_delta (M, 2), sim_value (M,)   sim targets
          target_gain (M,)     proves-best target (gain_targets)
          in_evidence (M,)     bool, member of its unit's subset
          ev_index (M,)        padded evidence slot of a member (_compact_index)
          held_out (M,)        ~in_evidence: the rows that carry loss
          evidence_size (P,)   members per unit
          slot (M,)            index within the position's trajectory
          all_moves, all_obs   raw flattened .sobs records (numpy)
          pre_move_diff (P,)   mover's pre-move score differential (numpy)
          positions            the P SobsPositions, obs re-attached
        """
        positions = [pos for pos, _ in units]
        masks = [mask for _, mask in units]
        spatial, scalar = self._board_inputs(positions)
        all_moves = np.concatenate([pos.sobs.moves for pos in positions])
        dense_obs = [pos.obs.densify() for pos in positions]
        all_obs = np.concatenate(dense_obs)
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
            "positions": [
                dataclasses.replace(pos.sobs, obs=obs)
                for pos, obs in zip(positions, dense_obs, strict=True)
            ],
        }

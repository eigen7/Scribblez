"""Training dataset for the move set evaluation model: .mset targets paired
with pre-move inputs reconstructed by replay.

Each position is identified by (game_index, turn_index) in a .slog file and
carries, in a companion .mset sidecar, a sampled set of candidate moves with the
teacher position evaluation model's readouts for each candidate's post-move state
(scribblez.move_set_eval.targets). This dataset:

  * indexes every labeled position across a directory's .mset/.slog pairs and
    holds the (small) candidate/target records in memory;
  * shuffles at the position level across all files each epoch (not inside any
    one file), per the replay-reconstruction pipeline
    (docs/architecture.md);
  * reconstructs each position's PRE-move board input on demand via
    decode_rows(post_move=False) -- the standard invariant: inputs are recomputed
    from the replay, targets come from the sidecar.

A batch is P positions. The variable-length candidate sets are flattened across
those positions with no padding: all M = sum of per-position candidate counts
moves are concatenated, and a `move_pos_id` array maps each move back to its
position row (0..P-1) so the model can attend each move to its own board and the
loss can be taken per move.

The board-input encoding arm (the opponent-leave block) is a property of the
process-wide FFI session; the caller must configure it (via
scribblez.ffi.set_opp_leave_input) to match the model being trained before
iterating. The .mset open-leaves flag is surfaced as `open_leaves` so the
caller can honor it.

A dataset is stratified or full-sweep throughout, after the .mset header flag
(`full_sweep`); mixing the two is refused. Swept positions are the held-out
evaluation slice the A3 gate metrics need and are never trained on, so a shared
store is split at file level before construction
(scribblez.move_set_eval.targets.partition_full_sweep).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch

from scribblez.dataset import row_layout
from scribblez.ffi import decode_rows, set_opp_leave_input

from . import moves as move_enc
from .targets import (
    MSET_FLAG_FULL_SWEEP,
    MSET_FLAG_OPEN_LEAVES,
    complete_pairs,
    dequantize_planes,
    read_mset,
    read_mset_flags,
)


def adopt_information_condition(mset_files: Iterable[str | Path]):
    """Point the FFI session's opponent-leave input arm at the arm `mset_files`
    were labeled under, before any of them is opened as a dataset.

    The arm is baked into the process-wide session when it is created, and a
    dataset creates it just by asking for the row layout -- so the condition has
    to be read from a header (read_mset_flags), which touches no session, rather
    than from a constructed dataset's `open_leaves`. One file answers for the
    corpus: MsetDataset refuses a set that mixes header flags.
    """
    first = next(iter(mset_files))
    set_opp_leave_input(bool(read_mset_flags(first) & MSET_FLAG_OPEN_LEAVES))


class _Position:
    """One labeled position: where to reconstruct its input, and its targets."""

    __slots__ = (
        "file_id",
        "game_index",
        "turn_index",
        "moves",
        "targets",
        "plane_scales",
        "planes",
        "num_legal_moves",
    )

    def __init__(
        self,
        file_id: int,
        game_index: int,
        turn_index: int,
        moves,
        targets,
        plane_scales,
        planes,
        num_legal_moves: int,
    ):
        self.file_id = file_id
        self.game_index = game_index
        self.turn_index = turn_index
        self.moves = moves  # (K,) MOVE_DTYPE
        self.targets = targets  # (K, 5) float32: [p_win, p_draw, p_loss, sd_mean, sd_std]
        # Quantized teacher placement planes, held as stored (u8 cells + f4
        # scales -- ~1/4 the memory of floats) and dequantized per batch; None
        # on a plane-less (full-sweep) corpus.
        self.plane_scales = plane_scales  # (K, 4) float32 | None
        self.planes = planes  # (K, 4, PLANE_WIDTH) uint8 | None
        self.num_legal_moves = num_legal_moves  # 0 unless swept (see targets.MsetPosition)


class MsetDataset:
    """Streams flattened candidate batches from .mset/.slog pairs in one or more
    directories, reconstructing pre-move board inputs by replay."""

    def __init__(
        self,
        data_dir: str | Path | Iterable[str | Path] | None = None,
        *,
        mset_files: Iterable[str | Path] | None = None,
        select: Callable[[Path], set[tuple[int, int]]] | None = None,
    ):
        """Exactly one source: `data_dir` (directories whose complete pairs are
        globbed) or `mset_files` (explicit .mset paths — the file-level-split
        case, where train and held-out pairs share a directory). `select`,
        given an .mset path, names the (game_index, turn_index) positions to
        keep from it; the rest of the file is not held (the evidence trainer
        reads only the trajectory positions' labels this way)."""
        assert (data_dir is None) != (mset_files is None), (
            "pass exactly one of data_dir or mset_files"
        )
        if mset_files is None:
            if isinstance(data_dir, (str, Path)):
                data_dirs = [Path(data_dir)]
            else:
                data_dirs = [Path(d) for d in data_dir]
            mset_files = sorted(f for d in data_dirs for f in complete_pairs(d))
            if not mset_files:
                dirs = ", ".join(str(d) for d in data_dirs)
                raise FileNotFoundError(f"No .mset files with a companion .slog in {dirs}")
        else:
            mset_files = [Path(f) for f in mset_files]
            if not mset_files:
                raise FileNotFoundError("empty mset_files list")

        self._slogs: list[Path] = []
        self._positions: list[_Position] = []
        self._files: list[Path] = []
        self.dropped_candidates = 0
        # A corpus must come from one blessed teacher and one information
        # condition; a mix would train against inconsistent targets. The
        # full-sweep bit is held to the same rule for a different reason: swept
        # files are evaluation-only, so a dataset that mixed them with
        # stratified ones is exactly the leak file-level routing prevents
        # (targets.partition_full_sweep is how a shared store is split). The
        # first file sets both; absorb() holds every later one to them.
        self.model_hash: str | None = None
        self._flags: int | None = None
        self._record_planes: int | None = None
        self._select = select
        self.absorb(mset_files)

        input_shapes, _ = row_layout()
        self._spatial_shape = tuple(input_shapes[0].dims)
        self._scalar_width = int(input_shapes[1].dims[0])
        self._spatial_floats = int(np.prod(self._spatial_shape))

        # Where the mover's pre-move score differential sits in the decoded
        # scalar input, so each candidate's resultant post-move differential
        # feature can be formed from the same value the board trunk sees.
        self._sd_index, self._sd_scale = move_enc.score_diff_input_layout()

    @property
    def flags(self) -> int:
        """The header flags every file in this corpus carries -- what a caller
        holding a growing store checks a candidate file against before offering
        it, since a dataset cannot mix them."""
        return self._flags

    @property
    def files(self) -> list[Path]:
        """The .mset files ingested so far, in ingest order -- what a caller
        watching a growing store diffs against to find the new ones."""
        return list(self._files)

    def absorb(self, mset_files: Iterable[str | Path]) -> int:
        """Ingest `mset_files` on top of what is already held, returning the
        positions added.

        A .mset is immutable once delivered, so a store that is still being
        written is absorbed by reading only its new files -- the cost of
        keeping up with a running generator is the new data, not a re-read of
        the corpus. Every file is held to the first one's teacher and header
        flags.
        """
        before, dropped_before = len(self._positions), self.dropped_candidates
        for mset_path in (Path(f) for f in mset_files):
            parsed = read_mset(mset_path)
            if self.model_hash is None:
                self.model_hash, self._flags = parsed.model_hash, parsed.flags
                self._record_planes = parsed.record_planes
            if parsed.model_hash != self.model_hash:
                raise ValueError(
                    f"mset corpus mixes teacher hashes: {self.model_hash}, {parsed.model_hash}"
                )
            if parsed.flags != self._flags:
                raise ValueError(f"mset corpus mixes header flags: {self._flags}, {parsed.flags}")
            if parsed.record_planes != self._record_planes:
                raise ValueError(
                    f"mset corpus mixes plane counts: {self._record_planes}, {parsed.record_planes}"
                )
            self._files.append(mset_path)
            file_id = len(self._slogs)
            self._slogs.append(mset_path.with_suffix(".slog"))
            selected = self._select(mset_path) if self._select is not None else None
            self._ingest_positions(parsed.positions, file_id, selected)
        if self.dropped_candidates > dropped_before:
            n = self.dropped_candidates - dropped_before
            print(f"dropped {n} candidate(s) with non-finite teacher targets")
        return len(self._positions) - before

    def _ingest_positions(self, positions, file_id: int, selected: set[tuple[int, int]] | None):
        """Append one file's positions (those in `selected`, when given),
        dropping candidates the teacher labeled non-finitely."""
        for pos in positions:
            if selected is not None and (pos.game_index, pos.turn_index) not in selected:
                continue
            moves, targets = pos.moves, pos.targets
            plane_scales, planes = pos.plane_scales, pos.planes
            # The generator stores the teacher's readouts verbatim, and the
            # FP16 score-diff std head can overflow to inf on near-terminal
            # post-move states. The generator now clamps the stored std
            # (kSdStdCap in engine/include/training/move_set_eval_target_log.h),
            # so this drop only fires on corpora generated before the clamp;
            # it goes when those corpora are retired.
            keep = np.isfinite(targets).all(axis=1)
            if not keep.all():
                self.dropped_candidates += int((~keep).sum())
                if not keep.any():
                    continue
                moves, targets = moves[keep], targets[keep]
                if planes is not None:
                    plane_scales, planes = plane_scales[keep], planes[keep]
            self._positions.append(
                _Position(
                    file_id,
                    pos.game_index,
                    pos.turn_index,
                    moves,
                    targets,
                    plane_scales,
                    planes,
                    pos.num_legal_moves,
                )
            )

    @property
    def num_positions(self) -> int:
        return len(self._positions)

    @property
    def num_candidates(self) -> int:
        return sum(len(p.moves) for p in self._positions)

    @property
    def full_sweep(self) -> bool:
        """Whether these positions are capped sweeps of their legal candidates
        rather than stratified samples -- the evaluation-only slice the A3 gate
        metrics are read on, never trained against."""
        return bool(self._flags & MSET_FLAG_FULL_SWEEP)

    @property
    def sweep_coverage(self) -> tuple[float, int]:
        """(mean fraction of legal moves swept, positions the cap truncated),
        over the positions carrying a legal-move count. (1.0, 0) when none do
        -- a stratified corpus records no counts, and nothing was truncated."""
        swept = [p for p in self._positions if p.num_legal_moves > 0]
        if not swept:
            return 1.0, 0
        covered = sum(len(p.moves) / p.num_legal_moves for p in swept)
        truncated = sum(1 for p in swept if len(p.moves) < p.num_legal_moves)
        return covered / len(swept), truncated

    @property
    def has_planes(self) -> bool:
        """Whether the records carry the teacher's quantized placement planes
        (batches then include "target_planes"). Stratified training files do;
        the full-sweep evaluation slice does not."""
        return bool(self._record_planes)

    @property
    def open_leaves(self) -> bool:
        """Whether the teacher labeled these positions under the open-leaves
        information condition (the caller must then enable the opponent-leave
        input block on the FFI session to match)."""
        return bool(self._flags & MSET_FLAG_OPEN_LEAVES)

    @property
    def spatial_planes(self) -> int:
        return self._spatial_shape[0]

    @property
    def scalar_size(self) -> int:
        return self._scalar_width

    def iter_batches(
        self,
        positions_per_batch: int,
        seed: int = 0,
        epoch_index: int = 0,
        max_candidates: int | None = None,
    ):
        """Yield one epoch of batch dicts.

        Positions are shuffled globally (deterministically for a given
        seed/epoch); each batch reconstructs its positions' board inputs and
        flattens their candidate sets. `positions_per_batch` bounds the
        positions in a batch and `max_candidates`, if given, the moves: a swept
        position carries hundreds of candidates against a stratified one's ~15,
        so over full-sweep positions the position count alone no longer bounds
        what a batch costs (each candidate carries its own move activations and
        attention weights on top of its position's board encoding).
        """
        rng = np.random.default_rng(seed + epoch_index)
        order = rng.permutation(len(self._positions))
        for chunk in self._batches_of(order, positions_per_batch, max_candidates):
            yield self._build_batch([self._positions[i] for i in chunk])

    def _batches_of(self, order, positions_per_batch: int, max_candidates: int | None):
        """Split a position ordering into batches under both bounds. A position
        whose candidate set alone exceeds `max_candidates` still forms a batch:
        candidate sets are indivisible here, and the whole point of the sweep is
        the positions with the most of them."""
        chunk: list[int] = []
        candidates = 0
        for i in order:
            k = len(self._positions[i].moves)
            if len(chunk) >= positions_per_batch or (
                max_candidates is not None and chunk and candidates + k > max_candidates
            ):
                yield chunk
                chunk, candidates = [], 0
            chunk.append(i)
            candidates += k
        if chunk:
            yield chunk

    def _build_batch(self, batch: list[_Position]) -> dict[str, torch.Tensor]:
        p = len(batch)
        spatial = np.empty((p, *self._spatial_shape), dtype=np.float32)
        scalar = np.empty((p, self._scalar_width), dtype=np.float32)

        # Reconstruct board inputs one decode_rows call per source file (each
        # call addresses a batch of positions in one .slog by identity).
        by_file: dict[int, list[int]] = defaultdict(list)
        for local_p, pos in enumerate(batch):
            by_file[pos.file_id].append(local_p)
        for file_id, locals_ in by_file.items():
            games = np.array([batch[j].game_index for j in locals_], dtype=np.int64)
            turns = np.array([batch[j].turn_index for j in locals_], dtype=np.int64)
            rows = decode_rows(self._slogs[file_id], games, turns, post_move=False)
            spatial_block = rows[:, : self._spatial_floats]
            spatial_block = spatial_block.reshape(len(locals_), *self._spatial_shape)
            scalar_block = rows[:, self._spatial_floats : self._spatial_floats + self._scalar_width]
            for k, j in enumerate(locals_):
                spatial[j] = spatial_block[k]
                scalar[j] = scalar_block[k]

        # Flatten candidate sets across positions (no padding); pos_id maps each
        # move back to its board row.
        all_moves = np.concatenate([pos.moves for pos in batch])
        all_targets = np.concatenate([pos.targets for pos in batch]).astype(np.float32)
        pos_id = np.concatenate(
            [np.full(len(pos.moves), local_p, dtype=np.int64) for local_p, pos in enumerate(batch)]
        )
        # Each position's pre-move differential (points) read from the decoded
        # score-diff scalar, expanded to one value per candidate move.
        pre_diff_points = np.rint(scalar[:, self._sd_index] * self._sd_scale).astype(np.int32)
        move_pre_diffs = pre_diff_points[pos_id]
        enc = move_enc.encode_moves(all_moves, move_pre_diffs)

        batch_out = {
            "input_spatial": torch.from_numpy(spatial),
            "input_scalar": torch.from_numpy(scalar),
            "move_letters": torch.from_numpy(enc["letters"]),
            "move_blanks": torch.from_numpy(enc["blanks"]),
            "move_squares": torch.from_numpy(enc["squares"]),
            "move_tile_mask": torch.from_numpy(enc["tile_mask"]),
            "move_scalars": torch.from_numpy(enc["scalars"]),
            "move_pos_id": torch.from_numpy(pos_id),
            "target_wld": torch.from_numpy(all_targets[:, :3].copy()),
            "target_score_diff": torch.from_numpy(all_targets[:, 3:5].copy()),
        }
        if self.has_planes:
            target_planes = np.concatenate(
                [dequantize_planes(pos.planes, pos.plane_scales) for pos in batch]
            )
            batch_out["target_planes"] = torch.from_numpy(target_planes)
        return batch_out

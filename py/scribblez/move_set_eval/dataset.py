"""Training dataset for the move set evaluation model: .mset targets paired
with pre-move board inputs reconstructed by replay.

A position is a (game_index, turn_index) in a .slog file. Its companion .mset
sidecar holds a sampled set of candidate moves, each with the teacher's
readouts for its post-move state (targets.py). The dataset holds those small
records in memory for every labeled position, shuffles positions across all
files each epoch, and rebuilds each position's pre-move board input on demand
with decode_rows(post_move=False). This is the project's replay-reconstruction
invariant (docs/architecture.md): inputs are recomputed from the replay,
targets come from the sidecar.

A batch is P positions whose variable-length candidate sets are concatenated
without padding into M moves. `move_pos_id` maps each move to its position row
in [0, P), so the model attends each move to its own board and the loss is
taken per move.

The opponent-leave block of the board input is a process-wide FFI session
setting. Call adopt_information_condition before building a dataset so the
inputs match the condition the corpus was labeled under.

A dataset is either all stratified or all full-sweep, per the .mset header
flag. Swept positions are the held-out evaluation slice for eval.py's recall
and regret metrics and are never trained on, so a store holding both is split
by targets.partition_full_sweep before construction.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch

from scribblez.dataset import row_layout
from scribblez.ffi import cross_check_deltas, decode_rows, set_opp_leave_input

from . import moves as move_enc
from .targets import (
    MSET_FLAG_FULL_SWEEP,
    MSET_FLAG_OPEN_LEAVES,
    complete_pairs,
    dequantize_planes,
    read_mset,
    read_mset_flags,
)

# Batch key -> scribblez.ffi.cross_check_deltas key, each (M, max_cross_deltas):
# a move's sparse cross-check changes (axis, square, letter masks before and
# after) plus the mask marking real entries.
CROSS_DELTA_KEYS = {
    "move_cross_axes": "axes",
    "move_cross_squares": "squares",
    "move_cross_old_masks": "old_masks",
    "move_cross_new_masks": "new_masks",
    "move_cross_mask": "delta_mask",
}


def adopt_information_condition(mset_files: Iterable[str | Path]):
    """Set the FFI session's opponent-leave input to the information condition
    `mset_files` were labeled under. Call before opening any of them as a
    dataset.

    The setting is fixed when the process-wide session is created, and
    constructing a dataset creates it (via row_layout). So the condition is
    read from a raw header rather than a dataset's `open_leaves`. The first
    file speaks for all of them, since MsetDataset refuses mixed flags.
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
        # Kept quantized (about 1/4 the memory of floats) and dequantized per
        # batch; None on a full-sweep corpus.
        self.plane_scales = plane_scales  # (K, num_planes) float32 | None
        self.planes = planes  # (K, num_planes, PLANE_WIDTH) uint8 | None
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
        with_cross_check_deltas: bool = False,
    ):
        """Pass exactly one source: `data_dir` (directories whose complete
        pairs are globbed) or `mset_files` (explicit paths, for when train and
        held-out pairs share a directory).

        `select`, given a .mset path, names the (game_index, turn_index)
        positions to keep from it; the evidence trainer uses it to hold only
        its trajectory positions' labels. `with_cross_check_deltas` adds each
        move's cross-check changes (CROSS_DELTA_KEYS) to the batches; only
        cross_check_diagnostic reads them so far."""
        self._with_cross_check_deltas = with_cross_check_deltas
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
        # A corpus must come from one teacher and one information condition, or
        # it trains against inconsistent targets. The full-sweep bit must also
        # agree, because swept files are evaluation-only and mixing them in
        # would leak them into training. The first file sets these; absorb()
        # holds every later file to them.
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
        # scalar input. Each candidate's post-move differential feature is built
        # from this same value, so it agrees with what the board trunk sees.
        self._sd_index, self._sd_scale = move_enc.score_diff_input_layout()

    @property
    def flags(self) -> int:
        """The header flags every file in this corpus carries. A caller
        watching a growing store checks new files against them before
        absorbing, since a dataset cannot mix flags."""
        return self._flags

    @property
    def files(self) -> list[Path]:
        """The .mset files ingested so far, in ingest order."""
        return list(self._files)

    def absorb(self, mset_files: Iterable[str | Path]) -> int:
        """Ingest more .mset files, returning the number of positions added.

        A delivered .mset never changes, so keeping up with a running generator
        only requires reading its new files. Every file must match the first
        one's teacher hash, header flags and plane count.
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
        """Append one file's positions (only those in `selected`, if given),
        dropping candidates with non-finite teacher targets."""
        for pos in positions:
            if selected is not None and (pos.game_index, pos.turn_index) not in selected:
                continue
            moves, targets = pos.moves, pos.targets
            plane_scales, planes = pos.plane_scales, pos.planes
            # The FP16 score-diff std head can overflow to inf on near-terminal
            # post-move states. The generator clamps the stored std (kSdStdCap
            # in move_set_eval_target_log.h), so this drop only matters for
            # corpora generated without the clamp and can be removed once none
            # remain.
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
        (the evaluation-only slice) rather than stratified samples."""
        return bool(self._flags & MSET_FLAG_FULL_SWEEP)

    @property
    def sweep_coverage(self) -> tuple[float, int]:
        """(mean fraction of legal moves swept, number of positions the cap
        truncated) over positions that record a legal-move count; (1.0, 0) for
        a stratified corpus, which records none."""
        swept = [p for p in self._positions if p.num_legal_moves > 0]
        if not swept:
            return 1.0, 0
        covered = sum(len(p.moves) / p.num_legal_moves for p in swept)
        truncated = sum(1 for p in swept if len(p.moves) < p.num_legal_moves)
        return covered / len(swept), truncated

    @property
    def has_planes(self) -> bool:
        """Whether batches include "target_planes". Stratified files carry
        placement planes; full-sweep files do not."""
        return bool(self._record_planes)

    @property
    def open_leaves(self) -> bool:
        """Whether the corpus was labeled under the open-leaves information
        condition (see adopt_information_condition)."""
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
        """Yield one epoch of batch dicts, positions shuffled deterministically
        by seed + epoch_index.

        `max_candidates`, if given, also caps the moves per batch. A swept
        position has hundreds of candidates against a stratified one's ~15, and
        each candidate carries its own activations, so the position count alone
        does not bound a sweep batch's memory.
        """
        rng = np.random.default_rng(seed + epoch_index)
        order = rng.permutation(len(self._positions))
        for chunk in self._batches_of(order, positions_per_batch, max_candidates):
            yield self._build_batch([self._positions[i] for i in chunk])

    def _batches_of(self, order, positions_per_batch: int, max_candidates: int | None):
        """Split a position ordering into batches under both bounds. A position
        whose candidate set alone exceeds `max_candidates` still gets a batch of
        its own: candidate sets are indivisible, and the largest ones are what
        the sweep is for."""
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

        # One decode_rows call per source .slog.
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

        all_moves = np.concatenate([pos.moves for pos in batch])
        all_targets = np.concatenate([pos.targets for pos in batch]).astype(np.float32)
        pos_id = np.concatenate(
            [np.full(len(pos.moves), local_p, dtype=np.int64) for local_p, pos in enumerate(batch)]
        )
        # Pre-move score differential in points, one per candidate.
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
        if self._with_cross_check_deltas:
            batch_out.update(self._cross_check_deltas(batch, by_file))
        if self.has_planes:
            target_planes = np.concatenate(
                [dequantize_planes(pos.planes, pos.plane_scales) for pos in batch]
            )
            batch_out["target_planes"] = torch.from_numpy(target_planes)
        return batch_out

    def _cross_check_deltas(
        self, batch: list[_Position], by_file: dict[int, list[int]]
    ) -> dict[str, torch.Tensor]:
        """Each move's cross-check entries (scribblez.ffi.cross_check_deltas),
        one replay call per source file, scattered into the batch's flattened
        move order."""
        counts = np.array([len(pos.moves) for pos in batch], dtype=np.int64)
        starts = np.cumsum(counts) - counts
        out: dict[str, np.ndarray] = {}
        for file_id, locals_ in by_file.items():
            deltas = cross_check_deltas(
                self._slogs[file_id],
                np.array([batch[j].game_index for j in locals_], dtype=np.int64),
                np.array([batch[j].turn_index for j in locals_], dtype=np.int64),
                counts[locals_],
                np.concatenate([batch[j].moves for j in locals_]),
            )
            rows = np.concatenate([np.arange(starts[j], starts[j] + counts[j]) for j in locals_])
            for key, values in deltas.items():
                if key not in out:
                    out[key] = np.zeros((int(counts.sum()), *values.shape[1:]), dtype=values.dtype)
                out[key][rows] = values
        # torch has no usable uint32: the 26-bit letter masks ride as int64.
        return {
            batch_key: torch.from_numpy(
                out[key].astype(np.int64 if "masks" in key else out[key].dtype)
            )
            for batch_key, key in CROSS_DELTA_KEYS.items()
        }

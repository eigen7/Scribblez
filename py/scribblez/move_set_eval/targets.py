"""Pure-numpy reader for .mset move set evaluation model distillation-target
sidecars.

The binary layout is owned by engine/include/training/move_set_eval_target_log.h (one
TargetFileHeader, then per position a TargetPositionHeader followed by
its (Move, targets) records); the dtypes below mirror those packed structs and
are guarded by itemsize checks so a C++ layout change fails loudly here rather
than misparsing. The record's target width comes from the header
(`record_floats`), so the record dtype is built per file.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.sim_evidence.sobs import MOVE_DTYPE

MSET_MAGIC = 0x5445534D  # "MSET"
MSET_VERSION = 1

# Version-1 target floats per candidate, in record order (mover's POV).
TARGET_NAMES_V1 = ("p_win", "p_draw", "p_loss", "sd_mean", "sd_std")

# TargetFileHeader.flags bits (mirrors the .sobs convention).
MSET_FLAG_OPEN_LEAVES = 2
# Every position in the file is a capped sweep of its legal candidates rather
# than a stratified sample -- an evaluation-only file, held out by construction.
MSET_FLAG_FULL_SWEEP = 4

_FILE_HEADER = np.dtype(
    {
        "names": [
            "magic",
            "version",
            "reserved",
            "num_positions",
            "record_floats",
            "flags",
            "model_hash",
        ],
        "formats": ["<u4", "<u2", "<u2", "<u4", "<u4", "<u4", "S64"],
        "offsets": [0, 4, 6, 8, 12, 16, 20],
        "itemsize": 84,
    }
)

_POSITION_HEADER = np.dtype(
    {
        "names": ["game_index", "turn_index", "num_candidates", "num_legal_moves"],
        "formats": ["<u4", "<u4", "<u4", "<u4"],
        "offsets": [0, 4, 8, 12],
        "itemsize": 16,
    }
)


def _record_dtype(record_floats: int) -> np.dtype:
    return np.dtype([("move", MOVE_DTYPE), ("targets", "<f4", (record_floats,))])


@dataclass
class MsetPosition:
    """One position's selected candidates and their teacher readouts."""

    game_index: int
    turn_index: int
    moves: np.ndarray  # (K,) MOVE_DTYPE
    targets: np.ndarray  # (K, record_floats) float32
    # Legal moves the position had, recorded for a swept position so that a
    # sweep the generator's cap truncated is visible as num_legal_moves > K.
    # 0 ("not recorded") on every stratified position.
    num_legal_moves: int = 0


@dataclass
class MsetFile:
    model_hash: str  # hex content hash of the teacher position evaluation model ONNX
    flags: int
    record_floats: int
    positions: list[MsetPosition]

    @property
    def full_sweep(self) -> bool:
        return bool(self.flags & MSET_FLAG_FULL_SWEEP)


def read_mset_flags(path: str | Path) -> int:
    """The file's TargetFileHeader.flags, read without parsing its positions --
    what routing a corpus at file granularity needs."""
    hdr = np.fromfile(str(path), dtype=_FILE_HEADER, count=1)[0]
    if hdr["magic"] != MSET_MAGIC:
        raise ValueError(f"bad .mset magic in {path}")
    return int(hdr["flags"])


def complete_pairs(directory: str | Path) -> list[Path]:
    """The .mset paths in `directory` whose companion .slog is present, sorted.
    A sidecar without its log is inert -- delivery writes the sidecar first, so
    an interrupted one can leave exactly that -- and every consumer needs both
    halves, the targets and the replay the inputs come from."""
    directory = Path(directory)
    return sorted(f for f in directory.glob("*.mset") if f.with_suffix(".slog").exists())


def partition_full_sweep(paths: Iterable[str | Path]) -> tuple[list[Path], list[Path]]:
    """(stratified, full_sweep) partition of .mset paths by header flag: the
    training pairs and the evaluation-only swept pairs, which a corpus holds
    side by side in one store."""
    stratified, swept = [], []
    for path in paths:
        path = Path(path)
        (swept if read_mset_flags(path) & MSET_FLAG_FULL_SWEEP else stratified).append(path)
    return stratified, swept


def read_mset(path: str | Path) -> MsetFile:
    """Parse a .mset file. Raises on a bad magic or a version mismatch (stale
    files must fail loudly)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    hdr = np.frombuffer(buf[: _FILE_HEADER.itemsize].tobytes(), dtype=_FILE_HEADER)[0]
    if hdr["magic"] != MSET_MAGIC:
        raise ValueError(f"bad .mset magic in {path}")
    if hdr["version"] != MSET_VERSION:
        raise ValueError(f".mset version mismatch in {path}: file={hdr['version']}")
    record_floats = int(hdr["record_floats"])
    rec_dtype = _record_dtype(record_floats)

    positions: list[MsetPosition] = []
    off = _FILE_HEADER.itemsize
    for _ in range(int(hdr["num_positions"])):
        ph_bytes = buf[off : off + _POSITION_HEADER.itemsize].tobytes()
        ph = np.frombuffer(ph_bytes, _POSITION_HEADER)[0]
        off += _POSITION_HEADER.itemsize
        k = int(ph["num_candidates"])
        records = np.frombuffer(buf[off : off + k * rec_dtype.itemsize].tobytes(), rec_dtype)
        off += k * rec_dtype.itemsize
        positions.append(
            MsetPosition(
                game_index=int(ph["game_index"]),
                turn_index=int(ph["turn_index"]),
                moves=records["move"],
                targets=records["targets"],
                num_legal_moves=int(ph["num_legal_moves"]),
            )
        )
    if off != len(buf):
        raise ValueError(f"trailing bytes in {path}")
    return MsetFile(
        model_hash=bytes(hdr["model_hash"]).rstrip(b"\x00").decode(),
        flags=int(hdr["flags"]),
        record_floats=record_floats,
        positions=positions,
    )

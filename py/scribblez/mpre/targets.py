"""Pure-numpy reader for .mpt M_pre distillation-target sidecars.

The binary layout is owned by engine/include/scribblez/mpre_target_log.h (one
MpreTargetFileHeader, then per position a MpreTargetPositionHeader followed by
its (Move, targets) records); the dtypes below mirror those packed structs and
are guarded by itemsize checks so a C++ layout change fails loudly here rather
than misparsing. The record's target width comes from the header
(`record_floats`), so the record dtype is built per file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.sim_evidence.sobs import MOVE_DTYPE

MPT_MAGIC = 0x4754504D  # "MPTG"
MPT_VERSION = 1

# Version-1 target floats per candidate, in record order (mover's POV).
TARGET_NAMES_V1 = ("p_win", "p_draw", "p_loss", "sd_mean", "sd_std")

# MpreTargetFileHeader.flags bits (mirrors the .sobs convention).
MPT_FLAG_OPEN_LEAVES = 2

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
        "names": ["game_index", "turn_index", "num_candidates", "reserved"],
        "formats": ["<u4", "<u4", "<u4", "<u4"],
        "offsets": [0, 4, 8, 12],
        "itemsize": 16,
    }
)


def _record_dtype(record_floats: int) -> np.dtype:
    return np.dtype([("move", MOVE_DTYPE), ("targets", "<f4", (record_floats,))])


@dataclass
class MptPosition:
    """One position's sampled candidates and their teacher readouts."""

    game_index: int
    turn_index: int
    moves: np.ndarray  # (K,) MOVE_DTYPE
    targets: np.ndarray  # (K, record_floats) float32


@dataclass
class MptFile:
    model_hash: str  # hex content hash of the teacher M_post ONNX
    flags: int
    record_floats: int
    positions: list[MptPosition]


def read_mpt(path: str | Path) -> MptFile:
    """Parse a .mpt file. Raises on a bad magic or a version mismatch (stale
    files must fail loudly)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    hdr = np.frombuffer(buf[: _FILE_HEADER.itemsize].tobytes(), dtype=_FILE_HEADER)[0]
    if hdr["magic"] != MPT_MAGIC:
        raise ValueError(f"bad .mpt magic in {path}")
    if hdr["version"] != MPT_VERSION:
        raise ValueError(f".mpt version mismatch in {path}: file={hdr['version']}")
    record_floats = int(hdr["record_floats"])
    rec_dtype = _record_dtype(record_floats)

    positions: list[MptPosition] = []
    off = _FILE_HEADER.itemsize
    for _ in range(int(hdr["num_positions"])):
        ph_bytes = buf[off : off + _POSITION_HEADER.itemsize].tobytes()
        ph = np.frombuffer(ph_bytes, _POSITION_HEADER)[0]
        off += _POSITION_HEADER.itemsize
        k = int(ph["num_candidates"])
        records = np.frombuffer(buf[off : off + k * rec_dtype.itemsize].tobytes(), rec_dtype)
        off += k * rec_dtype.itemsize
        positions.append(
            MptPosition(
                game_index=int(ph["game_index"]),
                turn_index=int(ph["turn_index"]),
                moves=records["move"],
                targets=records["targets"],
            )
        )
    if off != len(buf):
        raise ValueError(f"trailing bytes in {path}")
    return MptFile(
        model_hash=bytes(hdr["model_hash"]).rstrip(b"\x00").decode(),
        flags=int(hdr["flags"]),
        record_floats=record_floats,
        positions=positions,
    )

"""Pure-numpy reader for .pmt pre-move value model distillation-target sidecars.

The binary layout is owned by engine/include/scribblez/pre_move_value_target_log.h (one
TargetFileHeader, then per position a TargetPositionHeader followed by
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

PMT_MAGIC = 0x54564D50  # "PMVT"
PMT_VERSION = 1

# Version-1 target floats per candidate, in record order (mover's POV).
TARGET_NAMES_V1 = ("p_win", "p_draw", "p_loss", "sd_mean", "sd_std")

# TargetFileHeader.flags bits (mirrors the .sobs convention).
PMT_FLAG_OPEN_LEAVES = 2

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
class PmtPosition:
    """One position's sampled candidates and their teacher readouts."""

    game_index: int
    turn_index: int
    moves: np.ndarray  # (K,) MOVE_DTYPE
    targets: np.ndarray  # (K, record_floats) float32


@dataclass
class PmtFile:
    model_hash: str  # hex content hash of the teacher M_post ONNX
    flags: int
    record_floats: int
    positions: list[PmtPosition]


def read_pmt(path: str | Path) -> PmtFile:
    """Parse a .pmt file. Raises on a bad magic or a version mismatch (stale
    files must fail loudly)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    hdr = np.frombuffer(buf[: _FILE_HEADER.itemsize].tobytes(), dtype=_FILE_HEADER)[0]
    if hdr["magic"] != PMT_MAGIC:
        raise ValueError(f"bad .pmt magic in {path}")
    if hdr["version"] != PMT_VERSION:
        raise ValueError(f".pmt version mismatch in {path}: file={hdr['version']}")
    record_floats = int(hdr["record_floats"])
    rec_dtype = _record_dtype(record_floats)

    positions: list[PmtPosition] = []
    off = _FILE_HEADER.itemsize
    for _ in range(int(hdr["num_positions"])):
        ph_bytes = buf[off : off + _POSITION_HEADER.itemsize].tobytes()
        ph = np.frombuffer(ph_bytes, _POSITION_HEADER)[0]
        off += _POSITION_HEADER.itemsize
        k = int(ph["num_candidates"])
        records = np.frombuffer(buf[off : off + k * rec_dtype.itemsize].tobytes(), rec_dtype)
        off += k * rec_dtype.itemsize
        positions.append(
            PmtPosition(
                game_index=int(ph["game_index"]),
                turn_index=int(ph["turn_index"]),
                moves=records["move"],
                targets=records["targets"],
            )
        )
    if off != len(buf):
        raise ValueError(f"trailing bytes in {path}")
    return PmtFile(
        model_hash=bytes(hdr["model_hash"]).rstrip(b"\x00").decode(),
        flags=int(hdr["flags"]),
        record_floats=record_floats,
        positions=positions,
    )

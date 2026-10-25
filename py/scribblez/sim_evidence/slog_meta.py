"""Game metadata and moves read directly from .slog bytes: per-game turn counts
and the move played at any turn.

engine/include/data/binary_log.h owns the layout; the dtypes come from the
engine's format-layout document (scribblez.ffi.format_layout), so they cannot
drift from the packed structs. There is no version check here because every
consumer also decodes the same file through the FFI's decode_rows, which
rejects a version mismatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scribblez.ffi import format_layout, struct_dtype
from scribblez.sim_evidence.sobs import MOVE_DTYPE

SLOG_MAGIC = format_layout()["constants"]["slog"]["magic"]

_FILE_HEADER = struct_dtype("SlogFileHeader")
GAME_METADATA = struct_dtype("SlogGameMetadata")
_INITIAL_RACKS = struct_dtype("SlogInitialRacks")
_TURN_BLOB = struct_dtype("SlogTurnBlob")  # Move + the tiles drawn after it


def read_slog_bytes(path: str | Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    magic = int(np.frombuffer(buf[:4].tobytes(), "<u4")[0])
    if magic != SLOG_MAGIC:
        raise ValueError(f"bad .slog magic in {path}")
    return buf


def game_metas(buf: np.ndarray) -> np.ndarray:
    hdr = np.frombuffer(buf[: _FILE_HEADER.itemsize].tobytes(), _FILE_HEADER)[0]
    end = _FILE_HEADER.itemsize + GAME_METADATA.itemsize * int(hdr["num_games"])
    return np.frombuffer(buf[_FILE_HEADER.itemsize : end].tobytes(), GAME_METADATA)


def move_at(buf: np.ndarray, meta: np.void, turn: int) -> np.void:
    """The Move of `turn` within the game described by `meta`."""
    off = int(meta["start_offset"]) + _INITIAL_RACKS.itemsize + _TURN_BLOB.itemsize * turn
    return np.frombuffer(buf[off : off + MOVE_DTYPE.itemsize].tobytes(), MOVE_DTYPE)[0]

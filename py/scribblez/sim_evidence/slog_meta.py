"""Position-level metadata read directly from .slog bytes, for the kill test's
analyses: per-game turn counts and each position's preceding move.

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
from scribblez.sim_evidence.sobs import MOVE_DTYPE, MOVE_PLAY, SobsPosition

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


def position_meta(slog_path: str | Path, positions: list[SobsPosition]) -> dict[str, np.ndarray]:
    """Per-position metadata arrays, aligned with `positions` (read_sobs
    order):

      meta_turn          the pre-move turn index
      meta_remaining     moves left in the original game, a rollout-depth proxy
      meta_opp_unbiased  whether the sim's uniform opponent-rack sampling is
                         exactly the true conditional: the opponent has not
                         moved yet, or just played all 7 tiles, so their rack
                         is a fresh uniform draw from the unseen pool
    """
    buf = read_slog_bytes(slog_path)
    metas = game_metas(buf)
    n = len(positions)
    turn = np.zeros(n, dtype=np.int32)
    remaining = np.zeros(n, dtype=np.int32)
    unbiased = np.zeros(n, dtype=bool)
    for i, pos in enumerate(positions):
        meta = metas[pos.game_index]
        turn[i] = pos.turn_index
        remaining[i] = int(meta["num_turns"]) - pos.turn_index
        if pos.turn_index == 0:
            unbiased[i] = True
        else:
            prev = move_at(buf, meta, pos.turn_index - 1)
            unbiased[i] = int(prev["type"]) == MOVE_PLAY and int(prev["num_played"]) == 7
    return {"meta_turn": turn, "meta_remaining": remaining, "meta_opp_unbiased": unbiased}

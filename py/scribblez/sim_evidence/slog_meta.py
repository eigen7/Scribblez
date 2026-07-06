"""Position-level metadata read directly from .slog bytes.

The binary layout is owned by engine/include/scribblez/binary_log.h; the
dtypes below mirror its packed structs (itemsize-guarded so a C++ layout
change fails loudly here). Only the fields the sim-evidence analyses need are
exposed: per-game turn counts and each position's preceding opponent move.
Version enforcement is not duplicated here -- every consumer also decodes the
same file through the FFI (scribblez_decode_rows), which rejects a version
mismatch loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scribblez.sim_evidence.sobs import MOVE_DTYPE, MOVE_PLAY, SobsPosition

SLOG_MAGIC = 0x474F4C53  # "SLOG"

_FILE_HEADER_SIZE = 16
_INITIAL_RACKS_SIZE = 16
_TURN_BLOB_SIZE = 24  # Move (16) + drawn Rack (8)

GAME_METADATA = np.dtype(
    {
        "names": [
            "start_offset",
            "num_turns",
            "sampled_turn",
            "final_score_p0",
            "final_score_p1",
            "initial_score_p0",
            "initial_score_p1",
            "eligible_begin",
            "eligible_end",
        ],
        "formats": ["<u8", "<u4", "<u2", "<i2", "<i2", "<i2", "<i2", "<u1", "<u1"],
        "offsets": [0, 8, 12, 14, 16, 18, 20, 22, 23],
        "itemsize": 24,
    }
)


def read_slog_bytes(path: str | Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    magic = int(np.frombuffer(buf[:4].tobytes(), "<u4")[0])
    if magic != SLOG_MAGIC:
        raise ValueError(f"bad .slog magic in {path}")
    return buf


def game_metas(buf: np.ndarray) -> np.ndarray:
    num_games = int(np.frombuffer(buf[8:12].tobytes(), "<u4")[0])
    end = _FILE_HEADER_SIZE + GAME_METADATA.itemsize * num_games
    return np.frombuffer(buf[_FILE_HEADER_SIZE:end].tobytes(), GAME_METADATA)


def move_at(buf: np.ndarray, meta: np.void, turn: int) -> np.void:
    """The Move of `turn` within the game described by `meta`."""
    off = int(meta["start_offset"]) + _INITIAL_RACKS_SIZE + _TURN_BLOB_SIZE * turn
    return np.frombuffer(buf[off : off + MOVE_DTYPE.itemsize].tobytes(), MOVE_DTYPE)[0]


def position_meta(slog_path: str | Path, positions: list[SobsPosition]) -> dict[str, np.ndarray]:
    """Per-position analysis metadata, aligned with `positions` (read_sobs
    order, which is also the shard row order):

      meta_turn          the pre-move turn index;
      meta_remaining     moves left in the original game (a rollout-depth
                         proxy);
      meta_opp_unbiased  True iff the sim's uniform opponent-rack sampling is
                         exactly the true conditional at this position: the
                         opponent has not acted yet, or their last move was a
                         bingo (played all 7 tiles), so their rack is a fresh
                         uniform draw from the unseen pool.
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

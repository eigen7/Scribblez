"""Pure-numpy reader for .sobs sim-observation sidecars.

The binary layout is owned by engine/include/data/sim_observation_log.h; the
dtypes and constants below are built from the engine's own format-layout
document (scribblez.ffi.format_layout), whose offsets and sizes come from the
C++ compiler -- so they cannot drift from the packed structs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.ffi import format_layout, struct_dtype

_CONST = format_layout()["constants"]

SOBS_MAGIC = _CONST["sobs"]["magic"]
SOBS_VERSION = _CONST["sobs"]["version"]

# SimObsFileHeader.flags bits. The retired bit marked full-open-rack sims, an
# information condition no consumer supports; readers must reject files
# carrying it.
SOBS_FLAG_RETIRED_OPEN_RACK = _CONST["sobs"]["flag_retired_open_rack"]
SOBS_FLAG_OPEN_LEAVES = _CONST["sobs"]["flag_open_leaves"]

BOARD = _CONST["board_size"]
CELLS = BOARD * BOARD

# MoveType enum values (game/move.h).
MOVE_PLAY = _CONST["move_type"]["play"]
MOVE_EXCHANGE = _CONST["move_type"]["exchange"]
MOVE_PASS = _CONST["move_type"]["pass"]

_GLYPH = _CONST["glyph"]


def glyph_char(code: int) -> str:
    """Decode one Glyph byte (game/glyph.h's code table, served constants):
    played letters show A-Z, designated blanks their lowercased letter
    (matching the engine's move-description convention), and everything else
    -- empty or an unassigned blank -- "?"."""
    if _GLYPH["letter_min"] <= code <= _GLYPH["letter_max"]:
        return chr(ord("A") + code - _GLYPH["letter_min"])
    if _GLYPH["designated_blank_min"] <= code <= _GLYPH["designated_blank_max"]:
        return chr(ord("a") + code - _GLYPH["designated_blank_min"])
    return "?"


_FILE_HEADER = struct_dtype("SobsFileHeader")
_POSITION_HEADER = struct_dtype("SobsPositionHeader")
MOVE_DTYPE = struct_dtype("Move")
RECORD_DTYPE = struct_dtype("SobsRecord")


@dataclass
class SobsPosition:
    """One position's sim evidence: the simmed candidates and their raw
    observations, as recorded by sim_obs_tool."""

    game_index: int
    turn_index: int
    rollouts: int
    base_seed: int
    moves: np.ndarray  # (K,) MOVE_DTYPE
    obs: np.ndarray  # (K,) observation records


def read_sobs_flags(path: str | Path) -> int:
    """The .sobs header's flags word (SOBS_FLAG_* bits) -- e.g. whether the
    sims were generated under the open-leaves information condition. Reads only
    the 16-byte header."""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(_FILE_HEADER.itemsize), dtype=_FILE_HEADER)[0]
    if hdr["magic"] != SOBS_MAGIC:
        raise ValueError(f"bad .sobs magic in {path}")
    return int(hdr["flags"])


def read_sobs(path: str | Path) -> list[SobsPosition]:
    """Parse a .sobs file into per-position evidence. Raises on a bad magic or
    a version mismatch (stale files must fail loudly)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    hdr = np.frombuffer(buf[: _FILE_HEADER.itemsize].tobytes(), dtype=_FILE_HEADER)[0]
    if hdr["magic"] != SOBS_MAGIC:
        raise ValueError(f"bad .sobs magic in {path}")
    if hdr["version"] != SOBS_VERSION:
        raise ValueError(f".sobs version mismatch in {path}: file={hdr['version']}")

    positions: list[SobsPosition] = []
    off = _FILE_HEADER.itemsize
    for _ in range(int(hdr["num_positions"])):
        ph = np.frombuffer(buf[off : off + _POSITION_HEADER.itemsize].tobytes(), _POSITION_HEADER)[
            0
        ]
        off += _POSITION_HEADER.itemsize
        k = int(ph["num_candidates"])
        records = np.frombuffer(buf[off : off + k * RECORD_DTYPE.itemsize].tobytes(), RECORD_DTYPE)
        off += k * RECORD_DTYPE.itemsize
        positions.append(
            SobsPosition(
                game_index=int(ph["game_index"]),
                turn_index=int(ph["turn_index"]),
                rollouts=int(ph["rollouts"]),
                base_seed=int(ph["base_seed"]),
                moves=records["move"],
                obs=records["obs"],
            )
        )
    if off != len(buf):
        raise ValueError(f"trailing bytes in {path}")
    return positions


def move_footprint(move: np.void) -> np.ndarray:
    """The (15, 15) float32 mask of squares a MOVE_DTYPE record places tiles
    on (all zeros for EXCHANGE/PASS) -- the numpy mirror of the C++
    visit_placed_squares walk."""
    plane = np.zeros((BOARD, BOARD), dtype=np.float32)
    if move["type"] != MOVE_PLAY:
        return plane
    start = int(move["start"])
    mask = int(move["square_mask"])
    horizontal = bool(move["horizontal"])
    along = 0
    while mask:
        if mask & 1:
            r, c = (start, along) if horizontal else (along, start)
            plane[r, c] = 1.0
        mask >>= 1
        along += 1
    return plane


# Per-candidate scalar evidence features, in order. Frequencies are over the
# candidate's rollouts; delta moments are in score points (scaled to ~unit
# range); rank is the candidate's HastyBot-equity rank within the evidence set.
EVIDENCE_SCALAR_NAMES = (
    "win_freq",
    "draw_freq",
    "loss_freq",
    "delta_mean_100",
    "delta_std_100",
    "log1p_rollouts",
    "move_score_100",
    "tiles_placed_7",
    "is_play",
    "is_exchange",
    "is_pass",
    "rank_frac",
)
NUM_EVIDENCE_SCALARS = len(EVIDENCE_SCALAR_NAMES)


def evidence_features(pos: SobsPosition, max_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode one position's evidence set into fixed-size model inputs.

    Returns (planes, scalars, mask):
      planes  (max_k, 5, 15, 15) float16 -- the four count planes normalized
              by the rollout count, plus the candidate's own footprint;
      scalars (max_k, NUM_EVIDENCE_SCALARS) float32;
      mask    (max_k,) bool -- True for real candidates, False for padding.
    """
    k = min(len(pos.moves), max_k)
    planes = np.zeros((max_k, 5, BOARD, BOARD), dtype=np.float16)
    scalars = np.zeros((max_k, NUM_EVIDENCE_SCALARS), dtype=np.float32)
    mask = np.zeros(max_k, dtype=bool)

    for i in range(k):
        move, obs = pos.moves[i], pos.obs[i]
        n = max(int(obs["n"]), 1)
        for j, plane_name in enumerate(
            ("opp_next_count", "self_next_count", "opp_win_count", "self_win_count")
        ):
            planes[i, j] = (obs[plane_name].astype(np.float32) / n).reshape(BOARD, BOARD)
        planes[i, 4] = move_footprint(move)

        delta_mean = float(obs["delta_sum"]) / n
        delta_var = max(float(obs["delta_sq_sum"]) / n - delta_mean**2, 0.0)
        mtype = int(move["type"])
        scalars[i] = [
            float(obs["wins"]) / n,
            float(obs["draws"]) / n,
            float(obs["losses"]) / n,
            delta_mean / 100.0,
            delta_var**0.5 / 100.0,
            np.log1p(n) / 8.0,
            float(move["score"]) / 100.0,
            float(move["num_played"]) / 7.0,
            float(mtype == MOVE_PLAY),
            float(mtype == MOVE_EXCHANGE),
            float(mtype == MOVE_PASS),
            i / max(max_k - 1, 1),
        ]
        mask[i] = True
    return planes, scalars, mask

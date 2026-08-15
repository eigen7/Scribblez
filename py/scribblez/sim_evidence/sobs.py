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
# Record order is trajectory order: every prefix of a position's records is a
# valid evidence set (docs/roadmap.md item 4).
SOBS_FLAG_TRAJECTORY = _CONST["sobs"]["flag_trajectory"]

# SobsPositionHeader.flags bits: the position's last record is the
# uniform-exploration draw (absent when the trajectory exhausted the legal
# set first). A tail record contributes proves-best labels at every prefix
# size but never belongs to an evidence prefix itself.
SOBS_POS_FLAG_UNIFORM_TAIL = _CONST["sobs"]["pos_flag_uniform_tail"]

BOARD = _CONST["board_size"]
CELLS = BOARD * BOARD

# MoveType enum values (game/move.h).
MOVE_PLAY = _CONST["move_type"]["play"]
MOVE_EXCHANGE = _CONST["move_type"]["exchange"]
MOVE_PASS = _CONST["move_type"]["pass"]


def glyph_char(code: int) -> str:
    """Decode one Glyph byte: played letters (1..26) show A-Z, designated
    blanks (27..52) their lowercased letter (matching the engine's
    move-description convention), and everything else -- empty (0) or an
    unassigned blank (53) -- "?".

    The code table is game/glyph.h's, deliberately replicated rather than
    served: it is effectively frozen, and unit tests in both languages pin
    the same table (test_format_layout.py; tests_main.cpp's Glyph
    CodeTablePinnedForCrossLanguageReaders). A change on either side must be
    mirrored on the other."""
    if 1 <= code <= 26:
        return chr(ord("A") + code - 1)
    if 27 <= code <= 52:
        return chr(ord("a") + code - 27)
    return "?"


_FILE_HEADER = struct_dtype("SobsFileHeader")
_POSITION_HEADER = struct_dtype("SobsPositionHeader")
MOVE_DTYPE = struct_dtype("Move")
RECORD_DTYPE = struct_dtype("SobsRecord")


@dataclass
class SobsPosition:
    """One position's sim evidence: the simmed candidates and their raw
    observations. In a trajectory file (SOBS_FLAG_TRAJECTORY) the arrays are
    in trajectory order and `flags` carries the SOBS_POS_FLAG_* bits."""

    game_index: int
    turn_index: int
    rollouts: int
    base_seed: int
    num_legal_moves: int
    flags: int
    moves: np.ndarray  # (K,) MOVE_DTYPE
    obs: np.ndarray  # (K,) observation records

    @property
    def has_uniform_tail(self) -> bool:
        return bool(self.flags & SOBS_POS_FLAG_UNIFORM_TAIL)

    def evidence_prefix_sizes(self) -> range:
        """The valid evidence-prefix sizes of a trajectory position: 0 through
        the last proposer pick -- the uniform tail, when present, is a labeled
        candidate only, never evidence."""
        k = len(self.moves)
        return range((k - 1 if self.has_uniform_tail else k) + 1)


def read_sobs_flags(path: str | Path) -> int:
    """The .sobs header's flags word (SOBS_FLAG_* bits) -- e.g. whether the
    sims were generated under the open-leaves information condition. Reads
    only the header."""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(_FILE_HEADER.itemsize), dtype=_FILE_HEADER)[0]
    if hdr["magic"] != SOBS_MAGIC:
        raise ValueError(f"bad .sobs magic in {path}")
    return int(hdr["flags"])


def read_sobs_proposer_hash(path: str | Path) -> str:
    """The hex content hash of the model that proposed the file's candidates,
    "" for the equity-top-K proposer. Reads only the header. Trajectory
    corpora are proposer-versioned the way .mset corpora are teacher-versioned:
    a consumer should refuse to mix hashes."""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(_FILE_HEADER.itemsize), dtype=_FILE_HEADER)[0]
    if hdr["magic"] != SOBS_MAGIC:
        raise ValueError(f"bad .sobs magic in {path}")
    return bytes(hdr["proposer_hash"]).rstrip(b"\x00").decode()


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
                num_legal_moves=int(ph["num_legal_moves"]),
                flags=int(ph["flags"]),
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

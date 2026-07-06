"""Pure-numpy reader for .sobs sim-observation sidecars.

The binary layout is owned by engine/include/scribblez/sim_observation_log.h
(one SimObsFileHeader, then per position a SimObsPositionHeader followed by
its SimObsRecords); the dtypes below mirror those packed structs field for
field and are guarded by itemsize checks so a C++ layout change fails loudly
here rather than misparsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOBS_MAGIC = 0x53424F53  # "SOBS"
SOBS_VERSION = 1

BOARD = 15
CELLS = BOARD * BOARD

# MoveType enum values (engine/include/scribblez/move.h).
MOVE_PLAY = 0
MOVE_EXCHANGE = 1
MOVE_PASS = 2


def glyph_char(code: int) -> str:
    """Decode one Glyph byte (engine/include/scribblez/glyph.h): codes 1-26
    are played letters A-Z, 27 an undesignated blank, 28-52 designated blanks
    (lowercased, matching the engine's move-description convention)."""
    if 1 <= code <= 26:
        return chr(ord("A") + code - 1)
    if 28 <= code <= 52:
        return chr(ord("a") + code - 28)
    return "?"


_FILE_HEADER = np.dtype(
    {
        "names": ["magic", "version", "reserved", "num_positions", "reserved2"],
        "formats": ["<u4", "<u2", "<u2", "<u4", "<u4"],
        "offsets": [0, 4, 6, 8, 12],
        "itemsize": 16,
    }
)

_POSITION_HEADER = np.dtype(
    {
        "names": ["game_index", "turn_index", "num_candidates", "rollouts", "base_seed"],
        "formats": ["<u4", "<u4", "<u4", "<u4", "<u8"],
        "offsets": [0, 4, 8, 12, 16],
        "itemsize": 24,
    }
)

# Move packs into 16 bytes with one padding byte before the 2-aligned
# square_mask (see the sizeof static_assert in move.h).
MOVE_DTYPE = np.dtype(
    {
        "names": ["type", "horizontal", "start", "num_played", "glyphs", "square_mask", "score"],
        "formats": ["<u1", "<u1", "<i1", "<u1", ("<u1", (7,)), "<u2", "<u2"],
        "offsets": [0, 1, 2, 3, 4, 12, 14],
        "itemsize": 16,
    }
)

_OBSERVATION = np.dtype(
    [
        ("n", "<u4"),
        ("wins", "<u4"),
        ("draws", "<u4"),
        ("losses", "<u4"),
        ("delta_sum", "<i8"),
        ("delta_sq_sum", "<i8"),
        ("opp_next_count", "<u2", (CELLS,)),
        ("self_next_count", "<u2", (CELLS,)),
        ("opp_win_count", "<u2", (CELLS,)),
        ("self_win_count", "<u2", (CELLS,)),
    ]
)

RECORD_DTYPE = np.dtype([("move", MOVE_DTYPE), ("obs", _OBSERVATION)])
assert _OBSERVATION.itemsize == 32 + 4 * 2 * CELLS
assert RECORD_DTYPE.itemsize == 16 + _OBSERVATION.itemsize


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

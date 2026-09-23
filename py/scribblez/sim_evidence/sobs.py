"""Pure-numpy reader for .sobs sim-observation sidecars, plus the numpy
encodings of their records used as model inputs.

engine/include/data/sim_observation_log.h owns the layout. The dtypes and
constants here come from the engine's format-layout document
(scribblez.ffi.format_layout), whose offsets and sizes the C++ compiler
produces, so they cannot drift from the packed structs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.ffi import format_layout, struct_dtype
from scribblez.footprint_spatial import (
    ANCHORED,
    MAX_K,
    PASS_CLASS,
    SLOTS_PER_CELL,
    to_slot_planes,
)

_CONST = format_layout()["constants"]

SOBS_MAGIC = _CONST["sobs"]["magic"]
SOBS_VERSION = _CONST["sobs"]["version"]

# SimObsFileHeader.flags bits. The retired bit marked full-open-rack sims, an
# information condition no consumer supports; readers must reject it.
SOBS_FLAG_RETIRED_OPEN_RACK = _CONST["sobs"]["flag_retired_open_rack"]
SOBS_FLAG_OPEN_LEAVES = _CONST["sobs"]["flag_open_leaves"]
# Records are in trajectory order (docs/roadmap.md item 4); see
# SobsPosition.num_evidence for which prefixes are valid evidence sets.
SOBS_FLAG_TRAJECTORY = _CONST["sobs"]["flag_trajectory"]

# SimObsRole values (data/sim_obs_role.h). Anchor and on-policy records may
# serve as evidence; off-policy draws are labels only and never do.
ROLE_ANCHOR = _CONST["sobs"]["role_anchor"]
ROLE_ON_POLICY = _CONST["sobs"]["role_on_policy"]
ROLE_OFF_POLICY = _CONST["sobs"]["role_off_policy"]

BOARD = _CONST["board_size"]
CELLS = BOARD * BOARD

# MoveType enum values (game/move.h).
MOVE_PLAY = _CONST["move_type"]["play"]
MOVE_EXCHANGE = _CONST["move_type"]["exchange"]
MOVE_PASS = _CONST["move_type"]["pass"]


def glyph_char(code: int) -> str:
    """Decode one Glyph byte: letters 1..26 as A-Z, designated blanks 27..52
    as their lowercase letter (the engine's move-notation convention), and
    empty (0) or an unassigned blank (53) as "?".

    This replicates game/glyph.h's code table rather than querying it: the
    table is effectively frozen, and tests in both languages pin it
    (test_format_layout.py; Glyph.CodeTablePinnedForCrossLanguageReaders in
    tests_main.cpp). Change both sides together."""
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
    """One position's simmed candidates with their raw observations and
    evidence roles. In a trajectory file (SOBS_FLAG_TRAJECTORY) they are in
    trajectory order: anchor, on-policy picks, off-policy draws."""

    game_index: int
    turn_index: int
    rollouts: int
    base_seed: int
    num_legal_moves: int
    flags: int
    moves: np.ndarray  # (K,) MOVE_DTYPE
    obs: np.ndarray  # (K,) observation records
    roles: np.ndarray  # (K,) uint8 SimObsRole codes

    @property
    def evidence_mask(self) -> np.ndarray:
        """(K,) bool: False for the labels-only off-policy draws."""
        return self.roles != ROLE_OFF_POLICY

    @property
    def num_evidence(self) -> int:
        """The largest valid evidence prefix: the length of the leading run
        of evidence-eligible records. With records in trajectory order this is
        the eligible count; if an off-policy record ever came earlier, the
        prefix would stop there rather than admit it as evidence."""
        off = np.flatnonzero(self.roles == ROLE_OFF_POLICY)
        return int(off[0]) if off.size else len(self.roles)

    def evidence_prefix_sizes(self) -> range:
        """The valid evidence-prefix sizes, 0 through num_evidence."""
        return range(self.num_evidence + 1)


def _read_header(path: str | Path) -> np.void:
    """The .sobs file header, magic- and version-checked so that header-only
    reads also reject stale files."""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(_FILE_HEADER.itemsize), dtype=_FILE_HEADER)[0]
    if hdr["magic"] != SOBS_MAGIC:
        raise ValueError(f"bad .sobs magic in {path}")
    if hdr["version"] != SOBS_VERSION:
        raise ValueError(f".sobs version mismatch in {path}: file={hdr['version']}")
    return hdr


def read_sobs_flags(path: str | Path) -> int:
    """The header's SOBS_FLAG_* bits, reading only the header."""
    return int(_read_header(path)["flags"])


def read_sobs_proposer_hash(path: str | Path) -> str:
    """The hex content hash of the model that proposed the file's candidates,
    or "" for the equity-top-K proposer. Reads only the header. Consumers
    should refuse to mix proposers, as .mset consumers refuse to mix
    teachers."""
    return bytes(_read_header(path)["proposer_hash"]).rstrip(b"\x00").decode()


def read_sobs_leaf(path: str | Path) -> tuple[str, int]:
    """(leaf model hash, horizon plies) of the file's value-truncated sims,
    or ("", 0) for rollouts played to the end. Reads only the header.
    Truncated observations embed the leaf model's readouts, so consumers
    should refuse to mix files that disagree on either."""
    hdr = _read_header(path)
    return bytes(hdr["leaf_model_hash"]).rstrip(b"\x00").decode(), int(hdr["horizon_plies"])


def read_sobs(path: str | Path) -> list[SobsPosition]:
    """Parse a .sobs file. Raises on bad magic or a version mismatch."""
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
                roles=records["role"],
            )
        )
    if off != len(buf):
        raise ValueError(f"trailing bytes in {path}")
    return positions


def move_footprint(move: np.void) -> np.ndarray:
    """(15, 15) float32 mask of the squares a MOVE_DTYPE record places tiles
    on; all zeros for an exchange or pass. Mirrors the C++
    visit_placed_squares."""
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


def move_footprint_class(move: np.void) -> int:
    """The footprint class of a MOVE_DTYPE record in the natural frame,
    mirroring the C++ footprint_class (training/footprint.h):
    anchor_cell * SLOTS_PER_CELL + slot, where slot 0 is a single tile,
    1..6 horizontal with 2..7 tiles, and 7..12 vertical with 2..7 tiles.
    Exchanges and passes map to PASS_CLASS."""
    if move["type"] != MOVE_PLAY:
        return PASS_CLASS
    k = int(move["num_played"])
    mask = int(move["square_mask"])
    along0 = (mask & -mask).bit_length() - 1  # first placed lane cell
    horizontal = bool(move["horizontal"])
    r, c = (int(move["start"]), along0) if horizontal else (along0, int(move["start"]))
    if k <= 1:
        slot = 0
    else:
        slot = (1 if horizontal else MAX_K) + (k - 2)
    return (r * BOARD + c) * SLOTS_PER_CELL + slot


# SimObservation's footprint histogram fields, in placement-head order (the
# order of evidence_fusion.EVIDENCE_PLANE_NAMES' observed block).
COUNT_HEADS = ("opp_next_count", "self_next_count", "opp_win_count", "self_win_count")


def observed_slot_planes(obs: np.ndarray) -> np.ndarray:
    """(K,) .sobs records -> (K, 4 * SLOTS_PER_CELL, 15, 15) float32: each
    head's footprint histogram as per-slot board channels, divided by the
    candidate's rollout count. The catch-all classes are dropped without
    renormalizing, so rollouts that ended in them just lower the total."""
    k = len(obs)
    n = np.maximum(obs["n"].astype(np.float32), 1.0).reshape(k, 1, 1, 1)
    heads = [to_slot_planes(obs[name].astype(np.float32)) for name in COUNT_HEADS]
    return np.concatenate(heads, axis=-3).reshape(k, -1, BOARD, BOARD) / n


def candidate_slot_planes(moves: np.ndarray) -> np.ndarray:
    """(K,) MOVE_DTYPE records -> (K, SLOTS_PER_CELL, 15, 15) float32: each
    candidate's own footprint as a one-hot at (slot, anchor cell); all zeros
    for an exchange or pass, whose class is a catch-all."""
    planes = np.zeros((len(moves), SLOTS_PER_CELL, BOARD, BOARD), dtype=np.float32)
    for i in range(len(moves)):
        cls = move_footprint_class(moves[i])
        if cls < ANCHORED:
            cell, slot = divmod(cls, SLOTS_PER_CELL)
            planes[i, slot, cell // BOARD, cell % BOARD] = 1.0
    return planes


# The kill test's per-candidate evidence scalars (evidence_features), in order.
# Frequencies are over the candidate's rollouts, and the delta moments are in
# score points / 100. rank_frac is the record's index in the file scaled to
# [0, 1], which is its equity rank for an equity-top-K sidecar. The move
# proposal model uses evidence_fusion.EVIDENCE_SCALAR_NAMES instead.
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
    """Encode one position's evidence set as the kill-test model's fixed-size
    inputs, keeping at most the first `max_k` records.

    Returns (planes, scalars, mask):
      planes  (max_k, 5 * SLOTS_PER_CELL, 15, 15) float16:
              observed_slot_planes, then candidate_slot_planes
      scalars (max_k, NUM_EVIDENCE_SCALARS) float32, EVIDENCE_SCALAR_NAMES order
      mask    (max_k,) bool, False for padding
    """
    k = min(len(pos.moves), max_k)
    planes = np.zeros((max_k, 5 * SLOTS_PER_CELL, BOARD, BOARD), dtype=np.float16)
    scalars = np.zeros((max_k, NUM_EVIDENCE_SCALARS), dtype=np.float32)
    mask = np.zeros(max_k, dtype=bool)
    if k:
        planes[:k, : 4 * SLOTS_PER_CELL] = observed_slot_planes(pos.obs[:k])
        planes[:k, 4 * SLOTS_PER_CELL :] = candidate_slot_planes(pos.moves[:k])

    for i in range(k):
        move, obs = pos.moves[i], pos.obs[i]
        n = max(int(obs["n"]), 1)
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

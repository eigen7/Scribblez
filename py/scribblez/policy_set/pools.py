"""Reading the policy-set model's labeled pools: a .pool sidecar (engine
data/pool_log.h, which owns the format) and the model inputs its positions
replay to.

A pool names its position by (game, turn) in the companion .slog, so the
inputs are rebuilt by replay, as move_set_eval's are: ffi.move_set_positions
returns each position's pre-move board row and its candidates' cross-check
features. The pool lists the candidates, so no legal-move enumeration is
needed. Pools are hidden-leaves only, so the FFI session must be in that arm
(ffi.set_opp_leave_input(False), the default).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez import ffi

POOL = ffi.format_layout()["constants"]["pool"]
SUFFIX = ".pool"


@dataclass
class Pool:
    """One position's pool: its address in the .slog and, per candidate
    (parallel arrays), the move and its labels (engine sim/pool_labels.h)."""

    game: int
    turn: int
    num_legal_moves: int
    moves: np.ndarray  # (M,) Move records
    win_prob: np.ndarray  # (M,) float32
    paired_se: np.ndarray  # (M,) float32, against the leader
    q: np.ndarray  # (M,) float32, P(the move is best); sums to 1
    equity_rank: np.ndarray  # (M,) int32, -1 for a pass outside the ranking
    rollouts: np.ndarray  # (M,) uint16, completed


def read_pools(path: str | Path) -> tuple[np.void, list[Pool]]:
    """A .pool file's header and its pools, in file order. Raises ValueError
    for another format version or a malformed file."""
    hdr_t = ffi.struct_dtype("PoolFileHeader")
    pos_t = ffi.struct_dtype("PoolPositionHeader")
    cand_t = ffi.struct_dtype("PoolCandidate")
    buf = np.fromfile(path, np.uint8)
    if len(buf) < hdr_t.itemsize:
        raise ValueError(f"{path}: truncated header")
    hdr = np.frombuffer(buf, hdr_t, count=1)[0]
    if hdr["magic"] != POOL["magic"] or hdr["version"] != POOL["version"]:
        raise ValueError(
            f"{path}: .pool version {hdr['version']}, but this code reads version {POOL['version']}"
        )
    off, pools = hdr_t.itemsize, []
    for _ in range(int(hdr["num_positions"])):
        ph = np.frombuffer(buf, pos_t, count=1, offset=off)[0]
        off += pos_t.itemsize
        n = int(ph["num_candidates"])
        if off + n * cand_t.itemsize > len(buf):
            raise ValueError(f"{path}: truncated at position {len(pools)}")
        c = np.frombuffer(buf, cand_t, count=n, offset=off)
        off += n * cand_t.itemsize
        pools.append(_pool(ph, c))
    if off != len(buf):
        raise ValueError(f"{path}: {len(buf) - off} bytes past the last position")
    return hdr, pools


def _pool(ph: np.void, c: np.ndarray) -> Pool:
    return Pool(
        game=int(ph["game_index"]),
        turn=int(ph["turn_index"]),
        num_legal_moves=int(ph["num_legal_moves"]),
        moves=c["move"].copy(),
        win_prob=c["win_prob"].copy(),
        paired_se=c["paired_se"].copy(),
        q=c["q"].copy(),
        equity_rank=c["equity_rank"].copy(),
        rollouts=c["rollouts"].copy(),
    )


def pool_inputs(slog: str | Path, pools: list[Pool]) -> dict[str, np.ndarray]:
    """The model inputs of `pools`, positions of the .slog at `slog`:
    ffi.move_set_positions' "rows" (P, row floats), and "cells" and "letters"
    for the candidates of every pool in order (M = sum of the pool sizes)."""
    return ffi.move_set_positions(
        slog,
        np.array([p.game for p in pools]),
        np.array([p.turn for p in pools]),
        np.array([len(p.moves) for p in pools]),
        np.concatenate([p.moves for p in pools]),
    )

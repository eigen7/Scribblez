"""Reading grouped pools: a .group sidecar (engine data/group_log.h, which
owns the format), the search network's store input's training data
(docs/plans/deep_bot.md, "Training data: grouped pools").

A group names its parent by (game, turn) in the companion .slog, with the
mover to move there; it holds the mover's candidates with their equity ranks,
the racks sampled for the opponent, the opponent's pool at each (move, rack)
in the .pool record's form (pools.py's Pool, its rollout values cut to the
maps' count), and the maps of each rack's leading replies, each summed over
the reply's first `sims` rollouts.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez import ffi
from scribblez.policy_set.pools import Pool, parse_pool

GROUP = ffi.format_layout()["constants"]["group"]
SUFFIX = ".group"


def sim_maps_dtype() -> np.dtype:
    """rollout_abi::SimMaps as a structured dtype: each field a float32 array
    of its (rows, cols) shape, a vector or scalar where rows is 1."""
    names, formats = [], []
    for name, rows, cols in GROUP["sim_maps_fields"]:
        names.append(name)
        formats.append((np.float32, (rows, cols) if rows > 1 else (cols,) if cols > 1 else ()))
    return np.dtype({"names": names, "formats": formats})


SIM_MAPS = sim_maps_dtype()


@dataclass
class GroupMap:
    """One leading reply's maps: which (move, rack) pool, which of its
    candidates, over the reply's first `sims` rollouts; `maps` is a SIM_MAPS
    record (sums over those rollouts, the leaf network completing what a
    rollout left open)."""

    move: int
    rack: int
    reply: int
    sims: int
    maps: np.void


@dataclass
class Group:
    """One group: its parent's address in the .slog, the mover's candidates
    (Move records, best equity first) with their ranks in the parent's
    static-equity ranking (int32, -1 for a pass the ranking lacks), the
    opponent's racks as tile counts (R, 27) uint8, the pools (move-major,
    moves x racks) and the maps."""

    game: int
    turn: int
    moves: np.ndarray
    equity_ranks: np.ndarray
    racks: np.ndarray
    pools: list[list[Pool]]  # pools[move][rack]
    maps: list[GroupMap]


def read_groups(path: str | Path) -> tuple[np.void, list[Group]]:
    """A .group file's header and its groups, in file order. Raises
    ValueError for another format version or a malformed file."""
    hdr_t = ffi.struct_dtype("GroupFileHeader")
    buf = np.fromfile(path, np.uint8)
    if len(buf) < hdr_t.itemsize:
        raise ValueError(f"{path}: truncated header")
    hdr = np.frombuffer(buf, hdr_t, count=1)[0]
    if hdr["magic"] != GROUP["magic"] or hdr["version"] != GROUP["version"]:
        raise ValueError(
            f"{path}: .group version {hdr['version']}, but this code reads version "
            f"{GROUP['version']}"
        )
    reader = _Reader(buf, str(path), int(hdr["map_sims"]))
    groups = [reader.group() for _ in range(int(hdr["num_groups"]))]
    if reader.off != len(buf):
        raise ValueError(f"{path}: {len(buf) - reader.off} bytes past the last group")
    return hdr, groups


class _Reader:
    """A cursor over the file's bytes past its header."""

    def __init__(self, buf: np.ndarray, path: str, map_sims: int):
        self.buf = buf
        self.path = path
        self.map_sims = map_sims
        self.off = ffi.struct_dtype("GroupFileHeader").itemsize
        self.groups = 0

    def take(self, dtype: np.dtype, count: int) -> np.ndarray:
        n = dtype.itemsize * count
        if self.off + n > len(self.buf):
            raise ValueError(f"{self.path}: truncated at group {self.groups}")
        out = np.frombuffer(self.buf, dtype, count=count, offset=self.off)
        self.off += n
        return out

    def group(self) -> Group:
        gh = self.take(ffi.struct_dtype("GroupHeader"), 1)[0]
        moves = self.take(ffi.struct_dtype("Move"), int(gh["num_moves"])).copy()
        ranks = self.take(np.dtype(np.int32), len(moves)).copy()
        racks = self.take(ffi.struct_dtype("GroupRack"), int(gh["num_racks"]))["counts"].copy()
        pools = [[self.pool(gh) for _ in range(len(racks))] for _ in range(len(moves))]
        maps = [self.map() for _ in range(int(gh["num_maps"]))]
        self.groups += 1
        return Group(int(gh["game_index"]), int(gh["turn_index"]), moves, ranks, racks, pools, maps)

    def pool(self, gh: np.void) -> Pool:
        ph = self.take(ffi.struct_dtype("PoolPositionHeader"), 1)[0]
        rack = self.take(ffi.struct_dtype("PoolOppRack"), 1)[0]
        c = self.take(ffi.struct_dtype("PoolCandidate"), int(ph["num_candidates"]))
        values = self.take(
            np.dtype(np.float32), int(np.minimum(c["rollouts"], self.map_sims).sum())
        )
        return parse_pool(ph, rack, c, values)

    def map(self) -> GroupMap:
        mh = self.take(ffi.struct_dtype("GroupMapHeader"), 1)[0]
        maps = self.take(SIM_MAPS, 1)[0].copy()
        return GroupMap(int(mh["move"]), int(mh["rack"]), int(mh["reply"]), int(mh["sims"]), maps)

"""Pure-numpy reader for M1a's corpus: the .sprobe probe log, its replayed
game state, and the labels .sobs beside it.

engine/include/data/probe_log.h owns the .sprobe layout and
engine/include/data/probe_replay.h the replayed states; the dtypes come from
the engine's format-layout document, as in scribblez.sim_evidence.sobs. A
file's arrays are kept flat, every position's candidates, records and turns
concatenated, with offsets to address them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.ffi import format_layout, probe_replay, struct_dtype
from scribblez.sim_evidence.sobs import SOBS_FLAG_LABELS, read_sobs, read_sobs_flags

_CONST = format_layout()["constants"]

SPROBE_MAGIC = _CONST["sprobe"]["magic"]
SPROBE_VERSION = _CONST["sprobe"]["version"]
SPROBE_FLAG_FACE_UP_LEAVES = _CONST["sprobe"]["flag_face_up_leaves"]

FILE_HEADER = struct_dtype("ProbeFileHeader")
POSITION_HEADER = struct_dtype("ProbePositionHeader")
CANDIDATE = struct_dtype("ProbeCandidate")
RECORD = struct_dtype("ProbeRecord")
TURN = struct_dtype("SlogTurnBlob")

# A tile code (game/tile.h): letters 0-25, the blank 26, no tile 27.
TILE_KINDS = 27
NO_TILE = 27
RACK_SIZE = 7


@dataclass
class ProbeFile:
    """One .sprobe file. A position's candidates are
    candidates[candidate_start[p]:candidate_start[p + 1]]; candidate c's probe
    i is records[c * probes + i], with c a file-wide candidate index; record r's
    turns are turns[turn_start[r]:turn_start[r + 1]]."""

    path: Path
    flags: int
    horizon_plies: int
    probes: int
    leaf_model_hash: str
    lexicon: str
    positions: np.ndarray  # (P,) POSITION_HEADER
    candidates: np.ndarray  # (C,) CANDIDATE
    records: np.ndarray  # (C * probes,) RECORD
    turns: np.ndarray  # (T,) TURN
    candidate_start: np.ndarray  # (P + 1,) int64
    turn_start: np.ndarray  # (C * probes + 1,) int64

    @property
    def num_positions(self) -> int:
        return len(self.positions)


def read_sprobe(path: str | Path) -> ProbeFile:
    """Parse a .sprobe file. Raises on bad magic, a version mismatch, or a
    size that does not match its headers."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    hdr = np.frombuffer(buf, dtype=FILE_HEADER, count=1)[0]
    if hdr["magic"] != SPROBE_MAGIC:
        raise ValueError(f"bad .sprobe magic in {path}")
    if hdr["version"] != SPROBE_VERSION:
        raise ValueError(f".sprobe version mismatch in {path}: file={hdr['version']}")
    probes = int(hdr["probes"])
    positions, candidates, records, turns = [], [], [], []
    off = FILE_HEADER.itemsize
    for _ in range(int(hdr["num_positions"])):
        ph = np.frombuffer(buf, dtype=POSITION_HEADER, count=1, offset=off)
        off += POSITION_HEADER.itemsize
        k = int(ph[0]["num_candidates"])
        for dtype, count, out in (
            (CANDIDATE, k, candidates),
            (RECORD, k * probes, records),
            (TURN, int(ph[0]["num_turns"]), turns),
        ):
            out.append(np.frombuffer(buf, dtype=dtype, count=count, offset=off))
            off += count * dtype.itemsize
        positions.append(ph)
    if off != len(buf):
        raise ValueError(f"trailing bytes in {path}")
    positions = np.concatenate(positions)
    records = np.concatenate(records)
    return ProbeFile(
        path=Path(path),
        flags=int(hdr["flags"]),
        horizon_plies=int(hdr["horizon_plies"]),
        probes=probes,
        leaf_model_hash=bytes(hdr["leaf_model_hash"]).rstrip(b"\x00").decode(),
        lexicon=bytes(hdr["lexicon"]).rstrip(b"\x00").decode(),
        positions=positions,
        candidates=np.concatenate(candidates),
        records=records,
        turns=np.concatenate(turns),
        candidate_start=_starts(positions["num_candidates"]),
        turn_start=_starts(records["num_turns"]),
    )


def _starts(counts: np.ndarray) -> np.ndarray:
    """Offsets of consecutive runs of the given lengths, plus the total."""
    return np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])


@dataclass
class ProbeReplay:
    """A .sprobe file's replayed game state, in the file's orders
    (engine data/probe_replay.h): one root per position, one state per
    candidate, two deals per record, one state per turn."""

    roots: np.ndarray  # (P,) ProbeRootState
    candidates: np.ndarray  # (C,) ProbeCandidateState
    starts: np.ndarray  # (C * probes,) ProbeStartState
    turns: np.ndarray  # (T,) ProbeTurnState


def replay(probes: ProbeFile) -> ProbeReplay:
    """Replay `probes` against its companion .slog."""
    return ProbeReplay(
        **probe_replay(
            probes.path.with_suffix(".slog"),
            probes.path,
            probes.num_positions,
            len(probes.candidates),
            len(probes.records),
            len(probes.turns),
        )
    )


def read_labels(probes: ProbeFile) -> np.ndarray:
    """The labels .sobs beside `probes`: (C,) observation records, one per
    candidate in the .sprobe's order. Raises unless the file is a labels file
    for exactly these positions and candidates, simmed past the probes."""
    path = probes.path.with_suffix(".sobs")
    if not read_sobs_flags(path) & SOBS_FLAG_LABELS:
        raise ValueError(f"{path} is not a labels file")
    labels = read_sobs(path)
    for p, (ph, lab) in enumerate(zip(probes.positions, labels, strict=True)):
        moves = probes.candidates["move"][probes.candidate_start[p] : probes.candidate_start[p + 1]]
        if (
            lab.game_index != ph["game_index"]
            or lab.turn_index != ph["turn_index"]
            or lab.base_seed != ph["base_seed"] + probes.probes
            or moves.tobytes() != lab.moves.tobytes()
        ):
            raise ValueError(f"{path} position {p} does not match {probes.path}")
    return np.concatenate([lab.obs for lab in labels])


def tile_codes(field: np.ndarray) -> np.ndarray:
    """A replayed rack field (N,) of RACK_SIZE-byte strings -> (N, RACK_SIZE)
    uint8 tile codes, NO_TILE-padded."""
    return np.frombuffer(np.ascontiguousarray(field).tobytes(), dtype=np.uint8).reshape(
        -1, RACK_SIZE
    )


def tile_counts(codes: np.ndarray) -> np.ndarray:
    """(..., RACK_SIZE) tile codes -> (..., TILE_KINDS) per-kind counts."""
    one_hot = codes[..., None] == np.arange(TILE_KINDS, dtype=np.uint8)
    return one_hot.sum(axis=-2, dtype=np.uint8)

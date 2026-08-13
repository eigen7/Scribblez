"""Pure-numpy reader for .mset move set evaluation model distillation-target
sidecars.

The binary layout is owned by
engine/include/training/move_set_eval_target_log.h (one TargetFileHeader, then
per position a TargetPositionHeader followed by its (Move, targets, plane
scales, quantized planes) records). The dtypes and constants below are built
from the engine's own format-layout document (scribblez.ffi.format_layout),
whose offsets and sizes come from the C++ compiler -- so they cannot drift
from the packed structs. The record's target width and plane count come from
each file's header (`record_floats`, `record_planes`), so the record dtype is
built per file.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.ffi import format_layout, struct_dtype
from scribblez.sim_evidence.sobs import MOVE_DTYPE

_CONST = format_layout()["constants"]

MSET_MAGIC = _CONST["mset"]["magic"]
MSET_VERSION = _CONST["mset"]["version"]

# Version-1 target floats per candidate, in record order (mover's POV).
TARGET_NAMES_V1 = tuple(_CONST["mset"]["target_names_v1"])

# A record's placement planes, in plane order: the teacher's four masks at the
# candidate's post-move state (the placement heads of training_targets.h, also
# the SimObservation plane order). Stored absmax-quantized: per plane a float
# scale and PLANE_CELLS bytes, value = byte * scale (scale = plane max / 255).
PLANE_NAMES = tuple(_CONST["placement_head_names"])
PLANE_CELLS = _CONST["mset"]["plane_cells"]

# TargetFileHeader.flags bits (mirrors the .sobs convention). Full-sweep files
# -- capped sweeps of every legal candidate, evaluation-only and held out by
# construction -- carry no placement planes (record_planes == 0).
MSET_FLAG_OPEN_LEAVES = _CONST["mset"]["flag_open_leaves"]
MSET_FLAG_FULL_SWEEP = _CONST["mset"]["flag_full_sweep"]

_FILE_HEADER = struct_dtype("MsetFileHeader")
_POSITION_HEADER = struct_dtype("MsetPositionHeader")


def _record_dtype(record_floats: int, record_planes: int) -> np.dtype:
    fields = [("move", MOVE_DTYPE), ("targets", "<f4", (record_floats,))]
    if record_planes:
        fields += [
            ("plane_scales", "<f4", (record_planes,)),
            ("planes", "u1", (record_planes, PLANE_CELLS)),
        ]
    return np.dtype(fields)


def dequantize_planes(planes: np.ndarray, plane_scales: np.ndarray) -> np.ndarray:
    """(..., P, PLANE_CELLS) uint8 planes with their (..., P) scales -> float32
    probabilities."""
    return planes.astype(np.float32) * plane_scales[..., None]


@dataclass
class MsetPosition:
    """One position's selected candidates and their teacher readouts."""

    game_index: int
    turn_index: int
    moves: np.ndarray  # (K,) MOVE_DTYPE
    targets: np.ndarray  # (K, record_floats) float32
    # Quantized placement planes and their scales (dequantize_planes recovers
    # probabilities); None on a plane-less (full-sweep) file.
    plane_scales: np.ndarray | None  # (K, record_planes) float32
    planes: np.ndarray | None  # (K, record_planes, PLANE_CELLS) uint8
    # Legal moves the position had, recorded for a swept position so that a
    # sweep the generator's cap truncated is visible as num_legal_moves > K.
    # 0 ("not recorded") on every stratified position.
    num_legal_moves: int = 0


@dataclass
class MsetFile:
    model_hash: str  # hex content hash of the teacher position evaluation model ONNX
    flags: int
    record_floats: int
    record_planes: int
    positions: list[MsetPosition]

    @property
    def full_sweep(self) -> bool:
        return bool(self.flags & MSET_FLAG_FULL_SWEEP)


def read_mset_flags(path: str | Path) -> int:
    """The file's TargetFileHeader.flags, read without parsing its positions --
    what routing a corpus at file granularity needs."""
    hdr = np.fromfile(str(path), dtype=_FILE_HEADER, count=1)[0]
    if hdr["magic"] != MSET_MAGIC:
        raise ValueError(f"bad .mset magic in {path}")
    return int(hdr["flags"])


def complete_pairs(directory: str | Path) -> list[Path]:
    """The .mset paths in `directory` whose companion .slog is present, sorted.
    Every consumer needs both halves -- the targets, and the replay the inputs
    are recomputed from -- and a store can hold an orphaned sidecar
    (scribblez.workloads.pair_store documents when)."""
    directory = Path(directory)
    return sorted(f for f in directory.glob("*.mset") if f.with_suffix(".slog").exists())


def partition_full_sweep(paths: Iterable[str | Path]) -> tuple[list[Path], list[Path]]:
    """(stratified, full_sweep) partition of .mset paths by header flag: the
    training pairs and the evaluation-only swept pairs, which a corpus holds
    side by side in one store."""
    stratified, swept = [], []
    for path in paths:
        path = Path(path)
        (swept if read_mset_flags(path) & MSET_FLAG_FULL_SWEEP else stratified).append(path)
    return stratified, swept


def read_mset(path: str | Path) -> MsetFile:
    """Parse a .mset file. Raises on a bad magic or a version mismatch (stale
    files must fail loudly)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    hdr = np.frombuffer(buf[: _FILE_HEADER.itemsize].tobytes(), dtype=_FILE_HEADER)[0]
    if hdr["magic"] != MSET_MAGIC:
        raise ValueError(f"bad .mset magic in {path}")
    if hdr["version"] != MSET_VERSION:
        raise ValueError(f".mset version mismatch in {path}: file={hdr['version']}")
    record_floats = int(hdr["record_floats"])
    record_planes = int(hdr["record_planes"])
    rec_dtype = _record_dtype(record_floats, record_planes)

    positions: list[MsetPosition] = []
    off = _FILE_HEADER.itemsize
    for _ in range(int(hdr["num_positions"])):
        ph_bytes = buf[off : off + _POSITION_HEADER.itemsize].tobytes()
        ph = np.frombuffer(ph_bytes, _POSITION_HEADER)[0]
        off += _POSITION_HEADER.itemsize
        k = int(ph["num_candidates"])
        records = np.frombuffer(buf[off : off + k * rec_dtype.itemsize].tobytes(), rec_dtype)
        off += k * rec_dtype.itemsize
        positions.append(
            MsetPosition(
                game_index=int(ph["game_index"]),
                turn_index=int(ph["turn_index"]),
                moves=records["move"],
                targets=records["targets"],
                plane_scales=records["plane_scales"] if record_planes else None,
                planes=records["planes"] if record_planes else None,
                num_legal_moves=int(ph["num_legal_moves"]),
            )
        )
    if off != len(buf):
        raise ValueError(f"trailing bytes in {path}")
    return MsetFile(
        model_hash=bytes(hdr["model_hash"]).rstrip(b"\x00").decode(),
        flags=int(hdr["flags"]),
        record_floats=record_floats,
        record_planes=record_planes,
        positions=positions,
    )

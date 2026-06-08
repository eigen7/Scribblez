"""ctypes wrapper around libscribblez_ffi.so."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

_LIB: Optional[ctypes.CDLL] = None


def _find_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    # Search order: env var, then relative to repo root.
    env = os.environ.get("SCRIBBLEZ_FFI_LIB")
    if env and os.path.isfile(env):
        _LIB = ctypes.CDLL(env)
        return _LIB

    # Walk up from this file to find the repo root (contains build/).
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        candidate = ancestor / "build" / "engine" / "libscribblez_ffi.so"
        if candidate.is_file():
            _LIB = ctypes.CDLL(str(candidate))
            return _LIB

    raise RuntimeError(
        "Cannot find libscribblez_ffi.so. Set SCRIBBLEZ_FFI_LIB or build the project."
    )


# ---------------------------------------------------------------------------
# ctypes struct mirrors
# ---------------------------------------------------------------------------


class _ScribblezShape(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("dims", ctypes.POINTER(ctypes.c_int)),
        ("num_dims", ctypes.c_int),
        ("target_index", ctypes.c_int),
    ]


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeInfo:
    name: str
    dims: tuple[int, ...]
    target_index: int  # -1 for inputs


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _setup_lib(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restypes for every FFI entry point."""
    lib.scribblez_input_shapes.restype = ctypes.POINTER(_ScribblezShape)
    lib.scribblez_input_shapes.argtypes = []

    lib.scribblez_target_shapes.restype = ctypes.POINTER(_ScribblezShape)
    lib.scribblez_target_shapes.argtypes = []

    lib.scribblez_row_size_floats.restype = ctypes.c_int
    lib.scribblez_row_size_floats.argtypes = []

    lib.scribblez_read_file_header.restype = ctypes.c_int
    lib.scribblez_read_file_header.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]

    lib.scribblez_dl_new.restype = ctypes.c_void_p
    lib.scribblez_dl_new.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int]

    lib.scribblez_dl_delete.restype = None
    lib.scribblez_dl_delete.argtypes = [ctypes.c_void_p]

    lib.scribblez_dl_add_file.restype = None
    lib.scribblez_dl_add_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]

    lib.scribblez_dl_num_positions.restype = ctypes.c_int64
    lib.scribblez_dl_num_positions.argtypes = [ctypes.c_void_p]

    lib.scribblez_dl_load.restype = None
    lib.scribblez_dl_load.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]

    lib.scribblez_dl_epoch_start.restype = ctypes.c_int
    lib.scribblez_dl_epoch_start.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
    ]

    lib.scribblez_dl_load_batch.restype = ctypes.c_int
    lib.scribblez_dl_load_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]

    lib.scribblez_dl_resident_bytes.restype = ctypes.c_int64
    lib.scribblez_dl_resident_bytes.argtypes = [ctypes.c_void_p]


_SETUP_DONE = False


def _lib() -> ctypes.CDLL:
    global _SETUP_DONE
    lib = _find_lib()
    if not _SETUP_DONE:
        _setup_lib(lib)
        _SETUP_DONE = True
    return lib


# ---------------------------------------------------------------------------
# Shape queries
# ---------------------------------------------------------------------------


def _read_shapes(ptr) -> list[ShapeInfo]:
    shapes: list[ShapeInfo] = []
    i = 0
    while ptr[i].name:
        s = ptr[i]
        name = s.name.decode("utf-8")
        dims = tuple(s.dims[j] for j in range(s.num_dims))
        shapes.append(ShapeInfo(name=name, dims=dims, target_index=s.target_index))
        i += 1
    return shapes


def get_input_shapes() -> list[ShapeInfo]:
    return _read_shapes(_lib().scribblez_input_shapes())


def get_target_shapes() -> list[ShapeInfo]:
    return _read_shapes(_lib().scribblez_target_shapes())


def row_size_floats() -> int:
    return _lib().scribblez_row_size_floats()


# ---------------------------------------------------------------------------
# File header reader
# ---------------------------------------------------------------------------


def read_file_header(path: str | Path) -> tuple[int, int]:
    """Read a .slog header. Returns (num_positions, file_size)."""
    num_pos = ctypes.c_int64()
    file_sz = ctypes.c_int64()
    rc = _lib().scribblez_read_file_header(
        str(path).encode("utf-8"),
        ctypes.byref(num_pos),
        ctypes.byref(file_sz),
    )
    if rc != 0:
        raise IOError(f"Failed to read .slog header: {path}")
    return int(num_pos.value), int(file_sz.value)


# ---------------------------------------------------------------------------
# DataLoader wrapper
# ---------------------------------------------------------------------------


class NativeDataLoader:
    """Python wrapper around the C++ DataLoader via FFI."""

    def __init__(
        self,
        memory_budget: int = 256 * 1024 * 1024,
        num_workers: int = 4,
        num_prefetch: int = 2,
    ):
        self._lib = _lib()
        self._handle = self._lib.scribblez_dl_new(memory_budget, num_workers, num_prefetch)
        if not self._handle:
            raise RuntimeError("scribblez_dl_new returned NULL")
        self._row_floats = self._lib.scribblez_row_size_floats()

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.scribblez_dl_delete(self._handle)
            self._handle = None

    def add_file(self, path: str | Path, num_positions: int, file_size: int) -> None:
        self._lib.scribblez_dl_add_file(
            self._handle, str(path).encode("utf-8"), num_positions, file_size
        )

    @property
    def num_positions(self) -> int:
        return int(self._lib.scribblez_dl_num_positions(self._handle))

    @property
    def row_floats(self) -> int:
        return self._row_floats

    def load(
        self,
        start: int,
        stop: int,
        post_move: bool = True,
        apply_symmetry: bool = False,
    ) -> np.ndarray:
        """Load rows [start, stop) into a new numpy array."""
        n_rows = stop - start
        if n_rows <= 0:
            return np.empty((0, self._row_floats), dtype=np.float32)
        buf = np.empty((n_rows, self._row_floats), dtype=np.float32)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.scribblez_dl_load(
            self._handle, start, stop, int(post_move), int(apply_symmetry), ptr
        )
        return buf

    def epoch_start(
        self,
        batch_size: int,
        post_move: bool = True,
        apply_symmetry: bool = True,
        seed: int = 42,
    ) -> int:
        """Begin a new epoch. Returns number of complete batches."""
        self._batch_size = batch_size
        return self._lib.scribblez_dl_epoch_start(
            self._handle, batch_size, int(post_move), int(apply_symmetry), seed
        )

    def load_batch(self) -> np.ndarray | None:
        """Load the next batch. Returns None when epoch is exhausted."""
        buf = np.empty((self._batch_size, self._row_floats), dtype=np.float32)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n = self._lib.scribblez_dl_load_batch(self._handle, ptr)
        if n == 0:
            return None
        if n < self._batch_size:
            return buf[:n]
        return buf

    @property
    def resident_bytes(self) -> int:
        return int(self._lib.scribblez_dl_resident_bytes(self._handle))

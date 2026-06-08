"""ctypes wrapper around libscribblez_ffi.so."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

_FFI_LIB_PATH = '/workspace/repo/build/engine/libscribblez_ffi.so'
_LIB: Optional[ctypes.CDLL] = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    _LIB = ctypes.CDLL(_FFI_LIB_PATH)
    return _LIB


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
    lib = _load_lib()
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

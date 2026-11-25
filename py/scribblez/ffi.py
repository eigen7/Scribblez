"""ctypes bindings to the engine's C API (libscribblez_ffi.so).

The C declarations and their contracts live in
engine/include/serve/scribblez_ffi.h. Dictionary-dependent calls go through one
lazily created, process-wide session bound to DEFAULT_LEXICON.
"""

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.paths import ENGINE_DIR

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

_FFI_LIB_PATH = str(ENGINE_DIR / "libscribblez_ffi.so")
_LIB: ctypes.CDLL | None = None


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


def _pointer(array: np.ndarray, ctype):
    """A ctypes pointer to `array`'s (C-contiguous) data as `ctype`."""
    return array.ctypes.data_as(ctypes.POINTER(ctype))


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


BOARD_CELLS = 15 * 15


@dataclass(frozen=True)
class ShapeInfo:
    name: str
    dims: tuple[int, ...]
    target_index: int  # -1 for inputs


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _setup_lib(lib: ctypes.CDLL):
    """Declare argtypes/restypes for every FFI entry point."""
    lib.scribblez_session_new.restype = ctypes.c_void_p
    lib.scribblez_session_new.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
    ]  # lexicon, opp-rack input

    lib.scribblez_session_delete.restype = None
    lib.scribblez_session_delete.argtypes = [ctypes.c_void_p]

    lib.scribblez_set_worker_count.restype = None
    lib.scribblez_set_worker_count.argtypes = [ctypes.c_int]

    lib.scribblez_task_input_shapes.restype = ctypes.POINTER(_ScribblezShape)
    lib.scribblez_task_input_shapes.argtypes = [ctypes.c_void_p, ctypes.c_int]  # session, task

    lib.scribblez_task_target_shapes.restype = ctypes.POINTER(_ScribblezShape)
    lib.scribblez_task_target_shapes.argtypes = [ctypes.c_void_p, ctypes.c_int]  # session, task

    lib.scribblez_task_row_floats.restype = ctypes.c_int
    lib.scribblez_task_row_floats.argtypes = [ctypes.c_void_p, ctypes.c_int]  # session, task

    lib.scribblez_position_eval_analyze_gcg.restype = ctypes.c_int
    lib.scribblez_position_eval_analyze_gcg.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg_text
        ctypes.c_int,  # opp_leave_input
        ctypes.POINTER(ctypes.c_float),  # out_input
        ctypes.c_int,  # input_cap
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_position_eval_collapse_placement.restype = ctypes.c_int
    lib.scribblez_position_eval_collapse_placement.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg_text
        ctypes.POINTER(ctypes.c_float),  # raw (kPlacementHeads * kFootprintClasses)
        ctypes.c_int,  # raw_cap
        ctypes.POINTER(ctypes.c_float),  # out (kPlacementHeads * 225)
        ctypes.c_int,  # out_cap
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_position_eval_masked_placement.restype = ctypes.c_int
    lib.scribblez_position_eval_masked_placement.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg_text
        ctypes.POINTER(ctypes.c_float),  # raw (kPlacementHeads * kFootprintClasses)
        ctypes.c_int,  # raw_cap
        ctypes.POINTER(ctypes.c_float),  # out (kPlacementHeads * kFootprintClasses)
        ctypes.c_int,  # out_cap
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_position_eval_legal_placement.restype = ctypes.c_int
    lib.scribblez_position_eval_legal_placement.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg_text
        ctypes.POINTER(ctypes.c_float),  # out (kPlacementHeads * 225)
        ctypes.c_int,  # out_cap
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_position_eval_board_json.restype = ctypes.c_int
    lib.scribblez_position_eval_board_json.argtypes = [
        ctypes.c_char_p,  # gcg_text
        ctypes.c_char_p,  # out_json
        ctypes.c_int,  # out_cap
    ]

    lib.scribblez_position_eval_analyze_gcg_leaves.restype = ctypes.c_int
    lib.scribblez_position_eval_analyze_gcg_leaves.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg_text
        ctypes.c_char_p,  # leave_str
        ctypes.c_char_p,  # opp_leave_str (NULL keeps the recorded one)
        ctypes.c_int,  # opp_leave_input
        ctypes.POINTER(ctypes.c_float),  # out_input
        ctypes.c_int,  # input_cap
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_decode_rows.restype = ctypes.c_int
    lib.scribblez_decode_rows.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_int,  # task
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_int,  # post_move
        ctypes.c_int,  # hide_private
        ctypes.POINTER(ctypes.c_float),
    ]

    lib.scribblez_win_table_samples.restype = ctypes.c_int64
    lib.scribblez_win_table_samples.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # path
        ctypes.c_void_p,  # out_samples (WinTableSample records)
        ctypes.c_int64,  # cap
    ]

    lib.scribblez_leave_value_arrays_new.restype = ctypes.c_void_p
    lib.scribblez_leave_value_arrays_new.argtypes = [ctypes.c_void_p]  # session
    lib.scribblez_leave_value_arrays_delete.restype = None
    lib.scribblez_leave_value_arrays_delete.argtypes = [ctypes.c_void_p]
    lib.scribblez_leave_value_arrays_rows.restype = ctypes.c_int64
    lib.scribblez_leave_value_arrays_rows.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.scribblez_leave_value_arrays_copy.restype = ctypes.c_int
    lib.scribblez_leave_value_arrays_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,  # length
        ctypes.POINTER(ctypes.c_uint8),  # out_codes
        ctypes.POINTER(ctypes.c_float),  # out_values
    ]
    lib.scribblez_leave_value_arrays_content_hash.restype = ctypes.c_char_p
    lib.scribblez_leave_value_arrays_content_hash.argtypes = [ctypes.c_void_p]
    lib.scribblez_leave_sampler_query.restype = ctypes.c_int
    lib.scribblez_leave_sampler_query.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),  # params
        ctypes.c_int,  # length
        ctypes.POINTER(ctypes.c_int32),  # leave_pool
        ctypes.POINTER(ctypes.c_int64),  # rows
        ctypes.c_int64,  # n_rows
        ctypes.POINTER(ctypes.c_double),  # quantiles
        ctypes.c_int64,  # n_quantiles
        ctypes.POINTER(ctypes.c_double),  # out_log_z
        ctypes.POINTER(ctypes.c_double),  # out_log_probs
        ctypes.POINTER(ctypes.c_int64),  # out_samples
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_win_table_write.restype = ctypes.c_int
    lib.scribblez_win_table_write.argtypes = [
        ctypes.c_char_p,  # path
        ctypes.POINTER(ctypes.c_float),  # grid
        ctypes.c_char_p,  # header ("key=value" lines)
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_win_table_eval.restype = ctypes.c_int
    lib.scribblez_win_table_eval.argtypes = [
        ctypes.c_char_p,  # path
        ctypes.c_int,  # macondo_format
        ctypes.c_void_p,  # features (WinTableFeature records)
        ctypes.c_int64,  # n
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_read_file_header.restype = ctypes.c_int
    lib.scribblez_read_file_header.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]

    lib.scribblez_dl_new.restype = ctypes.c_void_p
    lib.scribblez_dl_new.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.scribblez_dl_delete.restype = None
    lib.scribblez_dl_delete.argtypes = [ctypes.c_void_p]

    lib.scribblez_dl_add_file.restype = ctypes.c_int
    lib.scribblez_dl_add_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_int,
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
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,  # hide_private_fraction
    ]

    lib.scribblez_dl_load_batch.restype = ctypes.c_int
    lib.scribblez_dl_load_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]

    lib.scribblez_format_layout_json.restype = ctypes.c_char_p
    lib.scribblez_format_layout_json.argtypes = []


_SETUP_DONE = False


def _lib() -> ctypes.CDLL:
    global _SETUP_DONE
    lib = _load_lib()
    if not _SETUP_DONE:
        _setup_lib(lib)
        _SETUP_DONE = True
    return lib


# ---------------------------------------------------------------------------
# Sidecar-format layout
# ---------------------------------------------------------------------------

_FORMAT_LAYOUT: dict | None = None
_STRUCT_DTYPES: dict[str, np.dtype] = {}


def format_layout() -> dict:
    """The engine's description of its binary formats (.slog and its sidecars):
    each struct's field names, offsets and numpy dtype codes, plus magics,
    versions, flag bits, code tables and shared constants. The engine derives it
    from the packed C++ structs, so it cannot drift from them
    (engine/include/data/format_layout.h). Cached per process."""
    global _FORMAT_LAYOUT
    if _FORMAT_LAYOUT is None:
        _FORMAT_LAYOUT = json.loads(_lib().scribblez_format_layout_json().decode())
    return _FORMAT_LAYOUT


def struct_dtype(name: str) -> np.dtype:
    """The numpy dtype of a struct named in format_layout()["structs"]."""
    if name not in _STRUCT_DTYPES:
        desc = format_layout()["structs"][name]
        names, formats, offsets = [], [], []
        for f in desc["fields"]:
            names.append(f["name"])
            fmt = f["dtype"]
            if isinstance(fmt, dict):
                fmt = struct_dtype(fmt["struct"])
            if "shape" in f:
                fmt = (fmt, tuple(f["shape"]))
            formats.append(fmt)
            offsets.append(f["offset"])
        _STRUCT_DTYPES[name] = np.dtype(
            {"names": names, "formats": formats, "offsets": offsets, "itemsize": desc["itemsize"]}
        )
    return _STRUCT_DTYPES[name]


DEFAULT_LEXICON = "NWL23"

_SESSION_HANDLE = None
_OPP_LEAVE_INPUT = False


def set_opp_leave_input(enabled: bool):
    """Choose whether the session's input layout includes the opponent-leave
    counts block (the open-leaves arm, where the opponent's retained leave is
    public but their draws stay hidden).

    Must be called before the session exists; changing it afterwards raises.
    The session's shape queries then report the chosen layout, so downstream
    code need not branch on it.
    """
    global _OPP_LEAVE_INPUT
    if _SESSION_HANDLE is not None and _OPP_LEAVE_INPUT != enabled:
        raise RuntimeError("set_opp_leave_input called after the FFI session was created")
    _OPP_LEAVE_INPUT = enabled


def _session() -> int:
    """The process-wide session handle, created on first use. Creating it
    loads the lexicon's .kwg; a missing lexicon terminates the process."""
    global _SESSION_HANDLE
    if _SESSION_HANDLE is None:
        _SESSION_HANDLE = _lib().scribblez_session_new(
            DEFAULT_LEXICON.encode("utf-8"), int(_OPP_LEAVE_INPUT)
        )
    return _SESSION_HANDLE


def set_worker_count(n: int):
    """Caps the threads the library's parallel steps use, process-wide
    (engine util/work_pool.h); by default every core the process may use."""
    _lib().scribblez_set_worker_count(int(n))


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


# The training tasks a DataLoader or decode_rows decodes, by their engine
# codes (scribblez_ffi.h, scribblez_task_input_shapes).
TASK_CODES = {
    "position_eval": 0,
    "leave_inference": 1,
    "public_leave_inference": 2,
    "deep_bot_leaf_eval": 3,
}


def _task_code(task: str) -> int:
    if task not in TASK_CODES:
        raise ValueError(f"unknown training task {task!r}")
    return TASK_CODES[task]


def task_input_shapes(task: str = "position_eval") -> list[ShapeInfo]:
    """A task's input tensors, in row order. Position evaluation's follow the
    session's input arm."""
    return _read_shapes(_lib().scribblez_task_input_shapes(_session(), _task_code(task)))


def task_target_shapes(task: str = "position_eval") -> list[ShapeInfo]:
    """A task's target tensors, in row order after the inputs."""
    return _read_shapes(_lib().scribblez_task_target_shapes(_session(), _task_code(task)))


def task_row_floats(task: str = "position_eval") -> int:
    """Floats in one training row of `task`: inputs plus targets."""
    return _lib().scribblez_task_row_floats(_session(), _task_code(task))


def task_input_floats(task: str = "position_eval") -> int:
    """Floats in one flat input row of `task`."""
    return sum(int(np.prod(s.dims)) for s in task_input_shapes(task))


def decode_rows(
    path: str | Path,
    game_idx: np.ndarray,
    turn_idx: np.ndarray,
    post_move: bool = True,
    task: str = "position_eval",
    hide_private: bool = False,
) -> np.ndarray:
    """Decode specific training rows of `task` from one .slog, addressed by
    (game_idx[j], turn_idx[j]).

    Rows are encoded exactly as the DataLoader encodes them, without symmetry
    augmentation: (n, task_row_floats(task)) float32. For consumers that pair
    rows with per-position data, such as a held-out set's cached results. The
    leave inference tasks are always pre-move.
    `hide_private` hides a leave inference row's observer's private history
    (the engine's LeaveHistory::hide_private).
    """
    games = np.ascontiguousarray(game_idx, dtype=np.int64)
    turns = np.ascontiguousarray(turn_idx, dtype=np.int64)
    if games.shape != turns.shape or games.ndim != 1:
        raise ValueError(f"game/turn index shapes differ: {games.shape} vs {turns.shape}")
    out = np.empty((len(games), task_row_floats(task)), dtype=np.float32)
    rc = _lib().scribblez_decode_rows(
        _session(),
        _task_code(task),
        str(path).encode("utf-8"),
        games.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        turns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(games),
        int(post_move),
        int(hide_private),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise OSError(f"decode_rows failed (rc={rc}) for {path}; see stderr for the reason")
    return out


def win_table_samples(path: str | Path) -> np.ndarray:
    """The win table's fitting samples from one .slog (engine
    sim/win_table_feature.h): one WinTableSample record per fittable pre-move
    position, a structured array of struct_dtype("WinTableSample")."""
    lib = _lib()
    encoded = str(path).encode("utf-8")
    n = lib.scribblez_win_table_samples(_session(), encoded, None, 0)
    if n == -2:
        raise ValueError(f"{path} is a face-up-leaves corpus; win tables are hidden-leaves only")
    if n < 0:
        raise OSError(f"win_table_samples failed for {path}")
    out = np.empty(n, dtype=struct_dtype("WinTableSample"))
    lib.scribblez_win_table_samples(_session(), encoded, out.ctypes.data, n)
    return out


# The message buffer of an FFI call that reports errors as text.
_ERR_CAP = 1024


class LeaveValueArrays:
    """The engine's LeaveValueArrays (lexicon/leave_value_arrays.h) for the
    session's lexicon: every leave of each length 1..6 with its HastyBot
    value, in the leave values file's order, its tiles as codes (letter *
    COPIES_PER_TILE + copy). Owns the engine object; close() or garbage
    collection frees it."""

    def __init__(self):
        layout = format_layout()["constants"]["leave_value_arrays"]
        self.max_length: int = layout["max_length"]
        self.copies_per_tile: int = layout["copies_per_tile"]
        self.num_codes: int = layout["codes"]
        self._lib = _lib()
        self._handle = self._lib.scribblez_leave_value_arrays_new(_session())

    def __del__(self):
        self.close()

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.scribblez_leave_value_arrays_delete(self._handle)
            self._handle = None

    @property
    def content_hash(self) -> str:
        """Which leave values these are: the .klv2's content hash."""
        return self._lib.scribblez_leave_value_arrays_content_hash(self._handle).decode()

    def of_length(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The length-k leaves: codes (k, R) uint8, one row per tile slot in
        ascending letter order, and values (R,) float32 in points."""
        n = self._lib.scribblez_leave_value_arrays_rows(self._handle, k)
        if n < 0:
            raise ValueError(f"no leaves of length {k}")
        codes = np.empty((k, n), dtype=np.uint8)
        values = np.empty(n, dtype=np.float32)
        self._lib.scribblez_leave_value_arrays_copy(
            self._handle,
            k,
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return codes, values

    def sampler_query(
        self,
        params: np.ndarray,
        length: int,
        leave_pool: np.ndarray,
        rows: np.ndarray = (),
        quantiles: np.ndarray = (),
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """One engine LeaveSampler query (belief/leave_sampler.h) over the
        length-`length` leaves: `params` the 29 exported floats, `leave_pool`
        27 tile counts. Returns log Z, the log probability of each row index in
        `rows`, and the row drawn at each quantile in `quantiles`."""
        p = np.ascontiguousarray(params, dtype=np.float32)
        pool = np.ascontiguousarray(leave_pool, dtype=np.int32)
        r = np.ascontiguousarray(rows, dtype=np.int64)
        q = np.ascontiguousarray(quantiles, dtype=np.float64)
        log_z = ctypes.c_double()
        log_probs = np.empty(len(r), dtype=np.float64)
        samples = np.empty(len(q), dtype=np.int64)
        err = ctypes.create_string_buffer(_ERR_CAP)
        rc = self._lib.scribblez_leave_sampler_query(
            self._handle,
            p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            length,
            pool.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            r.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(r),
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(q),
            ctypes.byref(log_z),
            log_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            err,
            _ERR_CAP,
        )
        if rc != 0:
            raise ValueError(err.value.decode())
        return log_z.value, log_probs, samples


def win_table_write(path: str | Path, grid: np.ndarray, header: dict[str, object]):
    """Write a win table (engine sim/win_pct_table.h, which owns the format)
    from a fitted `grid`, (rows, cols) in file order, with `header` as its
    provenance lines. Raises ValueError with the engine's message."""
    g = np.ascontiguousarray(grid, dtype=np.float32)
    lines = "".join(f"{k}={v}\n" for k, v in header.items()).encode("utf-8")
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = _lib().scribblez_win_table_write(
        str(path).encode("utf-8"),
        g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        lines,
        err,
        _ERR_CAP,
    )
    if rc != 0:
        raise ValueError(err.value.decode())


def win_table_eval(
    path: str | Path, features: np.ndarray, macondo_format: bool = False
) -> np.ndarray:
    """P(win) for each WinTableFeature record in `features` under the table at
    `path` (Macondo's winpct.csv if `macondo_format`), as float32. Raises
    ValueError with the engine's message, e.g. for another format version."""
    f = np.ascontiguousarray(features, dtype=struct_dtype("WinTableFeature"))
    out = np.empty(len(f), dtype=np.float32)
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = _lib().scribblez_win_table_eval(
        str(path).encode("utf-8"),
        int(macondo_format),
        f.ctypes.data,
        len(f),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        err,
        _ERR_CAP,
    )
    if rc != 0:
        raise ValueError(err.value.decode())
    return out


@dataclass(frozen=True)
class InputArm:
    """An input-encoding arm plus the input widths a model built for it
    declares. The engine refuses to encode a width the arm does not produce,
    so a model from an older encoding fails at the encode call instead of
    being fed a misaligned row. For a served ONNX model these come from its
    metadata and input shapes."""

    opp_leave_input: bool
    spatial_planes: int
    scalar_size: int

    @property
    def input_floats(self) -> int:
        return self.spatial_planes * BOARD_CELLS + self.scalar_size

    def split(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a flat row into (spatial (P, 15, 15), scalar (S,))."""
        spatial = flat[: self.spatial_planes * BOARD_CELLS].reshape(self.spatial_planes, 15, 15)
        return spatial, flat[self.spatial_planes * BOARD_CELLS :]


def session_input_arm() -> InputArm:
    """The arm the process-wide session encodes under, with its widths."""
    shapes = {s.name: s.dims for s in task_input_shapes()}
    return InputArm(
        _OPP_LEAVE_INPUT,
        shapes["input_spatial"][0],
        shapes["input_scalar"][0],
    )


def analyze_position_eval_gcg(gcg_text: str, arm: InputArm) -> np.ndarray:
    """Encode an eval-set GCG for the position evaluation model under `arm`.

    The position is the board after the final recorded move, from the POV of
    the player who made it, with that player's leave as the rack; this is the
    seat the Monte-Carlo ground truth scores. Only the session's dictionary is
    used, not its arm. Raises ValueError when the arm does not produce the
    declared widths, OSError on a parse error or a non-PLAY final move.
    """
    lib = _lib()
    inp = np.zeros(arm.input_floats, dtype=np.float32)
    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    err = ctypes.create_string_buffer(256)
    n = lib.scribblez_position_eval_analyze_gcg(
        _session(),
        gcg_text.encode("utf-8"),
        int(arm.opp_leave_input),
        inp_ptr,
        len(inp),
        err,
        len(err),
    )
    if n < 0:
        _raise_analysis_error(err.value.decode("utf-8"))
    return inp


def collapse_position_eval_placement(gcg_text: str, raw: np.ndarray) -> np.ndarray:
    """Collapse the four placement heads' raw footprint logits for an eval-set
    GCG into (4, 15, 15) per-cell marginals.

    The plays heads give Pr[the next move covers the cell]; the win heads give
    Pr[covers the cell and that seat wins]. Uses the same legality mask, softmax
    and scatter the engine's training targets use. `raw` is
    (4, num_classes) or flat. Raises OSError on a parse error or a non-PLAY
    final move.
    """
    consts = format_layout()["constants"]
    heads = len(consts["placement_head_names"])
    side = consts["footprint"]["side"]
    lib = _lib()
    raw32 = np.ascontiguousarray(raw, dtype=np.float32).reshape(-1)
    out = np.zeros(heads * side * side, dtype=np.float32)
    err = ctypes.create_string_buffer(256)
    n = lib.scribblez_position_eval_collapse_placement(
        _session(),
        gcg_text.encode("utf-8"),
        raw32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(raw32),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(out),
        err,
        len(err),
    )
    if n < 0:
        _raise_analysis_error(err.value.decode("utf-8"))
    return out.reshape(heads, side, side)


def masked_position_eval_placement(gcg_text: str, raw: np.ndarray) -> np.ndarray:
    """The four placement heads' footprint distributions for an eval-set GCG
    after the engine's legality mask and masked softmax: (4, num_classes), with
    illegal footprints at zero.

    This is the exact masked distribution over footprints, before collapsing to cells,
    exposed for measuring its sparsity. `raw` is (4, num_classes) or flat.
    Raises OSError on a parse error or a non-PLAY final move.
    """
    consts = format_layout()["constants"]
    heads = len(consts["placement_head_names"])
    classes = consts["footprint"]["num_classes"]
    lib = _lib()
    raw32 = np.ascontiguousarray(raw, dtype=np.float32).reshape(-1)
    out = np.zeros(heads * classes, dtype=np.float32)
    err = ctypes.create_string_buffer(256)
    n = lib.scribblez_position_eval_masked_placement(
        _session(),
        gcg_text.encode("utf-8"),
        raw32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(raw32),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(out),
        err,
        len(err),
    )
    if n < 0:
        _raise_analysis_error(err.value.decode("utf-8"))
    return out.reshape(heads, classes)


def legal_position_eval_placement(gcg_text: str) -> np.ndarray:
    """(4, 15, 15) bool per placement head: True where some legal anchored
    footprint covers the cell, for an eval-set GCG. Raises OSError on a parse
    error or a non-PLAY final move.
    """
    consts = format_layout()["constants"]
    heads = len(consts["placement_head_names"])
    side = consts["footprint"]["side"]
    lib = _lib()
    out = np.zeros(heads * side * side, dtype=np.float32)
    err = ctypes.create_string_buffer(256)
    n = lib.scribblez_position_eval_legal_placement(
        _session(),
        gcg_text.encode("utf-8"),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(out),
        err,
        len(err),
    )
    if n < 0:
        _raise_analysis_error(err.value.decode("utf-8"))
    return out.reshape(heads, side, side).astype(bool)


def _raise_analysis_error(reason: str):
    """Map the engine's error message to an exception type. The engine returns
    -1 both for a width the arm does not encode (the model is wrong: ValueError)
    and for a GCG that does not parse (the input is wrong: OSError)."""
    if "the arm encodes" in reason:
        raise ValueError(reason)
    raise OSError(reason or "GCG parse error or non-PLAY final move")


def analyze_position_eval_gcg_leaves(
    gcg_text: str, leave: str, opp_leave: str | None, arm: InputArm
) -> np.ndarray:
    """analyze_position_eval_gcg with substitute leaves, for the dashboard's
    what-if: `leave` for the POV player and, unless None, `opp_leave` for the
    opponent (read only by the opponent-leave arm). '?' is a blank.

    Only the rack, opponent-leave and unseen-pool features change. Raises
    ValueError with a readable reason on a bad or unavailable leave or a width
    the arm does not encode.
    """
    lib = _lib()
    inp = np.zeros(arm.input_floats, dtype=np.float32)
    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    err = ctypes.create_string_buffer(256)
    n = lib.scribblez_position_eval_analyze_gcg_leaves(
        _session(),
        gcg_text.encode("utf-8"),
        leave.encode("utf-8"),
        None if opp_leave is None else opp_leave.encode("utf-8"),
        int(arm.opp_leave_input),
        inp_ptr,
        len(inp),
        err,
        len(err),
    )
    if n < 0:
        raise ValueError(err.value.decode("utf-8") or "invalid alternate leave")
    return inp


def position_eval_board_json(gcg_text: str) -> dict:
    """The web board bundle (board, bonuses, rack, tile_scores, start_player,
    last_move, opp_leave) for the position analyze_position_eval_gcg encodes.

    The rack is the POV player's leave; opp_leave is what the opponent's last
    move kept ('?' is a blank). Raises OSError on a parse error or a non-PLAY
    final move.
    """
    lib = _lib()
    fn = lib.scribblez_position_eval_board_json
    encoded = gcg_text.encode("utf-8")
    cap = 1 << 16
    out = ctypes.create_string_buffer(cap)
    n = fn(encoded, out, cap)
    if n < 0:
        raise OSError("position_eval_board_json failed (GCG parse error or non-PLAY final move)")
    if n >= cap:  # JSON was truncated; retry once at the exact size
        cap = n + 1
        out = ctypes.create_string_buffer(cap)
        n = fn(encoded, out, cap)
    return json.loads(out.value.decode("utf-8"))


# First-try capacity of the legal-move buffer; a larger position retries at the
# reported count.
_GCG_MOVES_FIRST_CAP = 4096


# ---------------------------------------------------------------------------
# File header reader
# ---------------------------------------------------------------------------


def read_file_header(path: str | Path) -> tuple[int, int]:
    """Read a .slog header. Returns (num_games, file_size)."""
    num_games = ctypes.c_int64()
    file_sz = ctypes.c_int64()
    rc = _lib().scribblez_read_file_header(
        str(path).encode("utf-8"),
        ctypes.byref(num_games),
        ctypes.byref(file_sz),
    )
    if rc != 0:
        raise OSError(f"Failed to read .slog header: {path}")
    return int(num_games.value), int(file_sz.value)


# ---------------------------------------------------------------------------
# DataLoader wrapper
# ---------------------------------------------------------------------------


class NativeDataLoader:
    """The C++ DataLoader (engine/include/data/data_loader.h).

    `task` fixes which training row it decodes (TASK_CODES), over each game's
    training-eligible turns.
    """

    def __init__(
        self,
        memory_budget: int = 256 * 1024 * 1024,
        num_workers: int = 4,
        num_prefetch: int = 2,
        task: str = "position_eval",
    ):
        self._lib = _lib()
        self._handle = self._lib.scribblez_dl_new(
            _session(), memory_budget, num_workers, num_prefetch, _task_code(task)
        )
        self._row_floats = task_row_floats(task)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.scribblez_dl_delete(self._handle)
            self._handle = None

    def add_file(self, path: str | Path, num_games: int, file_size: int):
        """Register a .slog. Raises OSError if the task's rows read a .forgone
        sidecar that is missing or does not match the file."""
        err = ctypes.create_string_buffer(1024)
        rc = self._lib.scribblez_dl_add_file(
            self._handle, str(path).encode("utf-8"), num_games, file_size, err, len(err)
        )
        if rc != 0:
            raise OSError(err.value.decode("utf-8", "replace"))

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
        turns_per_game: int = 0,
        epoch_index: int = 0,
        hide_private_fraction: float = 0.0,
    ) -> int:
        """Begin an epoch; returns the number of complete batches. See
        SlogDataset.iter_batches for turns_per_game, epoch_index and
        hide_private_fraction."""
        self._batch_size = batch_size
        return self._lib.scribblez_dl_epoch_start(
            self._handle,
            batch_size,
            int(post_move),
            int(apply_symmetry),
            seed,
            turns_per_game,
            epoch_index,
            hide_private_fraction,
        )

    def load_batch(self) -> np.ndarray | None:
        """The next batch as a fresh array, or None when the epoch is done.

        Raises OSError if a registered .slog became unreadable mid-epoch, e.g.
        deleted or truncated under the reader; the native loader names the file
        on stderr.
        """
        buf = np.empty((self._batch_size, self._row_floats), dtype=np.float32)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n = self._lib.scribblez_dl_load_batch(self._handle, ptr)
        if n < 0:
            raise OSError(
                "DataLoader.load_batch failed: a registered .slog became unreadable "
                "(deleted or truncated under the reader); see stderr for the file"
            )
        if n == 0:
            return None
        if n < self._batch_size:
            return buf[:n]
        return buf

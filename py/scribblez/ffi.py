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


class _PolicySetOutputs(ctypes.Structure):
    """ScribblezPolicySetOutputs (scribblez_ffi.h)."""

    _fields_ = [
        ("rows", ctypes.POINTER(ctypes.c_float)),
        ("opp_rack", ctypes.POINTER(ctypes.c_float)),
        ("letters", ctypes.POINTER(ctypes.c_int32)),
        ("blanks", ctypes.POINTER(ctypes.c_uint8)),
        ("squares", ctypes.POINTER(ctypes.c_int32)),
        ("tile_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("scalars", ctypes.POINTER(ctypes.c_float)),
        ("cross_cells", ctypes.POINTER(ctypes.c_int32)),
        ("cross_letters", ctypes.POINTER(ctypes.c_uint8)),
        ("equity", ctypes.POINTER(ctypes.c_float)),
    ]


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

    lib.scribblez_max_move_per_lane_analyze_gcg.restype = ctypes.c_int
    lib.scribblez_max_move_per_lane_analyze_gcg.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg_text
        ctypes.c_char_p,  # out_json
        ctypes.c_int,  # out_cap
        ctypes.POINTER(ctypes.c_float),  # out_input
    ]

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

    lib.scribblez_move_set_encode_moves.restype = None
    lib.scribblez_move_set_encode_moves.argtypes = [
        ctypes.c_void_p,  # moves (packed 16-byte Move records)
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),  # pre_move_score_diffs
        ctypes.POINTER(ctypes.c_int32),  # letters
        ctypes.POINTER(ctypes.c_uint8),  # blanks
        ctypes.POINTER(ctypes.c_int32),  # squares
        ctypes.POINTER(ctypes.c_uint8),  # tile_mask
        ctypes.POINTER(ctypes.c_float),  # scalars
    ]

    lib.scribblez_move_set_move_dims.restype = None
    lib.scribblez_move_set_move_dims.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]

    lib.scribblez_move_set_positions.restype = ctypes.c_int
    lib.scribblez_move_set_positions.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # .slog path
        ctypes.POINTER(ctypes.c_int64),  # game_idx
        ctypes.POINTER(ctypes.c_int64),  # turn_idx
        ctypes.POINTER(ctypes.c_int64),  # move_counts
        ctypes.c_int64,  # n_positions
        ctypes.c_void_p,  # moves
        ctypes.POINTER(ctypes.c_float),  # out_rows
        ctypes.POINTER(ctypes.c_int32),  # out_cells
        ctypes.POINTER(ctypes.c_uint8),  # out_letters
        ctypes.POINTER(ctypes.c_uint8),  # transposes, nullable with out_moves
        ctypes.c_void_p,  # out_moves, nullable with transposes
    ]

    lib.scribblez_policy_set_positions.restype = ctypes.c_int
    lib.scribblez_policy_set_positions.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # .slog path
        ctypes.POINTER(ctypes.c_int64),  # game_idx
        ctypes.POINTER(ctypes.c_int64),  # turn_idx
        ctypes.POINTER(ctypes.c_int64),  # move_counts
        ctypes.c_int64,  # n_positions
        ctypes.c_void_p,  # moves
        ctypes.c_void_p,  # opp_racks (PoolOppRack[n_positions])
        ctypes.POINTER(ctypes.c_uint8),  # transposes, nullable
        ctypes.POINTER(_PolicySetOutputs),  # out
    ]

    lib.scribblez_gcg_cross_checks.restype = ctypes.c_int
    lib.scribblez_gcg_cross_checks.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg text
        ctypes.c_int,  # open_leaves
        ctypes.c_void_p,  # moves
        ctypes.c_int64,  # n
        ctypes.POINTER(ctypes.c_int32),  # out_cells
        ctypes.POINTER(ctypes.c_uint8),  # out_letters
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_move_set_cross_slots.restype = ctypes.c_int32
    lib.scribblez_move_set_cross_slots.argtypes = []

    lib.scribblez_move_set_encoding_version.restype = ctypes.c_int32
    lib.scribblez_move_set_encoding_version.argtypes = []

    lib.scribblez_score_diff_input_layout.restype = None
    lib.scribblez_score_diff_input_layout.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.POINTER(ctypes.c_int32),  # scalar_index
        ctypes.POINTER(ctypes.c_float),  # scale
    ]

    lib.scribblez_gcg_sim_evidence.restype = ctypes.c_int
    lib.scribblez_gcg_sim_evidence.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_int,  # open_leaves
        ctypes.c_int,  # solve_endgames
        ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_int),
    ]

    lib.scribblez_encode_candidate_rows.restype = ctypes.c_int
    lib.scribblez_encode_candidate_rows.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # .slog path
        ctypes.POINTER(ctypes.c_int64),  # game_idx
        ctypes.POINTER(ctypes.c_int64),  # turn_idx
        ctypes.POINTER(ctypes.c_int64),  # move_counts
        ctypes.c_int64,  # n_positions
        ctypes.c_void_p,  # moves
        ctypes.POINTER(ctypes.c_float),  # out
    ]

    lib.scribblez_probe_replay.restype = ctypes.c_int
    lib.scribblez_probe_replay.argtypes = [
        ctypes.c_char_p,  # .slog path
        ctypes.c_char_p,  # .sprobe path
        ctypes.c_int64,  # n_positions
        ctypes.c_int64,  # n_candidates
        ctypes.c_int64,  # n_records
        ctypes.c_int64,  # n_turns
        ctypes.c_void_p,  # out_roots
        ctypes.c_void_p,  # out_candidates
        ctypes.c_void_p,  # out_starts
        ctypes.c_void_p,  # out_turns
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_reply_blocking.restype = ctypes.c_int
    lib.scribblez_reply_blocking.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # .slog path
        ctypes.c_char_p,  # .sprobe path
        ctypes.c_int64,  # n_records
        ctypes.c_int,  # stride
        ctypes.POINTER(ctypes.c_uint8),  # out
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

    lib.scribblez_gcg_position_inputs.restype = ctypes.c_int
    lib.scribblez_gcg_position_inputs.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg text
        ctypes.c_int,  # opp_leave_input
        ctypes.POINTER(ctypes.c_float),  # out_input
        ctypes.c_int,  # input_cap
        ctypes.POINTER(ctypes.c_int32),  # out_score_diff
        ctypes.c_void_p,  # out_moves
        ctypes.c_int,  # moves_cap
        ctypes.c_char_p,  # out_err
        ctypes.c_int,  # err_cap
    ]

    lib.scribblez_gcg_position_board_json.restype = ctypes.c_int
    lib.scribblez_gcg_position_board_json.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]


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
    """The engine's description of its binary formats (.slog / .sobs / .mset):
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
    "max_move_per_lane": 1,
    "leave_inference": 2,
    "public_leave_inference": 3,
    "deep_bot_leaf_eval": 4,
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
    rows with per-position data such as .sobs sim observations. The
    max-move-per-lane and leave inference tasks are always pre-move.
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


def move_encoding_dims() -> tuple[int, int, int, int]:
    """The move set model's move-encoder dims (engine
    training/move_set_encoder.h): (max_placed, num_scalars, letter_vocab,
    cells)."""
    max_placed = ctypes.c_int32()
    num_scalars = ctypes.c_int32()
    letter_vocab = ctypes.c_int32()
    cells = ctypes.c_int32()
    _lib().scribblez_move_set_move_dims(
        ctypes.byref(max_placed),
        ctypes.byref(num_scalars),
        ctypes.byref(letter_vocab),
        ctypes.byref(cells),
    )
    return max_placed.value, num_scalars.value, letter_vocab.value, cells.value


def move_encoding_version() -> int:
    """The engine's move-encoding version (training/move_set_encoder.h
    kMoveEncodingVersion), recorded in move set checkpoints and ONNX metadata so
    a model is never run under a different encoding than it was trained on."""
    return int(_lib().scribblez_move_set_encoding_version())


def score_diff_input_layout() -> tuple[int, float]:
    """(scalar_index, scale) such that input_scalar[scalar_index] * scale is
    the position's score differential in points, under the session's arm."""
    index = ctypes.c_int32()
    scale = ctypes.c_float()
    _lib().scribblez_score_diff_input_layout(_session(), ctypes.byref(index), ctypes.byref(scale))
    return index.value, scale.value


def encode_moves(moves: np.ndarray, pre_move_score_diffs: np.ndarray) -> dict[str, np.ndarray]:
    """Encode (M,) Move records (sim_evidence.sobs.MOVE_DTYPE) into the move
    set model's move-encoder inputs, using the same engine encoder the move set
    agent uses.

    `pre_move_score_diffs` (M,) is the mover's score advantage in points before
    each move; the encoder derives the post-move differential from it. Returns
    letters and squares (M, max_placed) int64, blanks and tile_mask
    (M, max_placed) bool, scalars (M, num_scalars) float32.
    """
    from scribblez.sim_evidence.sobs import MOVE_DTYPE

    moves = np.ascontiguousarray(moves, dtype=MOVE_DTYPE)
    pre_diffs = np.ascontiguousarray(pre_move_score_diffs, dtype=np.int32)
    n = len(moves)
    if len(pre_diffs) != n:
        raise ValueError(f"pre_move_score_diffs length {len(pre_diffs)} != moves length {n}")
    max_placed, num_scalars, _, _ = move_encoding_dims()
    letters = np.zeros((n, max_placed), dtype=np.int32)
    blanks = np.zeros((n, max_placed), dtype=np.uint8)
    squares = np.zeros((n, max_placed), dtype=np.int32)
    tile_mask = np.zeros((n, max_placed), dtype=np.uint8)
    scalars = np.zeros((n, num_scalars), dtype=np.float32)
    if n:
        _lib().scribblez_move_set_encode_moves(
            moves.ctypes.data_as(ctypes.c_void_p),
            n,
            pre_diffs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            letters.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            blanks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            squares.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            tile_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            scalars.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
    return {
        "letters": letters.astype(np.int64),
        "blanks": blanks.astype(bool),
        "squares": squares.astype(np.int64),
        "tile_mask": tile_mask.astype(bool),
        "scalars": scalars,
    }


def move_cross_slots() -> int:
    """Cross-check slots per move (engine move_set_encoder.h kMoveCrossSlots);
    cross_letters carries 26 flags per slot."""
    return _lib().scribblez_move_set_cross_slots()


def _cross_check_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    slots = move_cross_slots()
    return np.zeros((n, slots), dtype=np.int32), np.zeros((n, slots * 26), dtype=np.uint8)


def _cross_check_result(cells: np.ndarray, letters: np.ndarray) -> dict[str, np.ndarray]:
    return {"cells": cells.astype(np.int64), "letters": letters}


def move_set_positions(
    path: str | Path,
    game_idx: np.ndarray,
    turn_idx: np.ndarray,
    move_counts: np.ndarray,
    moves: np.ndarray,
    transposes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """A move set batch's inputs from one .slog, one replay per position: each
    position's pre-move training row, and the cross-check features of its
    candidate moves on that board in the move set model's layout (engine
    move_set_encoder.h encode_move_cross_checks).

    Position j is the pre-move decision point (game_idx[j], turn_idx[j]) of the
    .slog at `path`; its candidates are the next move_counts[j] records of the
    (M,) MOVE_DTYPE array `moves`, M = sum(move_counts). Returns "rows"
    (P, task_row_floats()) float32, as decode_rows(post_move=False) decodes
    them; "cells" (M, slots) int64 -- 1 + axis * 225 + square per entry, 0 in
    empty slots; and "letters" (M, slots * 26) uint8, each entry's post-move
    legal letters. With `transposes` (P,) bool, position j where it is set is
    encoded as the transposed position (the board's diagonal symmetry),
    cross-checks included, and "moves" (M,) MOVE_DTYPE returns every candidate in its
    position's frame: transposed there, as given elsewhere.
    """
    from scribblez.sim_evidence.sobs import MOVE_DTYPE

    games = np.ascontiguousarray(game_idx, dtype=np.int64)
    turns = np.ascontiguousarray(turn_idx, dtype=np.int64)
    counts = np.ascontiguousarray(move_counts, dtype=np.int64)
    moves = np.ascontiguousarray(moves, dtype=MOVE_DTYPE)
    if not (games.shape == turns.shape == counts.shape) or games.ndim != 1:
        raise ValueError(f"per-position shapes differ: {games.shape} {turns.shape} {counts.shape}")
    if counts.sum() != len(moves):
        raise ValueError(f"move_counts sum {counts.sum()} != moves length {len(moves)}")
    rows = np.empty((len(games), task_row_floats()), dtype=np.float32)
    cells, letters = _cross_check_arrays(len(moves))
    flips = None if transposes is None else np.ascontiguousarray(transposes, dtype=np.uint8)
    if flips is not None and flips.shape != games.shape:
        raise ValueError(f"transposes shape {flips.shape} != positions {games.shape}")
    framed = None if flips is None else np.empty_like(moves)
    rc = _lib().scribblez_move_set_positions(
        _session(),
        str(path).encode("utf-8"),
        games.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        turns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(games),
        moves.ctypes.data_as(ctypes.c_void_p),
        rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cells.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        letters.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        None if flips is None else flips.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        None if framed is None else framed.ctypes.data_as(ctypes.c_void_p),
    )
    if rc != 0:
        raise OSError(f"move_set_positions failed (rc={rc}) for {path}")
    out = {"rows": rows, **_cross_check_result(cells, letters)}
    if framed is not None:
        out["moves"] = framed
    return out


def policy_set_positions(
    path: str | Path,
    game_idx: np.ndarray,
    turn_idx: np.ndarray,
    move_counts: np.ndarray,
    moves: np.ndarray,
    opp_racks: np.ndarray,
    transposes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """The policy-set model's inputs at pool positions of one .slog, from the
    engine code PolicyBot computes them with at play (training/
    policy_set_inputs.h). Positions, moves and `transposes` are addressed as in
    move_set_positions; `opp_racks` (P,) is each position's PoolOppRack record.
    The session must be in the hidden-leaves arm. Returns "rows" (P,
    task_row_floats()) and "opp_rack" (P, opp_rack_floats) float32, and per
    candidate, in its position's frame, encode_moves' arrays (letters,
    blanks, squares, tile_mask, scalars), "cells" and "letters_cross" as
    move_set_positions' "cells" and "letters", and "equity" (M,) float32.
    """
    from scribblez.sim_evidence.sobs import MOVE_DTYPE

    games = np.ascontiguousarray(game_idx, dtype=np.int64)
    turns = np.ascontiguousarray(turn_idx, dtype=np.int64)
    counts = np.ascontiguousarray(move_counts, dtype=np.int64)
    moves = np.ascontiguousarray(moves, dtype=MOVE_DTYPE)
    racks = np.ascontiguousarray(opp_racks, dtype=struct_dtype("PoolOppRack"))
    if not (games.shape == turns.shape == counts.shape == racks.shape) or games.ndim != 1:
        raise ValueError(
            f"per-position shapes differ: {games.shape} {turns.shape} {counts.shape} {racks.shape}"
        )
    if counts.sum() != len(moves):
        raise ValueError(f"move_counts sum {counts.sum()} != moves length {len(moves)}")
    flips = None if transposes is None else np.ascontiguousarray(transposes, dtype=np.uint8)
    if flips is not None and flips.shape != games.shape:
        raise ValueError(f"transposes shape {flips.shape} != positions {games.shape}")
    n = len(moves)
    max_placed, num_scalars, _, _ = move_encoding_dims()
    cells, cross_letters = _cross_check_arrays(n)
    out = {
        "rows": np.empty((len(games), task_row_floats()), dtype=np.float32),
        "opp_rack": np.empty(
            (len(games), format_layout()["constants"]["policy_set"]["opp_rack_floats"]),
            dtype=np.float32,
        ),
        "letters": np.zeros((n, max_placed), dtype=np.int32),
        "blanks": np.zeros((n, max_placed), dtype=np.uint8),
        "squares": np.zeros((n, max_placed), dtype=np.int32),
        "tile_mask": np.zeros((n, max_placed), dtype=np.uint8),
        "scalars": np.zeros((n, num_scalars), dtype=np.float32),
        "equity": np.zeros(n, dtype=np.float32),
    }
    c_float, c_int32, c_uint8 = ctypes.c_float, ctypes.c_int32, ctypes.c_uint8
    targets = _PolicySetOutputs(
        rows=_pointer(out["rows"], c_float),
        opp_rack=_pointer(out["opp_rack"], c_float),
        letters=_pointer(out["letters"], c_int32),
        blanks=_pointer(out["blanks"], c_uint8),
        squares=_pointer(out["squares"], c_int32),
        tile_mask=_pointer(out["tile_mask"], c_uint8),
        scalars=_pointer(out["scalars"], c_float),
        cross_cells=_pointer(cells, c_int32),
        cross_letters=_pointer(cross_letters, c_uint8),
        equity=_pointer(out["equity"], c_float),
    )
    rc = _lib().scribblez_policy_set_positions(
        _session(),
        str(path).encode("utf-8"),
        games.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        turns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(games),
        moves.ctypes.data_as(ctypes.c_void_p),
        racks.ctypes.data_as(ctypes.c_void_p),
        None if flips is None else flips.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(targets),
    )
    if rc != 0:
        raise OSError(f"policy_set_positions failed (rc={rc}) for {path}")
    cross = _cross_check_result(cells, cross_letters)
    return {
        **out,
        # encode_moves' dtypes.
        "letters": out["letters"].astype(np.int64),
        "blanks": out["blanks"].astype(bool),
        "squares": out["squares"].astype(np.int64),
        "tile_mask": out["tile_mask"].astype(bool),
        "cells": cross["cells"],
        "letters_cross": cross["letters"],
    }


def gcg_cross_checks(
    gcg_text: str, moves: np.ndarray, *, open_leaves: bool
) -> dict[str, np.ndarray]:
    """move_set_positions' cross-check features for moves of a position-set
    GCG's decision point, read as gcg_position_inputs reads it. Raises
    ValueError on an unparseable GCG."""
    from scribblez.sim_evidence.sobs import MOVE_DTYPE

    moves = np.ascontiguousarray(moves, dtype=MOVE_DTYPE)
    cells, letters = _cross_check_arrays(len(moves))
    err = ctypes.create_string_buffer(512)
    rc = _lib().scribblez_gcg_cross_checks(
        _session(),
        gcg_text.encode("utf-8"),
        int(open_leaves),
        moves.ctypes.data_as(ctypes.c_void_p),
        len(moves),
        cells.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        letters.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        err,
        len(err),
    )
    if rc != 0:
        raise ValueError(err.value.decode("utf-8") or "gcg_cross_checks failed")
    return _cross_check_result(cells, letters)


def encode_candidate_rows(
    path: str | Path,
    game_idx: np.ndarray,
    turn_idx: np.ndarray,
    move_counts: np.ndarray,
    moves: np.ndarray,
) -> np.ndarray:
    """Post-move input rows for candidate moves, as a position-evaluation model
    scores them: per candidate, the position after it and before the refill,
    from the mover's point of view, under the session's arm.

    Positions and moves are addressed as in move_set_positions. Returns
    (M, task_input_floats()) float32, the spatial planes then the scalars.
    """
    from scribblez.sim_evidence.sobs import MOVE_DTYPE

    games = np.ascontiguousarray(game_idx, dtype=np.int64)
    turns = np.ascontiguousarray(turn_idx, dtype=np.int64)
    counts = np.ascontiguousarray(move_counts, dtype=np.int64)
    moves = np.ascontiguousarray(moves, dtype=MOVE_DTYPE)
    if not (games.shape == turns.shape == counts.shape) or games.ndim != 1:
        raise ValueError(f"per-position shapes differ: {games.shape} {turns.shape} {counts.shape}")
    if counts.sum() != len(moves):
        raise ValueError(f"move_counts sum {counts.sum()} != moves length {len(moves)}")
    out = np.empty((len(moves), task_input_floats()), dtype=np.float32)
    rc = _lib().scribblez_encode_candidate_rows(
        _session(),
        str(path).encode("utf-8"),
        games.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        turns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(games),
        moves.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise OSError(f"encode_candidate_rows failed (rc={rc}) for {path}")
    return out


def probe_replay(
    slog_path: str | Path,
    sprobe_path: str | Path,
    n_positions: int,
    n_candidates: int,
    n_records: int,
    n_turns: int,
) -> dict[str, np.ndarray]:
    """The game state along every probe of a .sprobe file, replayed against its
    companion .slog (engine data/probe_replay.h). The counts are the file's
    totals, which size the outputs. Returns structured arrays keyed "roots"
    (ProbeRootState), "candidates" (ProbeCandidateState), "starts"
    (ProbeStartState) and "turns" (ProbeTurnState)."""
    out = {
        "roots": np.zeros(n_positions, dtype=struct_dtype("ProbeRootState")),
        "candidates": np.zeros(n_candidates, dtype=struct_dtype("ProbeCandidateState")),
        "starts": np.zeros(n_records, dtype=struct_dtype("ProbeStartState")),
        "turns": np.zeros(n_turns, dtype=struct_dtype("ProbeTurnState")),
    }
    err = ctypes.create_string_buffer(512)
    rc = _lib().scribblez_probe_replay(
        str(slog_path).encode("utf-8"),
        str(sprobe_path).encode("utf-8"),
        n_positions,
        n_candidates,
        n_records,
        n_turns,
        *(a.ctypes.data_as(ctypes.c_void_p) for a in out.values()),
        err,
        len(err),
    )
    if rc != 0:
        raise OSError(f"probe_replay failed for {sprobe_path}: {err.value.decode('utf-8')}")
    return out


def reply_blocking(
    slog_path: str | Path, sprobe_path: str | Path, n_records: int, stride: int
) -> np.ndarray:
    """(n_records, stride) uint8: for each record of a .sprobe file and each
    candidate of its position, 1 when the candidate blocks the record's
    opponent reply (engine sim/reply_blocking.h)."""
    out = np.zeros((n_records, stride), dtype=np.uint8)
    err = ctypes.create_string_buffer(512)
    rc = _lib().scribblez_reply_blocking(
        _session(),
        str(slog_path).encode("utf-8"),
        str(sprobe_path).encode("utf-8"),
        n_records,
        stride,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        err,
        len(err),
    )
    if rc != 0:
        raise OSError(f"reply_blocking failed for {sprobe_path}: {err.value.decode('utf-8')}")
    return out


def gcg_sim_evidence(
    gcg_text: str,
    top_k: int = 10,
    rollouts: int = 200,
    seed: int = 0,
    open_leaves: bool = False,
    solve_endgames: bool = False,
) -> tuple[np.ndarray, int]:
    """Sim the top-K moves by HastyBot equity at the decision point before a
    GCG's final recorded move, with common random numbers.

    With `open_leaves`, each rollout's opponent starts from the leave their last
    recorded move kept, with draws still sampled. With `solve_endgames`, rollouts
    solve their endgames rather than play them greedily. Returns (records,
    played_rank): .sobs records (sim_evidence.sobs.RECORD_DTYPE) and the index
    of the GCG's final move among them, or -1 if it is outside the top K.
    Raises on a parse error or an endgame decision point.
    """
    from scribblez.sim_evidence.sobs import RECORD_DTYPE

    buf = ctypes.create_string_buffer(top_k * RECORD_DTYPE.itemsize)
    played_rank = ctypes.c_int(-1)
    n = _lib().scribblez_gcg_sim_evidence(
        _session(),
        gcg_text.encode("utf-8"),
        int(top_k),
        int(rollouts),
        int(seed),
        int(open_leaves),
        int(solve_endgames),
        buf,
        ctypes.byref(played_rank),
    )
    if n < 0:
        raise OSError("gcg_sim_evidence failed (parse error or endgame decision point)")
    # Copy via bytes rather than a structured-array .copy(), which copies field
    # by field and leaves padding bytes uninitialized, breaking byte-level
    # comparisons of the records.
    records = np.frombuffer(bytes(buf.raw[: n * RECORD_DTYPE.itemsize]), dtype=RECORD_DTYPE)
    return records, int(played_rank.value)


def analyze_gcg(gcg_text: str) -> tuple[dict, np.ndarray]:
    """The max-move-per-lane analysis of the position after a GCG's last move.

    Returns (bundle, model_input): `bundle` is the dashboard's lane-analysis
    JSON (board, rack, per-lane ground truth and maximal plays) and
    `model_input` the flat float32 model input, with the side to move's rack.
    Raises OSError on a parse error.
    """
    lib = _lib()
    fn = lib.scribblez_max_move_per_lane_analyze_gcg
    encoded = gcg_text.encode("utf-8")
    inp = np.zeros(task_input_floats("max_move_per_lane"), dtype=np.float32)
    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    cap = 1 << 16
    out = ctypes.create_string_buffer(cap)
    n = fn(_session(), encoded, out, cap, inp_ptr)
    if n < 0:
        raise OSError("analyze_gcg failed (GCG parse error)")
    if n >= cap:  # JSON was truncated; retry once at the exact size
        cap = n + 1
        out = ctypes.create_string_buffer(cap)
        n = fn(_session(), encoded, out, cap, inp_ptr)
    return json.loads(out.value.decode("utf-8")), inp


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
    and scatter the engine applies to the .mset teacher planes. `raw` is
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

    This is the exact .mset distillation target, before collapsing to cells,
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


@dataclass(frozen=True)
class GcgPositionInputs:
    """A position-set GCG's decision point as move set model inputs
    (see gcg_position_inputs)."""

    input_spatial: np.ndarray  # (planes, 15, 15) float32
    input_scalar: np.ndarray  # (scalars,) float32
    score_diff: int  # the mover's pre-move score differential (points)
    moves: np.ndarray  # (N,) MOVE_DTYPE: the full legal move list, equity-ranked


# First-try capacity of the legal-move buffer; a larger position retries at the
# reported count.
_GCG_MOVES_FIRST_CAP = 4096


def gcg_position_inputs(
    gcg_text: str,
    *,
    opp_leave_input: bool,
    spatial_planes: int,
    scalar_size: int,
) -> GcgPositionInputs:
    """Encode a position-set GCG's decision point for the move set model.

    The decision point is the state after the last recorded move, with the side
    to move's rack taken from its #RackN pragma. Returns the board row under the
    given arm (a checkpoint's own, not the session's), the score differential,
    and every legal move in the equity order the trajectory generator draws
    candidates from. The widths come from the model's checkpoint config.
    Raises ValueError on a width the arm does not encode or an unparseable GCG.
    """
    from scribblez.sim_evidence.sobs import MOVE_DTYPE

    lib = _lib()
    inp = np.zeros(spatial_planes * BOARD_CELLS + scalar_size, dtype=np.float32)
    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    score_diff = ctypes.c_int32(0)
    err = ctypes.create_string_buffer(512)
    cap = _GCG_MOVES_FIRST_CAP
    while True:
        moves = np.zeros(cap, dtype=MOVE_DTYPE)
        n = lib.scribblez_gcg_position_inputs(
            _session(),
            gcg_text.encode("utf-8"),
            int(opp_leave_input),
            inp_ptr,
            len(inp),
            ctypes.byref(score_diff),
            moves.ctypes.data_as(ctypes.c_void_p),
            cap,
            err,
            len(err),
        )
        if n < 0:
            raise ValueError(err.value.decode("utf-8") or "gcg_position_inputs failed")
        if n <= cap:
            break
        cap = n
    return GcgPositionInputs(
        input_spatial=inp[: spatial_planes * BOARD_CELLS].reshape(spatial_planes, 15, 15),
        input_scalar=inp[spatial_planes * BOARD_CELLS :],
        score_diff=int(score_diff.value),
        moves=moves[:n].copy(),
    )


def gcg_position_board_json(gcg_text: str, open_leaves: bool) -> dict:
    """The dashboard trajectory pane's bundle for a position-set GCG's decision
    point: the mover's-POV GameState plus `mover`, `opp_leave`, `last_move`,
    and `moves`, every legal move's GCG notation in gcg_position_inputs' order
    (under the same `open_leaves`). Raises OSError on a parse error."""
    lib = _lib()
    fn = lib.scribblez_gcg_position_board_json
    encoded = gcg_text.encode("utf-8")
    cap = 1 << 16
    out = ctypes.create_string_buffer(cap)
    n = fn(_session(), encoded, int(open_leaves), out, cap)
    if n < 0:
        raise OSError("gcg_position_board_json failed (GCG parse error)")
    if n >= cap:  # JSON was truncated; retry once at the exact size
        cap = n + 1
        out = ctypes.create_string_buffer(cap)
        n = fn(_session(), encoded, int(open_leaves), out, cap)
    return json.loads(out.value.decode("utf-8"))


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

    `task` fixes which training row it decodes (TASK_CODES):
    "max_move_per_lane" over every turn, the others over each game's
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

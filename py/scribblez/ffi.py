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

    lib.scribblez_input_shapes.restype = ctypes.POINTER(_ScribblezShape)
    lib.scribblez_input_shapes.argtypes = [ctypes.c_void_p]  # session

    lib.scribblez_target_shapes.restype = ctypes.POINTER(_ScribblezShape)
    lib.scribblez_target_shapes.argtypes = []

    lib.scribblez_row_size_floats.restype = ctypes.c_int
    lib.scribblez_row_size_floats.argtypes = [ctypes.c_void_p]  # session

    lib.scribblez_input_floats.restype = ctypes.c_int
    lib.scribblez_input_floats.argtypes = [ctypes.c_void_p]  # session

    # Max-move-per-lane task: sibling shape/size queries.
    lib.scribblez_max_move_per_lane_input_shapes.restype = ctypes.POINTER(_ScribblezShape)
    lib.scribblez_max_move_per_lane_input_shapes.argtypes = []
    lib.scribblez_max_move_per_lane_target_shapes.restype = ctypes.POINTER(_ScribblezShape)
    lib.scribblez_max_move_per_lane_target_shapes.argtypes = []
    lib.scribblez_max_move_per_lane_row_size_floats.restype = ctypes.c_int
    lib.scribblez_max_move_per_lane_row_size_floats.argtypes = []
    lib.scribblez_max_move_per_lane_input_floats.restype = ctypes.c_int
    lib.scribblez_max_move_per_lane_input_floats.argtypes = []

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
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
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

    lib.scribblez_move_set_cross_check_deltas.restype = ctypes.c_int
    lib.scribblez_move_set_cross_check_deltas.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # .slog path
        ctypes.POINTER(ctypes.c_int64),  # game_idx
        ctypes.POINTER(ctypes.c_int64),  # turn_idx
        ctypes.POINTER(ctypes.c_int64),  # move_counts
        ctypes.c_int64,  # n_positions
        ctypes.c_void_p,  # moves
        ctypes.POINTER(ctypes.c_uint8),  # out_axes
        ctypes.POINTER(ctypes.c_int32),  # out_squares
        ctypes.POINTER(ctypes.c_uint32),  # out_old_masks
        ctypes.POINTER(ctypes.c_uint32),  # out_new_masks
        ctypes.POINTER(ctypes.c_uint8),  # out_delta_mask
    ]

    lib.scribblez_move_set_max_cross_deltas.restype = ctypes.c_int32
    lib.scribblez_move_set_max_cross_deltas.argtypes = []

    lib.scribblez_cross_check_plane0.restype = ctypes.c_int32
    lib.scribblez_cross_check_plane0.argtypes = []

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
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_int,  # open_leaves
        ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_int),
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
        ctypes.c_int,
        ctypes.c_int,
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
    return _read_shapes(_lib().scribblez_input_shapes(_session()))


def get_target_shapes() -> list[ShapeInfo]:
    return _read_shapes(_lib().scribblez_target_shapes())


def row_size_floats() -> int:
    return _lib().scribblez_row_size_floats(_session())


def input_floats() -> int:
    """Floats in one flat input row (spatial + scalar)."""
    return _lib().scribblez_input_floats(_session())


def get_max_move_per_lane_input_shapes() -> list[ShapeInfo]:
    return _read_shapes(_lib().scribblez_max_move_per_lane_input_shapes())


def get_max_move_per_lane_target_shapes() -> list[ShapeInfo]:
    return _read_shapes(_lib().scribblez_max_move_per_lane_target_shapes())


def max_move_per_lane_row_size_floats() -> int:
    return _lib().scribblez_max_move_per_lane_row_size_floats()


def decode_rows(
    path: str | Path,
    game_idx: np.ndarray,
    turn_idx: np.ndarray,
    post_move: bool = True,
) -> np.ndarray:
    """Decode specific training rows of one .slog, addressed by
    (game_idx[j], turn_idx[j]).

    Rows are encoded exactly as the DataLoader encodes them, without symmetry
    augmentation: (n, row_size_floats()) float32. For consumers that pair rows
    with per-position sidecar data such as .sobs sim observations.
    """
    games = np.ascontiguousarray(game_idx, dtype=np.int64)
    turns = np.ascontiguousarray(turn_idx, dtype=np.int64)
    if games.shape != turns.shape or games.ndim != 1:
        raise ValueError(f"game/turn index shapes differ: {games.shape} vs {turns.shape}")
    out = np.empty((len(games), row_size_floats()), dtype=np.float32)
    rc = _lib().scribblez_decode_rows(
        _session(),
        str(path).encode("utf-8"),
        games.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        turns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(games),
        int(post_move),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise OSError(f"decode_rows failed (rc={rc}) for {path}")
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


def cross_check_plane0() -> int:
    """Index of the board input's first cross-check plane. A cross_check_deltas
    entry's (axis, letter) is plane cross_check_plane0() + 26 * axis + letter."""
    return _lib().scribblez_cross_check_plane0()


def cross_check_deltas(
    path: str | Path,
    game_idx: np.ndarray,
    turn_idx: np.ndarray,
    move_counts: np.ndarray,
    moves: np.ndarray,
) -> dict[str, np.ndarray]:
    """The cross-check entries each candidate move changes: a sparse form of
    the post-move cross-check planes (engine training/cross_check_delta.h).

    Position j is the pre-move decision point (game_idx[j], turn_idx[j]) of the
    .slog at `path`; its candidates are the next move_counts[j] records of
    `moves` (MOVE_DTYPE). Returns, each (M, max_cross_deltas):
        axes         int64   0 = horizontal-play planes, 1 = vertical-play
        squares      int64   r*15 + c
        old_masks    uint32  bit L set iff letter L is legal there before the move
        new_masks    uint32  ... after the move
        delta_mask   bool    True on real entries; padding is all-zero
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
    shape = (len(moves), _lib().scribblez_move_set_max_cross_deltas())
    axes = np.zeros(shape, dtype=np.uint8)
    squares = np.zeros(shape, dtype=np.int32)
    old_masks = np.zeros(shape, dtype=np.uint32)
    new_masks = np.zeros(shape, dtype=np.uint32)
    delta_mask = np.zeros(shape, dtype=np.uint8)
    rc = _lib().scribblez_move_set_cross_check_deltas(
        _session(),
        str(path).encode("utf-8"),
        games.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        turns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(games),
        moves.ctypes.data_as(ctypes.c_void_p),
        axes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        squares.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        old_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        new_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        delta_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if rc != 0:
        raise OSError(f"cross_check_deltas failed (rc={rc}) for {path}")
    return {
        "axes": axes.astype(np.int64),
        "squares": squares.astype(np.int64),
        "old_masks": old_masks,
        "new_masks": new_masks,
        "delta_mask": delta_mask.astype(bool),
    }


def gcg_sim_evidence(
    gcg_text: str,
    top_k: int = 10,
    rollouts: int = 200,
    threads: int = 8,
    seed: int = 0,
    open_leaves: bool = False,
) -> tuple[np.ndarray, int]:
    """Sim the top-K moves by HastyBot equity at the decision point before a
    GCG's final recorded move, with common random numbers.

    With `open_leaves`, each rollout's opponent starts from the leave their last
    recorded move kept, with draws still sampled. Returns (records,
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
        int(threads),
        int(seed),
        int(open_leaves),
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
    inp = np.zeros(lib.scribblez_max_move_per_lane_input_floats(), dtype=np.float32)
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
    shapes = {s.name: s.dims for s in get_input_shapes()}
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
    num_pos = ctypes.c_int64()
    file_sz = ctypes.c_int64()
    rc = _lib().scribblez_read_file_header(
        str(path).encode("utf-8"),
        ctypes.byref(num_pos),
        ctypes.byref(file_sz),
    )
    if rc != 0:
        raise OSError(f"Failed to read .slog header: {path}")
    return int(num_pos.value), int(file_sz.value)


# ---------------------------------------------------------------------------
# DataLoader wrapper
# ---------------------------------------------------------------------------


class NativeDataLoader:
    """The C++ DataLoader (engine/include/data/data_loader.h).

    `task` fixes which training row it decodes: "position_eval" (over each
    game's training-eligible turns) or "max_move_per_lane" (over every turn).
    """

    _TASK_CODES = {"position_eval": 0, "max_move_per_lane": 1}

    def __init__(
        self,
        memory_budget: int = 256 * 1024 * 1024,
        num_workers: int = 4,
        num_prefetch: int = 2,
        task: str = "position_eval",
    ):
        if task not in self._TASK_CODES:
            raise ValueError(f"unknown dataloader task {task!r}")
        self._lib = _lib()
        self._handle = self._lib.scribblez_dl_new(
            _session(), memory_budget, num_workers, num_prefetch, self._TASK_CODES[task]
        )
        self._row_floats = (
            max_move_per_lane_row_size_floats()
            if task == "max_move_per_lane"
            else self._lib.scribblez_row_size_floats(_session())
        )

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.scribblez_dl_delete(self._handle)
            self._handle = None

    def add_file(self, path: str | Path, num_positions: int, file_size: int):
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
        turns_per_game: int = 0,
        epoch_index: int = 0,
    ) -> int:
        """Begin an epoch; returns the number of complete batches. See
        SlogDataset.iter_batches for turns_per_game and epoch_index."""
        self._batch_size = batch_size
        return self._lib.scribblez_dl_epoch_start(
            self._handle,
            batch_size,
            int(post_move),
            int(apply_symmetry),
            seed,
            turns_per_game,
            epoch_index,
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

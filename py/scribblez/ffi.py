"""ctypes wrapper around libscribblez_ffi.so."""

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

    lib.scribblez_encode_score_diff_sweep.restype = ctypes.c_int
    lib.scribblez_encode_score_diff_sweep.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
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

    lib.scribblez_dump_position.restype = ctypes.c_int
    lib.scribblez_dump_position.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]

    lib.scribblez_dump_position_json.restype = ctypes.c_int
    lib.scribblez_dump_position_json.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]

    lib.scribblez_sample_slog.restype = ctypes.c_int
    lib.scribblez_sample_slog.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
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

    lib.scribblez_dl_resident_bytes.restype = ctypes.c_int64
    lib.scribblez_dl_resident_bytes.argtypes = [ctypes.c_void_p]

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
    """The engine's description of the sidecar binary formats (.slog / .sobs /
    .mset): per struct the field names, offsets, and numpy dtype codes --
    compiler-derived, so they cannot drift from the packed C++ structs -- plus
    magics, versions, flag bits, and the Move/Glyph code tables
    (engine/include/data/format_layout.h documents the shape). Sessionless;
    parsed once per process."""
    global _FORMAT_LAYOUT
    if _FORMAT_LAYOUT is None:
        _FORMAT_LAYOUT = json.loads(_lib().scribblez_format_layout_json().decode())
    return _FORMAT_LAYOUT


def struct_dtype(name: str) -> np.dtype:
    """The numpy dtype of one engine-described struct (format_layout's
    "structs" keys), nested struct references resolved recursively."""
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


# The lexicon this process's FFI session binds to. Every dictionary-dependent
# entry point (encoding, GCG analysis, DataLoader construction) is a method of
# one C++ ScribblezSession, created lazily on first use.
DEFAULT_LEXICON = "NWL23"

_SESSION_HANDLE = None
_OPP_LEAVE_INPUT = False


def set_opp_leave_input(enabled: bool):
    """Choose the open-leaves experiment arm before any dictionary-dependent
    FFI call: whether the input layout includes the opponent-leave counts
    block (the open-leaves information condition of
    docs/sim_residual_feedback.md -- the opponent's retained leave is public,
    their replenishment draws stay hidden). The session's shape/size queries
    report whichever layout it encodes, so no downstream code branches on this.
    The flag is baked into the process-wide session at creation, so flipping it
    afterwards is an error.
    """
    global _OPP_LEAVE_INPUT
    if _SESSION_HANDLE is not None and _OPP_LEAVE_INPUT != enabled:
        raise RuntimeError("set_opp_leave_input called after the FFI session was created")
    _OPP_LEAVE_INPUT = enabled


def _session() -> int:
    """The process-wide ScribblezSession handle, created on first use.

    Constructing the session loads the lexicon's .kwg; a missing lexicon throws
    out of the C++ constructor, which terminates the process (nothing useful can
    be done without a dictionary).
    """
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
    """Floats in a single input tensor (spatial + scalar)."""
    return _lib().scribblez_input_floats(_session())


def get_max_move_per_lane_input_shapes() -> list[ShapeInfo]:
    return _read_shapes(_lib().scribblez_max_move_per_lane_input_shapes())


def get_max_move_per_lane_target_shapes() -> list[ShapeInfo]:
    return _read_shapes(_lib().scribblez_max_move_per_lane_target_shapes())


def max_move_per_lane_row_size_floats() -> int:
    return _lib().scribblez_max_move_per_lane_row_size_floats()


def encode_score_diff_sweep(
    path: str | Path,
    game_idx: int,
    diff_lo: int,
    diff_hi: int,
    post_move: bool = True,
) -> np.ndarray:
    """Encode sampled positions swept across a score-differential range.

    Replays positions of the .slog file at `path` and re-encodes each once per
    integer score differential in [diff_lo, diff_hi], varying only the active
    player's score advantage. With `game_idx >= 0` only that game is encoded
    (R rows); with `game_idx < 0` every game is encoded (num_games * R rows,
    position-major). Returns a (rows, input_floats()) float32 array, where
    R = diff_hi - diff_lo + 1.
    """
    r = diff_hi - diff_lo + 1
    if r <= 0:
        raise ValueError(f"empty score-diff range [{diff_lo}, {diff_hi}]")
    num_games = read_file_header(path)[0] if game_idx < 0 else 1
    width = input_floats()
    out = np.empty((num_games * r, width), dtype=np.float32)
    rc = _lib().scribblez_encode_score_diff_sweep(
        _session(),
        str(path).encode("utf-8"),
        int(game_idx),
        int(post_move),
        int(diff_lo),
        int(diff_hi),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise OSError(f"encode_score_diff_sweep failed (rc={rc}) for {path} game {game_idx}")
    return out


def decode_rows(
    path: str | Path,
    game_idx: np.ndarray,
    turn_idx: np.ndarray,
    post_move: bool = True,
) -> np.ndarray:
    """Decode explicit training rows of one .slog file by position identity.

    Row j is the position at (game_idx[j], turn_idx[j]), encoded exactly like a
    DataLoader training row (input floats followed by the label block) with no
    symmetry transpose. Returns a (n, row_size_floats()) float32 array. Serves
    consumers that pair rows with per-position sidecar data (the .sobs sim
    observations) and so must address positions by identity rather than stream
    them shuffled.
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
    """The move set evaluation model's move-encoder layout, owned by the engine
    (scribblez/move_set_encoder.h): (max_placed, num_scalars,
    letter_vocab, cells). Needs no session -- it is a pure layout query."""
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
    """The engine's move-feature semantics version (move_set_encoder.h
    kMoveEncodingVersion). Recorded in mset checkpoints and ONNX metadata; a
    model trained under one version must never run against another."""
    return int(_lib().scribblez_move_set_encoding_version())


def score_diff_input_layout() -> tuple[int, float]:
    """(scalar_index, scale) locating the board input's score-diff scalar for
    this session's arm, so a caller can read a position's pre-move differential
    in points as input_scalar[scalar_index] * scale."""
    index = ctypes.c_int32()
    scale = ctypes.c_float()
    _lib().scribblez_score_diff_input_layout(_session(), ctypes.byref(index), ctypes.byref(scale))
    return index.value, scale.value


def encode_moves(moves: np.ndarray, pre_move_score_diffs: np.ndarray) -> dict[str, np.ndarray]:
    """Encode a (M,) array of packed 16-byte Move records
    (scribblez.sim_evidence.sobs.MOVE_DTYPE) into the move set evaluation
    model's move-encoder inputs, via the engine's encoder (the single source
    of truth, shared with the move set evaluation agent). `pre_move_score_diffs`
    is the (M,) per-move mover pre-move score advantage in points, used for the
    resultant post-move differential feature. Returns letters/squares
    (M, max_placed) int64, blanks and tile_mask (M, max_placed) bool, and
    scalars (M, num_scalars) float32.
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


def gcg_sim_evidence(
    gcg_text: str,
    top_k: int = 10,
    rollouts: int = 200,
    threads: int = 8,
    seed: int = 0,
    open_leaves: bool = False,
) -> tuple[np.ndarray, int]:
    """Sim evidence for a penultimate-bingo analysis GCG's final decision point.

    Replays to the state before the final recorded move, ranks the mover's
    legal moves by HastyBot equity, and sims the top-K with common random
    numbers. `open_leaves` starts every rollout's opponent from the leave
    their last recorded move retained (the open-leaves information condition;
    replenishments stay hidden and sampled) -- an empty leave (bingo, or no
    recorded move) is legitimate and equivalent to fully hidden. Returns
    (records, played_rank): `records` is a structured array in the .sobs
    record layout (scribblez.sim_evidence.sobs.RECORD_DTYPE) and
    `played_rank` is the GCG's final move's index within it (-1 if outside
    the top-K). Raises on a parse error or an endgame decision point.
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
    # View over an owned bytes copy. (A structured-array .copy() would rewrite
    # field by field and leave the dtype's padding bytes uninitialized, breaking
    # byte-level comparisons of the records.)
    records = np.frombuffer(bytes(buf.raw[: n * RECORD_DTYPE.itemsize]), dtype=RECORD_DTYPE)
    return records, int(played_rank.value)


def _read_string_ffi(fn, path: str | Path, game_idx: int, post_move: bool, what: str) -> str:
    """Call a (session, path, game_idx, post_move, out, cap)->len FFI, growing the buffer once."""
    encoded = str(path).encode("utf-8")
    cap = 4096
    out = ctypes.create_string_buffer(cap)
    n = fn(_session(), encoded, int(game_idx), int(post_move), out, cap)
    if n < 0:
        raise OSError(f"{what} failed for {path} game {game_idx}")
    if n >= cap:  # buffer was too small; retry once with the exact size
        cap = n + 1
        out = ctypes.create_string_buffer(cap)
        n = fn(_session(), encoded, int(game_idx), int(post_move), out, cap)
    return out.value.decode("utf-8", errors="replace")


def dump_position_json(path: str | Path, game_idx: int, post_move: bool = True) -> str:
    """Return the web UI's GameState JSON for a game's sampled position."""
    return _read_string_ffi(
        _lib().scribblez_dump_position_json, path, game_idx, post_move, "dump_position_json"
    )


def analyze_gcg(gcg_text: str) -> tuple[dict, np.ndarray]:
    """Parse GCG text into the max-move-per-lane analysis bundle.

    Returns (bundle, model_input): `bundle` is the lane-analysis JSON parsed to a
    dict (the web board/bonuses/rack the dashboard renders, plus per-lane ground
    truth and maximal plays); `model_input` is the flat float32 model-input tensor
    for the analysis position (board after all recorded moves, on-move player's
    rack). Raises IOError on a parse error.
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
    """A board-row encoding arm together with the input widths a model built
    for it declares. The engine encodes under the arm and refuses a width it
    does not produce, so a model from another encoding era is caught at the
    encode call rather than fed a misaligned row. `session_input_arm()` is the
    process-wide session's own; a served ONNX model's comes from its metadata
    and declared input shapes."""

    opp_leave_input: bool
    spatial_planes: int
    scalar_size: int

    @property
    def input_floats(self) -> int:
        return self.spatial_planes * BOARD_CELLS + self.scalar_size

    def split(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A flat row under this arm as its (spatial (P, 15, 15), scalar (S,)) halves."""
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
    """Encode a dataset GCG's post-move position into the position evaluation
    model's flat float32 input tensor under `arm` (arm.input_floats long; the
    session contributes only its dictionary).

    The position is the board after the final recorded move, encoded from the POV of
    the player that made it (its leave is the encode-time rack) -- the same seat the
    Monte-Carlo ground truth scores; the opponent-leave arm also reads what the
    opponent's last move retained. Raises ValueError when the arm does not encode
    the declared widths (a model from another encoding era), OSError on a parse error
    or a non-PLAY final move.
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
    """Collapse a dataset GCG's post-move placement logits into per-cell marginals.

    `raw` is the four placement heads' raw footprint logits, shape
    (4, footprint num_classes) or flat; the result is the four board-frame
    (4, 15, 15) occupancy marginals -- Pr[the next move covers cell] for the
    plays heads and Pr[covers cell AND that seat wins] for the win heads -- via
    the same mask + softmax + scatter the engine applies to the .mset teacher
    planes. OSError on a parse error or a non-PLAY final move.
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
    """The four placement heads' MASKED footprint distributions for a dataset GCG.

    `raw` is the four heads' raw footprint logits, shape (4, num_classes) or flat;
    the result is (4, num_classes), each head a distribution over the 2927 classes
    (illegal footprints at zero) via the same board-legality mask + masked softmax
    the engine applies to the .mset teacher target -- but per class, not collapsed
    to cells. This is the exact distilled placement target, exposed so its
    per-footprint sparsity/fidelity can be measured. OSError on a parse error or a
    non-PLAY final move.
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
    """Computes each of the four placement heads' per-cell legality for a dataset
    GCG's post-move position: (4, 15, 15) bool, True where some legal anchored
    footprint (see collapse_position_eval_placement) covers the cell. OSError on
    a parse error or a non-PLAY final move.
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
    """The engine's one -1 covers two callers' concerns: a width the arm does not
    encode (the caller's model is wrong for today's layout) and a position that
    does not parse (the GCG is wrong)."""
    if "the arm encodes" in reason:
        raise ValueError(reason)
    raise OSError(reason or "GCG parse error or non-PLAY final move")


def analyze_position_eval_gcg_leaves(
    gcg_text: str, leave: str, opp_leave: str | None, arm: InputArm
) -> np.ndarray:
    """Encode the analysis position with explicit alternate leaves ('?' = a
    blank) instead of the GCG's recorded ones -- a dashboard what-if: `leave`
    for the POV and, unless None, `opp_leave` for the opponent (read by the
    opponent-leave arm only).

    Board, scores, and moves are unchanged; only the rack, opponent-leave, and
    unseen-pool features reflect the new leaves. Raises ValueError with a
    human-readable reason on a size mismatch, unavailable tiles, or a width the
    arm does not encode; OSError on a GCG parse error.
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
    """The web-render board bundle for a dataset GCG's post-move position (board
    / bonuses / rack / tile_scores / start_player / last_move / opp_leave), parsed
    to a dict.

    The board is the position after the final recorded move; the rack is the leave of
    the player that made it (the evaluated POV); opp_leave is what the opponent's last
    recorded move retained ('?' = a blank). Raises IOError on a parse error or a
    non-PLAY final move.
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
    """A position-set .gcg's decision point as the move set evaluation model's
    inputs (see gcg_position_inputs)."""

    input_spatial: np.ndarray  # (planes, 15, 15) float32
    input_scalar: np.ndarray  # (scalars,) float32
    score_diff: int  # the mover's pre-move score differential (points)
    moves: np.ndarray  # (N,) MOVE_DTYPE: the full legal move list, equity-ranked


# The legal-move buffer's first-try capacity; a busier position retries at the
# reported count.
_GCG_MOVES_FIRST_CAP = 4096


def gcg_position_inputs(
    gcg_text: str,
    *,
    opp_leave_input: bool,
    spatial_planes: int,
    scalar_size: int,
) -> GcgPositionInputs:
    """Encode a position-set .gcg's decision point (final recorded state, side
    to move next, rack from its #RackN pragma) for the move set evaluation
    model: the mover's pre-move board row under the given arm (independent of
    the process-wide session's arm -- a checkpoint's config names its own),
    the pre-move score differential, and the FULL legal move list in the
    equity ranking the trajectory generator drew its candidates from.
    `spatial_planes` / `scalar_size` are the model's (its checkpoint config);
    a width the arm does not encode is a ValueError, as is a GCG that does
    not parse."""
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
    """The trajectory pane's web-render bundle for a position-set .gcg's
    decision point: the mover's-POV GameState (board / bonuses / rack / scores /
    bag and opponent-rack counts / tile_scores) plus `mover`, `opp_leave`,
    `last_move`, and `moves` -- every legal move's GCG notation, in
    gcg_position_inputs' order under the same `open_leaves`. Raises OSError on
    a parse error."""
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


def sample_slog(dst_path: str | Path, picks: list[tuple[str | Path, int]]):
    """Write a new .slog at `dst_path` from selected (source path, game index) picks."""
    n = len(picks)
    src_arr = (ctypes.c_char_p * n)(*[str(p).encode("utf-8") for p, _ in picks])
    idx_arr = (ctypes.c_int64 * n)(*[int(g) for _, g in picks])
    rc = _lib().scribblez_sample_slog(str(dst_path).encode("utf-8"), src_arr, idx_arr, n)
    if rc != 0:
        raise OSError(f"sample_slog failed (rc={rc}) writing {dst_path}")


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
        raise OSError(f"Failed to read .slog header: {path}")
    return int(num_pos.value), int(file_sz.value)


# ---------------------------------------------------------------------------
# DataLoader wrapper
# ---------------------------------------------------------------------------


class NativeDataLoader:
    """Python wrapper around the C++ DataLoader via FFI.

    `task` selects which training row the loader decodes from each .slog game:
    "position_eval" (the position-evaluation row, over each game's eligible-turn prefix)
    or "max_move_per_lane" (the per-lane row, over every turn). It fixes the row
    width and is baked into the handle at construction.
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
        """Begin a new epoch. Returns number of complete batches.

        turns_per_game: 0 = every eligible turn of every game; k > 0 = k turns
        per game per epoch, with epoch_index selecting which turns so successive
        epochs cover distinct turns.
        """
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
        """Load the next batch. Returns None when epoch is exhausted.

        Raises OSError if a registered .slog became unreadable mid-epoch (the
        native loader reports -1 rather than blocking forever on a load that
        can never complete -- e.g. a window file deleted or truncated under the
        reader); the specific file is named on stderr.
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

    @property
    def resident_bytes(self) -> int:
        return int(self._lib.scribblez_dl_resident_bytes(self._handle))

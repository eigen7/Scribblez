"""ctypes wrapper around libscribblez_ffi.so."""

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

_FFI_LIB_PATH = "/workspace/repo/target/engine/libscribblez_ffi.so"
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
    lib.scribblez_session_new.argtypes = [ctypes.c_char_p, ctypes.c_int]  # lexicon, contingent

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

    lib.scribblez_post_move_value_analyze_gcg.restype = ctypes.c_int
    lib.scribblez_post_move_value_analyze_gcg.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg_text
        ctypes.POINTER(ctypes.c_float),  # out_input
    ]

    lib.scribblez_post_move_value_board_json.restype = ctypes.c_int
    lib.scribblez_post_move_value_board_json.argtypes = [
        ctypes.c_char_p,  # gcg_text
        ctypes.c_char_p,  # out_json
        ctypes.c_int,  # out_cap
    ]

    lib.scribblez_post_move_value_analyze_gcg_leave.restype = ctypes.c_int
    lib.scribblez_post_move_value_analyze_gcg_leave.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,  # gcg_text
        ctypes.c_char_p,  # leave_str
        ctypes.POINTER(ctypes.c_float),  # out_input
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

    lib.scribblez_gcg_sim_evidence.restype = ctypes.c_int
    lib.scribblez_gcg_sim_evidence.argtypes = [
        ctypes.c_void_p,  # session
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
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


_SETUP_DONE = False


def _lib() -> ctypes.CDLL:
    global _SETUP_DONE
    lib = _load_lib()
    if not _SETUP_DONE:
        _setup_lib(lib)
        _SETUP_DONE = True
    return lib


# The lexicon this process's FFI session binds to. Every dictionary-dependent
# entry point (encoding, GCG analysis, DataLoader construction) is a method of
# one C++ ScribblezSession, created lazily on first use.
DEFAULT_LEXICON = "NWL23"

_SESSION_HANDLE = None
_CONTINGENT_FEATURES = True


def set_contingent_features(enabled: bool):
    """Choose the process's experiment arm before any dictionary-dependent FFI
    call: whether the engine encodes the full input layout including the
    contingent-draw potential features (True), or skips their move generation
    and encodes the smaller base layout (False). The session's shape/size
    queries report whichever layout it encodes, so no downstream code branches
    on this. The flag is baked into the process-wide session at creation, so
    flipping it afterwards is an error.
    """
    global _CONTINGENT_FEATURES
    if _SESSION_HANDLE is not None and _CONTINGENT_FEATURES != enabled:
        raise RuntimeError("set_contingent_features called after the FFI session was created")
    _CONTINGENT_FEATURES = enabled


def _session() -> int:
    """The process-wide ScribblezSession handle, created on first use.

    Constructing the session loads the lexicon's .kwg; a missing lexicon throws
    out of the C++ constructor, which terminates the process (nothing useful can
    be done without a dictionary).
    """
    global _SESSION_HANDLE
    if _SESSION_HANDLE is None:
        _SESSION_HANDLE = _lib().scribblez_session_new(
            DEFAULT_LEXICON.encode("utf-8"), int(_CONTINGENT_FEATURES)
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
    symmetry flip. Returns a (n, row_size_floats()) float32 array. Serves
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


def gcg_sim_evidence(
    gcg_text: str,
    top_k: int = 10,
    rollouts: int = 200,
    threads: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Sim evidence for a penultimate-bingo analysis GCG's final decision point.

    Replays to the state before the final recorded move, ranks the mover's
    legal moves by HastyBot equity, and sims the top-K with common random
    numbers. Returns (records, played_rank): `records` is a structured array
    in the .sobs record layout (scribblez.sim_evidence.sobs.RECORD_DTYPE) and
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


def analyze_post_move_gcg(gcg_text: str) -> np.ndarray:
    """Encode a penultimate-bingo GCG's post-move analysis position into the
    post-move value model's flat float32 input tensor (input_floats() long).

    The position is the board after the final recorded move, encoded from the POV of
    the player that made it (its leave is the encode-time rack) -- the same seat the
    Monte-Carlo ground truth scores. Raises IOError on a parse error or a non-PLAY
    final move.
    """
    lib = _lib()
    inp = np.zeros(lib.scribblez_input_floats(_session()), dtype=np.float32)
    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if lib.scribblez_post_move_value_analyze_gcg(_session(), gcg_text.encode("utf-8"), inp_ptr) < 0:
        raise OSError("analyze_post_move_gcg failed (GCG parse error or non-PLAY final move)")
    return inp


def analyze_post_move_gcg_leave(gcg_text: str, leave: str) -> np.ndarray:
    """Encode the post-move analysis position with an explicit alternate `leave` ('?' =
    a blank) instead of the GCG's recorded one -- a dashboard what-if.

    Board, scores, and moves are unchanged; only the rack and unseen-pool features
    reflect the new leave. Raises ValueError with a human-readable reason on a size
    mismatch or unavailable tiles, OSError on a GCG parse error.
    """
    lib = _lib()
    inp = np.zeros(lib.scribblez_input_floats(_session()), dtype=np.float32)
    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    err = ctypes.create_string_buffer(256)
    n = lib.scribblez_post_move_value_analyze_gcg_leave(
        _session(), gcg_text.encode("utf-8"), leave.encode("utf-8"), inp_ptr, err, len(err)
    )
    if n < 0:
        raise ValueError(err.value.decode("utf-8") or "invalid alternate leave")
    return inp


def post_move_board_json(gcg_text: str) -> dict:
    """The web-render board bundle for a penultimate-bingo GCG's post-move analysis
    position (board / bonuses / rack / tile_scores / start_player), parsed to a dict.

    The board is the position after the final recorded move; the rack is the leave of
    the player that made it (the evaluated POV). Raises IOError on a parse error or a
    non-PLAY final move.
    """
    lib = _lib()
    fn = lib.scribblez_post_move_value_board_json
    encoded = gcg_text.encode("utf-8")
    cap = 1 << 16
    out = ctypes.create_string_buffer(cap)
    n = fn(encoded, out, cap)
    if n < 0:
        raise OSError("post_move_board_json failed (GCG parse error or non-PLAY final move)")
    if n >= cap:  # JSON was truncated; retry once at the exact size
        cap = n + 1
        out = ctypes.create_string_buffer(cap)
        n = fn(encoded, out, cap)
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
    "post_move" (the post-move value row, over each game's eligible-turn prefix)
    or "max_move_per_lane" (the per-lane row, over every turn). It fixes the row
    width and is baked into the handle at construction.
    """

    _TASK_CODES = {"post_move": 0, "max_move_per_lane": 1}

    def __init__(
        self,
        memory_budget: int = 256 * 1024 * 1024,
        num_workers: int = 4,
        num_prefetch: int = 2,
        task: str = "post_move",
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

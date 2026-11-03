"""The DeepBot leaf eval task's rows through the FFI (engine
training/leaf_eval_task.h): the DataLoader and decode_rows over .slog files
written by the C++ test_slog_writer binary. The join planes are recomputed here
from each row's own board and cross-check planes, and the per-square fates from
the boards and score differentials of the game's later rows."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scribblez import ffi
from scribblez.dataset import SlogDataset
from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.supply_registers import TILE_COUNTS

# This checkout's own binaries, not the primary checkout's -- see the note in
# test_move_set_eval_targets.py.
_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
_TASK = "deep_bot_leaf_eval"

N = 15
LETTERS = 26
TILE_KINDS = 27
BLANK_PLANE = 26
CROSS_CHECK0 = 31
JOIN0 = CROSS_CHECK0 + 2 * LETTERS
SPATIAL_PLANES = JOIN0 + 12
MY_LEAVE0, OPP_LEAVE0, POOL0 = 0, 27, 54
SCORE_DIFF, MY_LEAVE_VALUE, OPP_LEAVE_VALUE, POOL_SIZE = 81, 82, 83, 84
SCALARS = 85
# Square fates (encoding/square_fates.h).
FILLED, OWN_BINGO, OWN_OTHER, OPP_BINGO, OPP_OTHER, STAYS_EMPTY = -1, 0, 1, 2, 3, 4
SCORE_DIFF_SCALE = 100.0
TILE_VALUES = np.array(
    [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10, 0]
)


def _write_slogs(out_dir: Path) -> list[Path]:
    binary = _ENGINE_DIR / "test_slog_writer"
    if not binary.is_file():
        pytest.skip("test_slog_writer not built -- run py/build.py first")
    subprocess.run([str(binary), str(out_dir), "8", "4"], check=True, capture_output=True)
    return sorted(out_dir.glob("*.slog"))


@pytest.fixture(scope="module")
def slogs(tmp_path_factory) -> list[Path]:
    return _write_slogs(tmp_path_factory.mktemp("slogs"))


@pytest.fixture(scope="module")
def batch(slogs) -> tuple[np.ndarray, np.ndarray, dict]:
    """(spatial (B, planes, 15, 15), scalar (B, 85), targets), transposed rows
    included."""
    ds = SlogDataset(slogs[0].parent, task=_TASK, apply_symmetry=True)
    b = next(ds.iter_batches(batch_size=256, seed=3))
    return b["input_spatial"].numpy(), b["input_scalar"].numpy(), b


def _occupied(spatial: np.ndarray) -> np.ndarray:
    return spatial[:, :LETTERS].sum(axis=1) > 0


def _hook_squares(occupied: np.ndarray, axis: int) -> np.ndarray:
    """Empty squares with a tile above or below (axis 0, a horizontal play) or
    left or right (axis 1)."""
    pad = np.pad(occupied, ((0, 0), (1, 1), (1, 1)))
    if axis == 0:
        neighbor = pad[:, :-2, 1:-1] | pad[:, 2:, 1:-1]
    else:
        neighbor = pad[:, 1:-1, :-2] | pad[:, 1:-1, 2:]
    return neighbor & ~occupied


def _expected_join(seat: np.ndarray, masks: np.ndarray, fractional: bool) -> np.ndarray:
    """(B, 2, 15, 15) fit and face-value planes for one seat's (B, 27) counts
    against (B, 26, 15, 15) cross-check masks, before the hook-square cut."""
    fits = np.concatenate([masks, masks.any(axis=1, keepdims=True)], axis=1)  # blank last
    fitting = fits * seat[:, :, None, None]
    n = fitting.sum(axis=1)
    if fractional:
        fit = np.where(n > 0, n / np.maximum(seat.sum(axis=1), 1)[:, None, None], 0.0)
    else:
        fit = (n > 0).astype(np.float64)
    values = np.where(fitting > 0, TILE_VALUES[None, :, None, None], 0)
    return np.stack([fit, values.max(axis=1) / 10.0], axis=1)


def test_shapes_cover_the_row():
    inputs = ffi.task_input_shapes(_TASK)
    targets = ffi.task_target_shapes(_TASK)
    assert [(s.name, s.dims) for s in inputs] == [
        ("input_spatial", (SPATIAL_PLANES, N, N)),
        ("input_scalar", (SCALARS,)),
    ]
    assert [(s.name, s.dims) for s in targets] == [
        ("wld", (3,)),
        ("score_diff", (1,)),
        ("ownership", (N, N)),
        ("square_points", (N, N)),
    ]
    total = sum(int(np.prod(s.dims)) for s in inputs + targets)
    assert total == ffi.task_row_floats(_TASK)


def test_board_leaves_and_pool_hold_every_tile_once(batch):
    spatial, scalar, _ = batch
    blanks_on_board = spatial[:, BLANK_PLANE].sum(axis=(1, 2))
    blank_letters = (spatial[:, :LETTERS] * spatial[:, BLANK_PLANE : BLANK_PLANE + 1]).sum(
        axis=(2, 3)
    )
    board = np.concatenate(
        [spatial[:, :LETTERS].sum(axis=(2, 3)) - blank_letters, blanks_on_board[:, None]], axis=1
    )
    seats = scalar[:, MY_LEAVE0:POOL0].reshape(-1, 2, TILE_KINDS).sum(axis=1)
    pool = scalar[:, POOL0 : POOL0 + TILE_KINDS]
    np.testing.assert_array_equal(board + seats + pool, np.tile(TILE_COUNTS, (len(scalar), 1)))
    np.testing.assert_allclose(scalar[:, POOL_SIZE], pool.sum(axis=1) / 100.0, rtol=1e-6)
    assert (scalar[:, MY_LEAVE0:OPP_LEAVE0].sum(axis=1) <= 7).all()
    assert (scalar[:, OPP_LEAVE0:POOL0].sum(axis=1) <= 7).all()


def test_join_planes_match_a_recomputation(batch):
    spatial, scalar, _ = batch
    occupied = _occupied(spatial)
    seats = [
        (scalar[:, MY_LEAVE0:OPP_LEAVE0], False),
        (scalar[:, OPP_LEAVE0:POOL0], False),
        (scalar[:, POOL0 : POOL0 + TILE_KINDS], True),
    ]
    for axis in range(2):
        masks = spatial[:, CROSS_CHECK0 + axis * LETTERS : CROSS_CHECK0 + (axis + 1) * LETTERS] > 0
        hooks = _hook_squares(occupied, axis)[:, None]
        for s, (counts, fractional) in enumerate(seats):
            plane0 = JOIN0 + (s * 2 + axis) * 2
            expected = np.where(hooks, _expected_join(counts, masks, fractional), 0.0)
            np.testing.assert_allclose(spatial[:, plane0 : plane0 + 2], expected, atol=1e-6)
    # The batch exercises the planes: some hook squares fit and some do not.
    leave_fit = spatial[:, JOIN0 : JOIN0 + 4 : 2]
    assert leave_fit.max() == 1.0
    assert (
        leave_fit[np.broadcast_to(_hook_squares(occupied, 0)[:, None], leave_fit.shape)] == 0
    ).any()


def test_targets_are_the_movers_outcome(batch):
    _, _, b = batch
    wld = b["wld"].numpy()
    diff = b["score_diff"].numpy()[:, 0]
    np.testing.assert_array_equal(wld.sum(axis=1), 1.0)
    np.testing.assert_array_equal(wld[:, 0] == 1, diff > 0)
    np.testing.assert_array_equal(wld[:, 2] == 1, diff < 0)


def test_first_turn_row_has_no_opponent_leave(slogs):
    """Before the opponent moves, their leave is empty and their whole rack is
    in the pool, which is the leaf the model sees after a bingo too."""
    rows = ffi.decode_rows(slogs[0], np.array([0]), np.array([0]), post_move=True, task=_TASK)
    scalar = rows[0, SPATIAL_PLANES * N * N : SPATIAL_PLANES * N * N + SCALARS]
    assert scalar[OPP_LEAVE0:POOL0].sum() == 0
    assert scalar[OPP_LEAVE_VALUE] == 0
    assert scalar[MY_LEAVE0:OPP_LEAVE0].sum() > 0


def _leave_key(sorted_letters: np.ndarray) -> np.ndarray:
    """(k, R) ascending letter indices -> (R,) base-28 keys, unique per leave."""
    key = np.zeros(sorted_letters.shape[1], dtype=np.int64)
    for row in sorted_letters:
        key = key * 28 + row + 1
    return key


def _expected_leave_values(counts: np.ndarray) -> np.ndarray:
    """HastyBot's value of each (R, 27) leave; 0 for one not in the table."""
    arrays = ffi.LeaveValueArrays()
    lengths = counts.sum(axis=1)
    out = np.zeros(len(counts))
    for k in range(1, arrays.max_length + 1):
        rows = np.flatnonzero(lengths == k)
        if len(rows) == 0:
            continue
        codes, values = arrays.of_length(k)
        table_keys = _leave_key(codes // arrays.copies_per_tile)
        letters = np.stack([np.repeat(np.arange(TILE_KINDS), c) for c in counts[rows]])
        wanted = _leave_key(letters.T)
        order = np.argsort(table_keys)
        found = order[np.searchsorted(table_keys, wanted, sorter=order)]
        assert (table_keys[found] == wanted).all()
        out[rows] = values[found]
    return out


def test_leave_values_are_hastybots(batch):
    _, scalar, _ = batch
    for offset, value_index in ((MY_LEAVE0, MY_LEAVE_VALUE), (OPP_LEAVE0, OPP_LEAVE_VALUE)):
        counts = scalar[:, offset : offset + TILE_KINDS].astype(np.int64)
        # An empty leave, or a whole rack kept by a pass, is not in the table.
        expected = _expected_leave_values(counts)
        np.testing.assert_allclose(scalar[:, value_index] * 25.0, expected, atol=1e-4)
    assert (scalar[:, MY_LEAVE_VALUE] != 0).any()


def test_format_layout_serves_the_offsets_these_tests_assume():
    layout = ffi.format_layout()["constants"][_TASK]
    assert layout["format_version"] >= 1
    assert (layout["join_plane0"], layout["join_planes"]) == (JOIN0, SPATIAL_PLANES - JOIN0)
    offsets = ("my_leave0", "opp_leave0", "pool0", "score_diff", "my_leave_value")
    assert tuple(layout[k] for k in offsets) == (
        MY_LEAVE0,
        OPP_LEAVE0,
        POOL0,
        SCORE_DIFF,
        MY_LEAVE_VALUE,
    )
    assert (layout["opp_leave_value"], layout["pool_size"]) == (OPP_LEAVE_VALUE, POOL_SIZE)


@pytest.fixture(scope="module")
def played_games(tmp_path_factory) -> Path:
    """Real HastyBot self-play, whose bags run out (test_slog_writer's short
    games never empty theirs)."""
    if not (_ENGINE_DIR / "play_game").is_file():
        pytest.skip("play_game not built")
    out = tmp_path_factory.mktemp("played")
    run_games(out, 6, threads=1, player_spec=hasty_player_spec(0.0, 10, endgame=True), seed=7)
    return out


def test_every_row_keeps_tiles_in_the_bag_after_both_refills(played_games):
    """DeepBot values a leaf only if the bag still holds tiles after both
    players refill (rollouts that empty it play to the end), so every row must
    be such a leaf: the pool outlasts both refills to 7."""
    ds = SlogDataset(played_games, task=_TASK, apply_symmetry=False, num_workers=1)
    scalar = np.concatenate([b["input_scalar"].numpy() for b in ds.iter_batches(256, seed=0)])
    my = scalar[:, MY_LEAVE0:OPP_LEAVE0].sum(axis=1)
    opp = scalar[:, OPP_LEAVE0:POOL0].sum(axis=1)
    pool = scalar[:, POOL0 : POOL0 + TILE_KINDS].sum(axis=1)
    assert (pool > (7 - opp) + (7 - my)).all()
    assert (pool <= 14).any()  # the late rows this guards are present


def test_each_game_drops_its_last_eligible_turn(played_games):
    """The leaf rows are the eligible turns but the last, whose draw empties
    the bag."""
    header = ffi.struct_dtype("SlogFileHeader")
    expected = 0
    for path in sorted(played_games.glob("*.slog")):
        num_games = int(np.fromfile(path, header, count=1)[0]["num_games"])
        metas = np.fromfile(
            path, ffi.struct_dtype("SlogGameMetadata"), count=num_games, offset=header.itemsize
        )
        begin, end = metas["eligible_begin"].astype(int), metas["eligible_end"].astype(int)
        expected += int(np.maximum(end - 1 - begin, 0).sum())
    assert SlogDataset(played_games, task=_TASK).num_samples == expected


def test_square_fates_are_consistent(batch):
    """Filled squares carry no fate, a square that stays empty scores nothing,
    each side's moves carry its sign, and a bingo scores its 50-point bonus."""
    spatial, _, b = batch
    fate = b["ownership"].numpy()
    points = b["square_points"].numpy()
    np.testing.assert_array_equal(fate == FILLED, _occupied(spatial))
    assert set(np.unique(fate)) <= {FILLED, OWN_BINGO, OWN_OTHER, OPP_BINGO, OPP_OTHER, STAYS_EMPTY}
    assert (points[(fate == FILLED) | (fate == STAYS_EMPTY)] == 0).all()
    assert (points[(fate == OWN_BINGO) | (fate == OWN_OTHER)] >= 0).all()
    assert (points[(fate == OPP_BINGO) | (fate == OPP_OTHER)] <= 0).all()
    assert (np.abs(points[(fate == OWN_BINGO) | (fate == OPP_BINGO)]) >= 50).all()
    # Short test games leave squares of every kind but bingos.
    assert {OWN_OTHER, OPP_OTHER, STAYS_EMPTY} <= set(np.unique(fate))


def _rows_by_turn(path: Path, game: int, turns: np.ndarray) -> dict[str, np.ndarray]:
    rows = ffi.decode_rows(path, np.full(len(turns), game), turns, post_move=True, task=_TASK)
    spatial_floats = SPATIAL_PLANES * N * N
    labels = rows[:, spatial_floats + SCALARS :]
    return {
        "occupied": rows[:, : LETTERS * N * N].reshape(-1, LETTERS, N, N).sum(axis=1) > 0,
        "diff": np.rint(rows[:, spatial_floats + SCORE_DIFF] * SCORE_DIFF_SCALE),
        "fate": labels[:, 4 : 4 + N * N].reshape(-1, N, N),
        "points": labels[:, 4 + N * N :].reshape(-1, N, N),
    }


def _check_next_move(rows: dict, t: int, k: int):
    """At row t, the squares turn t + k fills carry that move's fate: the
    evaluator's (k even) or the opponent's (k odd), a bingo iff it placed 7
    tiles, and its signed score. A move's score is the sum of the spreads
    before and after it, each seen by its own mover."""
    placed = rows["occupied"][t + k] & ~rows["occupied"][t + k - 1]
    score = rows["diff"][t + k] + rows["diff"][t + k - 1]
    if not placed.any():
        assert score == 0
        return
    mine = k % 2 == 0
    bingo = placed.sum() == 7
    expected = (OWN_BINGO if bingo else OWN_OTHER) if mine else (OPP_BINGO if bingo else OPP_OTHER)
    assert (rows["fate"][t][placed] == expected).all()
    assert (rows["points"][t][placed] == (score if mine else -score)).all()


def test_square_fates_match_the_games_later_moves(played_games):
    header = ffi.struct_dtype("SlogFileHeader")
    checked = 0
    for path in sorted(played_games.glob("*.slog")):
        num_games = int(np.fromfile(path, header, count=1)[0]["num_games"])
        metas = np.fromfile(
            path, ffi.struct_dtype("SlogGameMetadata"), count=num_games, offset=header.itemsize
        )
        for game, meta in enumerate(metas):
            turns = np.arange(int(meta["eligible_begin"]), int(meta["eligible_end"]) - 1)
            if len(turns) < 3:
                continue
            rows = _rows_by_turn(path, game, turns)
            for t in range(len(turns) - 2):
                _check_next_move(rows, t, 1)
                _check_next_move(rows, t, 2)
                checked += 1
    assert checked > 20

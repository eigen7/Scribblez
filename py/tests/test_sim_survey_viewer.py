"""The sim-survey viewer's data builder."""

from scribblez.sim_survey_viewer import Tile, board_before, bonuses, placed_tiles


def test_placed_tiles_reads_both_orientations_blanks_and_played_through_squares():
    # Column-first reads down: K6 is column K (10), row 6 (5).
    assert placed_tiles("K6 AC.TA") == [
        Tile(5, 10, "A", False),
        Tile(6, 10, "C", False),
        Tile(8, 10, "T", False),
        Tile(9, 10, "A", False),
    ]
    # Row-first reads across; a lowercase letter is a blank.
    assert placed_tiles("6H .Ar")[-1] == Tile(5, 9, "R", True)
    assert placed_tiles("-ABC") == [] and placed_tiles("-") == []


def test_board_before_replays_only_the_earlier_turns():
    gcg = "\n".join(
        [
            "#player1 a a",
            ">a: EEEFGKR 8H GREEK +30 30",
            ">b: AACITTZ -AC +0 0",
            ">a: AEFNOOU 9G Zo +27 57",
        ]
    )
    board = board_before(gcg, 2)
    assert "".join(board[7][7:12]) == "GREEK"
    assert board[8][6] is None  # turn 2 (0-based) itself is not applied
    assert board_before(gcg, 3)[8][6:8] == ["Z", "o"]


def test_bonuses_is_the_standard_symmetric_layout():
    grid = bonuses()
    assert len(grid) == 15 and all(len(row) == 15 for row in grid)
    assert grid[0][0] == grid[7][0] == grid[14][14] == "TW"
    assert grid[7][7] == "DW" and grid[5][9] == "TL" and grid[6][6] == "DL"
    assert grid == [list(col) for col in zip(*grid, strict=True)]  # diagonal symmetry

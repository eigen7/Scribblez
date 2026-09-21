"""The sim-survey viewer's data builder."""

from scribblez.sim_survey_viewer import Tile, board_before, bonuses, placed_tiles, position_entry


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


def test_position_entry_counts_the_opponents_rack(tmp_path):
    # 5 tiles on the board, 7 on the mover's rack, 81 in the bag: the other 7 are
    # the opponent's.
    (tmp_path / "chunk-g0-turn2.gcg").write_text(">a: EEEFGKR 8H GREEK +30 30\n")
    summary = {
        "n": 100, "wins": 60, "draws": 0, "delta_sum": 0.0, "delta_sq_sum": 0.0, "delta_hist": [],
        "end_swing_sum": 0.0, "opp_stranded_sum": 0.0, "self_stranded_sum": 0.0,
        "self_went_out": 0, "opp_went_out": 0,
    }  # fmt: skip
    side = {"score_sum": 0.0, "bingos": 0, "non_plays": 0, "adjacent": 0, "score_hist": []}
    summary |= {"opp_reply": side, "self_next": side}
    candidate = {"equity": 1.0, "score": 7, "leave": "ITZ", "is_setup": False}
    position = {
        "game": 0, "turn": 1, "mover": 1, "rack": "AACITTZ", "opp_known_leave": "EF",
        "scores": [30, 0], "bag_size": 81, "num_legal_moves": 137, "played": "K6 AC.TA",
        "candidates": [
            candidate | {"move": "9K TIZ", "equity_rank": 0},
            candidate | {"move": "K6 AC.TA", "equity_rank": 61},
        ],
        "confirm": [
            {"candidate": 0, "summary": summary | {"wins": 40}, "win_diff_vs_cut": [[0.0, 0.0]]},
            {"candidate": 1, "summary": summary, "win_diff_vs_cut": [[20.0, 20.0]]},
        ],
    }  # fmt: skip
    entry = position_entry("chunk", position, 10, tmp_path, min_sigmas=2.0)
    assert entry["opp_rack_count"] == 7
    assert [m["move"] for m in entry["moves"]] == ["K6 AC.TA", "9K TIZ"]  # outside play first
    assert entry["moves"][0]["versus"] == "9K TIZ"

"""The blind-spot picks readout (scripts.deep_bot.blind_spot_picks): the kept
positions of a tag's reports, and each pick read against its report."""

import json
from types import SimpleNamespace

import pytest
from scripts.deep_bot import blind_spot_picks as picks


def _candidate(move, win_prob, in_cut):
    return {"move": move, "win_prob": win_prob, "in_cut": in_cut}


def _position(game, bag, outside, kept=True):
    """A report position: two cut moves, then `outside` (the best first)."""
    cut = [_candidate("8H AB", 0.40, True), _candidate("8H BA", 0.50, True)]
    return {
        "game": game,
        "turn": 4,
        "bag_size": bag,
        "kept": kept,
        "best_cut": 1,
        "best_outside": 2,
        "candidates": cut + outside,
    }


@pytest.fixture
def found(tmp_path):
    positions = [
        _position(0, 20, [_candidate("9C SET", 0.60, False), _candidate("9C SETS", 0.55, False)]),
        _position(1, 1, [_candidate("-", 0.90, False)]),
        _position(2, 3, [_candidate("7A UP", 0.70, False)]),
        _position(3, 3, [_candidate("7A UP", 0.70, False)], kept=False),
    ]
    (tmp_path / "a.blindspots.json").write_text(json.dumps({"positions": positions}))
    return tmp_path


def _flags(**changes):
    flags = {"min_bag": 1, "max_bag": 100, "passes": False, "limit": 0, "seed": 0}
    return SimpleNamespace(**(flags | changes))


def _pick(found, game, move):
    return {"slog": str(found / "a.slog"), "game": game, "turn": 4, "pick": move, "seconds": 2.0}


def test_the_kept_positions_are_read_and_passes_left_out_unless_asked(found):
    spots = picks.read_spots(found)
    assert [s.game for s in spots] == [0, 1, 2]
    assert [s.game for s in picks.select(spots, _flags())] == [0, 2]
    assert [s.game for s in picks.select(spots, _flags(passes=True, max_bag=3))] == [1, 2]
    assert len(picks.select(spots, _flags(passes=True, limit=2))) == 2


def test_a_pick_is_read_against_its_positions_report(found):
    spots = picks.read_spots(found)
    table = picks.tabulate(
        spots,
        [_pick(found, 0, "9C SETS"), _pick(found, 2, "7A UP"), _pick(found, 9, "8H AB")],
    )
    assert set(table) == {"bag 8+", "bag 2-3", "all"}
    assert table["bag 8+"]["other_outside"] == 1
    assert table["bag 8+"]["gain"] == pytest.approx(0.05)
    assert table["bag 8+"]["available"] == pytest.approx(0.10)
    assert table["all"]["positions"] == 2  # game 9 is not one of the tag's positions
    assert table["all"]["best_outside"] == 0.5


def test_a_pick_the_report_never_simmed_has_no_gain(found):
    table = picks.tabulate(picks.read_spots(found), [_pick(found, 0, "1A ZZZ")])
    assert (table["all"]["unsimmed"], table["all"]["gain"]) == (1, 0)


def test_the_pre_endgame_ranks_score_a_pick_by_the_draws_it_wins(found, tmp_path):
    spots = picks.read_spots(found)
    rank = {
        "slog": str(found / "a.slog"),
        "game": 2,
        "turn": 4,
        "bag": 3,
        "total": 8,
        "plays": [{"move": "8H BA", "points": 6.0}, {"move": "7A UP", "points": 2.0}],
    }
    (tmp_path / "ranks.jsonl").write_text(json.dumps(rank) + "\n")
    ranks = picks.read_peg_ranks(tmp_path / "ranks.jsonl")
    table = picks.tabulate(spots, [_pick(found, 2, "7A UP"), _pick(found, 0, "9C SET")], ranks)
    assert table["all"]["ranked"] == 1  # game 0 has no ranking
    assert (table["all"]["peg"], table["all"]["peg_best"]) == (0.25, 0.75)

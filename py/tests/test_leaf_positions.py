"""A leaf export's analysis of the known positions (deep_bot_leaf_eval/positions.py)
and the Positions tab's data API over it (dashboard/deep_bot_leaf_eval_api.py):
the matches' pick is the best of the top candidates unless one would empty the
bag, the moves are ranked by the leaf, and the tab lists the best and every
named move of a generation's analysis beside that generation's match."""

import json

from scribblez.dashboard import db
from scribblez.dashboard import deep_bot_leaf_eval_api as api
from scribblez.deep_bot import known_positions
from scribblez.deep_bot_leaf_eval import positions as lp
from scribblez.paths import TagPaths


def _v(move: str, value: float, rank: int) -> dict:
    return {"move": move, "value": value, "equity_rank": rank}


def test_the_pick_is_the_best_of_the_top_candidates():
    # The leaf likes the 4th-ranked move best, but the matches choose among 3.
    record = {
        "pick": "X",
        "values": [_v("a", 0.2, 1), _v("b", 0.5, 2), _v("c", 0.3, 3), _v("d", 0.9, 4)],
    }
    assert lp.match_pick(record, 3) == "b"
    assert lp.match_pick(record, 4) == "d"


def test_where_a_top_candidate_empties_the_bag_the_pick_is_hastybots():
    # Rank 2 empties the bag, so it is not valued; the tool fell back.
    record = {"pick": "hasty", "values": [_v("a", 0.2, 1), _v("c", 0.9, 3)]}
    assert lp.match_pick(record, 3) == "hasty"


def test_with_fewer_moves_than_candidates_the_pick_is_the_records():
    record = {"pick": "b", "values": [_v("a", 0.2, 1), _v("b", 0.5, 2)]}
    assert lp.match_pick(record, 10) == "b"


def test_an_entry_ranks_the_moves_by_the_leaf_and_reads_the_pick():
    tori = next(p for p in known_positions.load() if p.id == "tori")
    record = {
        "pick": "ignored",
        "values": [_v("15D .ROTIC", 0.40, 1), _v("G11 TORI", 0.52, 2), _v("X", 0.1, 3)],
    }
    e = lp.entry(tori, record, 2)
    assert [(v["move"], v["leaf_rank"]) for v in e["values"]] == [
        ("G11 TORI", 1),
        ("15D .ROTIC", 2),
        ("X", 3),
    ]
    assert (e["pick"], e["reading"]) == ("G11 TORI", known_positions.REFERENCE)


def _tag(tmp_path) -> TagPaths:
    return TagPaths("v2", "deep_bot_leaf_eval", tmp_path)


def _write_analysis(paths: TagPaths, generation: int, values: list[dict], pick: str):
    doc = {
        "generation": generation,
        "candidates": 10,
        "positions": {"tori": {"values": values, "pick": pick, "reading": "equity"}},
    }
    lp.write(doc, paths.data_dir / lp.rel(generation))


def test_the_generations_are_the_analyzed_ones_with_their_match(tmp_path):
    paths = _tag(tmp_path)
    _write_analysis(paths, 20, [], "15D .ROTIC")
    _write_analysis(paths, 10, [], "15D .ROTIC")
    paths.root.mkdir(parents=True, exist_ok=True)
    conn = db.connect(paths.dashboard_db)
    db.write_match_eval(
        conn,
        20,
        {
            "positions": 8_000_000,
            "opponent": "--type=hastybot-endgame",
            "games": 10000,
            "wins": 5400,
            "draws": 20,
            "losses": 4580,
            "pair_counts": [0, 0, 5000, 0, 0],
            "score": 0.541,
            "ci_half_width": 0.012,
            "elapsed_s": 70.0,
        },
    )
    conn.close()
    gens = api.generation_list(paths)
    assert [g["generation"] for g in gens] == [10, 20]
    assert gens[0]["score"] is None
    assert (gens[1]["rows_trained"], gens[1]["score"]) == (8_000_000, 0.541)


def test_the_table_lists_the_best_and_every_named_move(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "SHOWN_VALUES", 1)
    paths = _tag(tmp_path)
    values = [
        {**_v("15D .ROTIC", 0.40, 1), "points": 36, "leaf_rank": 1},
        {**_v("8H ZZZ", 0.30, 7), "leaf_rank": 2},
        {**_v("G11 TORI", 0.20, 31), "leaf_rank": 3},
    ]
    _write_analysis(paths, 20, values, "15D .ROTIC")
    index = [p.id for p in known_positions.load()].index("tori")
    p = api.position_payload((paths.tag, paths.task, str(tmp_path)), 20, index)
    # The best, and the reference; not the unnamed second.
    assert [v["move"] for v in p["values"]] == ["15D .ROTIC", "G11 TORI"]
    assert p["values"][1]["display"] == "G11 TORI"
    assert p["values"][0]["points"] == 36  # the pick tool's, passed through
    assert p["num_values"] == 3
    assert (p["pick"]["move"], p["pick"]["reading"]) == ("15D .ROTIC", "equity")
    assert p["match"] is None
    json.dumps(p)  # the handler writes it as JSON


def test_a_moves_history_runs_over_every_analyzed_generation(tmp_path):
    paths = _tag(tmp_path)
    _write_analysis(paths, 10, [{**_v("G11 TORI", 0.30, 2), "leaf_rank": 1}], "G11 TORI")
    # Generation 20 did not value it: a move that would empty the bag.
    _write_analysis(paths, 20, [{**_v("8H ZZZ", 0.50, 1), "leaf_rank": 1}], "8H ZZZ")
    _write_analysis(paths, 30, [{**_v("G11 TORI", 0.45, 1), "leaf_rank": 1}], "G11 TORI")
    key = (paths.tag, paths.task, str(tmp_path))
    assert api.history(key, "tori", "G11 TORI") == [
        {"generation": 10, "value": 0.30},
        {"generation": 20, "value": None},
        {"generation": 30, "value": 0.45},
    ]

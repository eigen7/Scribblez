"""The blind-spots viewer's data: a hidden_blind_spots report and its .gcg
exports turned into the sim-survey viewer's payload."""

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
from scribblez.blind_spots_viewer import BeliefPanel, played_move, readable, viewer_data
from scribblez.survey_viewer_server import survey_handler

GCG = """#character-encoding UTF-8
#lexicon NWL23
#player1 hasty0 hasty0
#player2 hasty1 hasty1
>hasty0: AEINRST 8D RETAINS +68 68
>hasty1: DGOOQUV -QV +0 0
>hasty0: ABCDEFG 9D AB +14 82
"""


def candidate(move: str, rank: int, win: float, in_cut: bool) -> dict:
    return {
        "move": move,
        "display": move,
        "equity_rank": rank,
        "score": 10,
        "leave": "XY",
        "in_cut": in_cut,
        "screen_win_prob": win + 0.01,
        "screen_q": 0.2,
        "win_prob": win,
    }


def position(game: int, turn: int, kept: bool, bag: int = 40) -> dict:
    return {
        "game": game,
        "turn": turn,
        "mover": 0,
        "rack": "ABCDEFG",
        "scores": [82, 0],
        "bag_size": bag,
        "unseen": bag + 7,
        "kept": kept,
        "best_cut": 1,
        "best_outside": 3,
        "gain": 0.08,
        "gain_se": 0.004,
        "candidates": [
            candidate("9D AB", 0, 0.40, True),
            candidate("9E BA", 1, 0.42, True),
            candidate("10A CAB", 30, 0.45, False),
            candidate("H1 FADE", 50, 0.50, False),
        ],
    }


@pytest.fixture
def tag(tmp_path) -> tuple[Path, Path]:
    found, gcg = tmp_path / "found", tmp_path / "gcg"
    found.mkdir()
    gcg.mkdir()
    report = {"version": 1, "cut": 10, "rollouts": 5000, "min_gain": 5.0}
    report["positions"] = [position(0, 2, True), position(1, 2, False), position(2, 9, True, bag=3)]
    (found / "s.blindspots.json").write_text(json.dumps(report))
    # Only the kept positions have games, each running through its own turn.
    (gcg / "s-g0-turn3.gcg").write_text(GCG)
    (gcg / "s-g2-turn10.gcg").write_text(GCG + ">hasty1: AB - +0 0\n" * 7)
    return found, gcg


def test_shows_the_kept_positions_in_name_order(tag):
    data = viewer_data(*tag)
    assert [p["name"] for p in data["positions"]] == ["s-g0-turn3", "s-g2-turn10"]
    assert data["survey"]["cut"] == 10


def test_min_bag_drops_the_late_positions(tag):
    assert [p["name"] for p in viewer_data(*tag, min_bag=8)["positions"]] == ["s-g0-turn3"]


def test_a_position_is_the_board_before_its_turn(tag):
    p = viewer_data(*tag)["positions"][0]
    assert p["turn"] == 3 and p["played"] == "9D AB"
    assert "".join(p["board"][7][3:10]) == "RETAINS" and p["board"][8][3] is None
    assert p["opp_rack_count"] == 100 - 7 - 7 - 40 and p["opp_known_leave"] == ""
    # The opponent's exchange before it, its tiles hidden.
    assert p["last_move"] == {"text": "exch 2", "squares": []}


def test_moves_list_the_outside_plays_first_against_the_best_top_move(tag):
    moves = viewer_data(*tag)["positions"][0]["moves"]
    assert [m["move"] for m in moves] == ["H1 FADE", "10A CAB", "9D AB", "9E BA"]
    fade, cab, top = moves[0], moves[1], moves[2]
    assert fade["outside"] and not top["outside"]
    assert (
        fade["gain_pct"] == pytest.approx(8.0) and fade["beats_cut"] and fade["versus"] == "9E BA"
    )
    assert cab["gain_pct"] == pytest.approx(3.0) and not cab["beats_cut"]
    assert "sigmas" not in fade and "sigmas" not in cab
    assert "gain_pct" not in top and top["hasty_rank"] == 1
    assert fade["win_pct"] == pytest.approx(50.0) and fade["screen_win_pct"] == pytest.approx(51.0)
    assert fade["rollouts"] == 5000 and "stats" not in fade
    assert fade["tiles"][0] == {"row": 0, "col": 7, "letter": "F", "isBlank": False}


def test_passes_and_exchanges_read_as_words():
    assert readable("-") == "pass" and readable("-QV") == "exch QV" and readable("8D AB") == "8D AB"


def test_played_move_reads_plays_exchanges_and_passes():
    assert played_move(GCG) == "9D AB"
    assert played_move(">a: QV -QV +0 0\n") == "-QV"
    assert played_move(">a: QV - +0 0\n") == "-"


class StubBeliefs:
    """LeaveBeliefs' surface BeliefPanel calls, recording what it was asked."""

    def __init__(self):
        self.calls = []

    def beliefs(self, slog, positions, unseen, rack_sizes):
        self.calls.append((slog.name, positions, [int(u.sum()) for u in unseen], rack_sizes))
        return [SimpleNamespace(modelled=True, leave_length=3, refill=s - 3) for s in rack_sizes]

    def rack_presence(self, b, uniform=False):
        return np.full(27, 0.25 if uniform else 0.5)

    def sample_racks(self, b, n, rng):
        return [{"leave": "ABC", "refill": "D" * b.refill}][:n]


def test_the_belief_panel_charts_each_position_and_serves_samples(tag):
    found, gcg = tag
    data = viewer_data(found, gcg)
    model = StubBeliefs()
    panel = BeliefPanel(data, found, model)
    # One query per .slog, at each position's 0-based turn, with its unseen
    # pool (everything off the board and the mover's rack) and the opponent's
    # rack size.
    entries = data["positions"]
    on_board = [sum(c is not None for row in e["board"] for c in row) for e in entries]
    assert model.calls == [
        (
            "s.slog",
            [(0, 2), (2, 9)],
            [100 - n - 7 for n in on_board],
            [e["opp_rack_count"] for e in entries],
        )
    ]
    chart = entries[0]["belief"]
    assert (chart["modelled"], chart["leave_length"]) == (True, 3)
    assert chart["refill"] == entries[0]["opp_rack_count"] - 3
    assert chart["presence"][26] == {"tile": "?", "inferred": 0.5, "uniform": 0.25}
    racks = panel.samples({"position": ["1"], "seed": ["5"]})["racks"]
    assert racks == [{"leave": "ABC", "refill": "D" * (entries[1]["opp_rack_count"] - 3)}]


def test_the_server_answers_routes_and_refuses_bad_queries():
    def route(query):
        return {"echo": int(query["x"][0])}

    server = ThreadingHTTPServer(("127.0.0.1", 0), survey_handler(b"{}", {"/api/r": route}))
    Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert json.loads(urlopen(f"{base}/api/r?x=4").read()) == {"echo": 4}
        assert urlopen(f"{base}/api/survey").read() == b"{}"
        for bad in ("/api/r?x=no", "/api/r", "/api/nothing"):
            with pytest.raises(HTTPError):
                urlopen(base + bad)
    finally:
        server.shutdown()

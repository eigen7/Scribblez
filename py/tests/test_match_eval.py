"""Tests for the match harness and the match_eval role's pure parts."""

import json

import pytest
from scribblez.dashboard import api, db
from scribblez.match_eval import harness, runner
from scribblez.paths import POSITION_EVAL, TagPaths


def _line(seed, seat_players, seat_scores):
    return {
        "seed": seed,
        "seat_players": seat_players,
        "seat_scores": seat_scores,
        "turns": 24,
        "end_reason": "out",
    }


def test_pair_by_seed_pairs_out_of_order_games():
    # Completion order interleaves the pairs; seat assignment flips within each.
    lines = [
        _line(7, [0, 1], [400, 350]),  # player 0 wins from seat 0
        _line(8, [1, 0], [400, 350]),  # player 0 loses from seat 1
        _line(7, [1, 0], [300, 380]),  # player 0 wins from seat 1
        _line(8, [0, 1], [360, 360]),  # draw
    ]
    scores = sorted(harness._pair_by_seed(lines))
    assert scores == [0.25, 1.0]


def test_pair_by_seed_rejects_unpaired_seed():
    with pytest.raises(RuntimeError):
        harness._pair_by_seed([_line(7, [0, 1], [400, 350])])


def test_play_round_parses_results(tmp_path, monkeypatch):
    results = tmp_path / "results.jsonl"

    def fake_run(cmd, capture_output):
        lines = [
            _line(1, [0, 1], [400, 300]),
            _line(1, [1, 0], [300, 400]),
            _line(2, [1, 0], [420, 400]),
            _line(2, [0, 1], [350, 390]),
        ]
        results.write_text("".join(json.dumps(ln) + "\n" for ln in lines))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    r = harness.play_round("--type=a", "--type=b", 2, threads=2, seed=1, results_file=results)
    assert sorted(r.pair_scores) == [0.0, 1.0]
    assert (r.wins, r.draws, r.losses) == (2, 0, 2)


def test_play_round_rejects_seed_zero(tmp_path):
    with pytest.raises(ValueError):
        harness.play_round("a", "b", 1, threads=1, seed=0, results_file=tmp_path / "r.jsonl")


def _match_record(**overrides):
    record = {
        "positions": 1000,
        "opponent": "--type=hastybot-endgame",
        "games": 100,
        "wins": 55,
        "draws": 2,
        "losses": 43,
        "pair_counts": [5, 10, 15, 12, 8],
        "score": 0.56,
        "ci_half_width": 0.04,
        "llr": 1.2,
        "llr_lower": -2.94,
        "llr_upper": 2.94,
        "decision": "continue",
        "elapsed_s": 60.0,
    }
    record.update(overrides)
    return record


def test_match_eval_db_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "dashboard.db")
    db.write_match_eval(conn, 5, _match_record())
    db.write_match_eval(conn, 10, _match_record(decision="H1", llr=3.1))
    # Replaying a generation's match replaces its row.
    db.write_match_eval(conn, 5, _match_record(score=0.5))
    rows = db.read_all_match_eval(conn)
    assert [r["epoch"] for r in rows] == [5, 10]
    assert rows[0]["score"] == 0.5
    assert rows[0]["pair_counts"] == [5, 10, 15, 12, 8]
    assert rows[1]["decision"] == "H1"


def test_match_eval_figure_is_serializable(tmp_path):
    conn = db.connect(tmp_path / "dashboard.db")
    assert api.build_figure_item(conn, "match_eval", {}, tmp_path, str(tmp_path)) is None
    db.write_match_eval(conn, 5, _match_record())
    db.write_match_eval(conn, 10, _match_record(decision="H1", llr=3.2))
    item = api.build_figure_item(conn, "match_eval", {}, tmp_path, str(tmp_path))
    assert item is not None
    assert {"doc", "root_id", "target_id"} <= set(item)


def test_pending_generation_prefers_the_frontier(tmp_path):
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    conn = db.connect(paths.dashboard_db)
    paths.onnx_dir.mkdir(parents=True)
    for gen in (4, 5, 10, 15, 16):
        paths.onnx_path(gen).touch()

    assert runner._pending_generation(paths, conn, every=5) == 15
    db.write_match_eval(conn, 15, _match_record())
    assert runner._pending_generation(paths, conn, every=5) == 10
    db.write_match_eval(conn, 10, _match_record())
    db.write_match_eval(conn, 5, _match_record())
    assert runner._pending_generation(paths, conn, every=5) is None
    # every=1 picks up the non-multiple stragglers too.
    assert runner._pending_generation(paths, conn, every=1) == 16

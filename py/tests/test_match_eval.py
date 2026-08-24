"""Tests for the match harness and the match_eval role's worker half."""

import json
from pathlib import Path

import pytest
from cloud.sinks import LocalSink
from scribblez import selfplay
from scribblez.dashboard import api, db
from scribblez.match_eval import harness, runner
from scribblez.match_eval.harness import RoundResult
from scribblez.paths import DONE_SUFFIX, POSITION_EVAL, TagPaths
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.position_eval import SPEC, PositionEvalParams


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


def _fake_play_game(results_file, lines, returncode=0, expect_games=None):
    """A subprocess.run stand-in that checks the paired-match command line and
    writes `lines` as the results file (`expect_games` defaults to their count;
    pass it explicitly to simulate an engine delivering the wrong number)."""

    def fake_run(cmd, capture_output):
        assert cmd[0] == selfplay.PLAY_GAME
        assert "--paired" in cmd
        assert cmd[cmd.index("--games") + 1] == str(expect_games or len(lines))
        assert cmd[cmd.index("--seed") + 1] == "1"
        assert cmd[cmd.index("--results-file") + 1] == str(results_file)
        results_file.write_text("".join(json.dumps(ln) + "\n" for ln in lines))
        return type("R", (), {"returncode": returncode})()

    return fake_run


def test_play_round_parses_results(tmp_path, monkeypatch):
    results = tmp_path / "results.jsonl"
    lines = [
        _line(1, [0, 1], [400, 300]),
        _line(1, [1, 0], [300, 400]),
        _line(2, [1, 0], [420, 400]),
        _line(2, [0, 1], [350, 390]),
    ]
    monkeypatch.setattr(selfplay.subprocess, "run", _fake_play_game(results, lines))
    r = harness.play_round("--type=a", "--type=b", 2, threads=2, seed=1, results_file=results)
    assert sorted(r.pair_scores) == [0.0, 1.0]
    assert (r.wins, r.draws, r.losses) == (2, 0, 2)


def test_play_round_rejects_seed_zero(tmp_path):
    with pytest.raises(ValueError):
        harness.play_round("a", "b", 1, threads=1, seed=0, results_file=tmp_path / "r.jsonl")


def test_play_round_raises_on_engine_failure(tmp_path, monkeypatch):
    results = tmp_path / "results.jsonl"
    lines = [_line(1, [0, 1], [400, 300]), _line(1, [1, 0], [300, 400])]
    monkeypatch.setattr(selfplay.subprocess, "run", _fake_play_game(results, lines, returncode=3))
    with pytest.raises(RuntimeError, match="exit code 3"):
        harness.play_round("a", "b", 1, threads=1, seed=1, results_file=results)


def test_play_round_raises_on_missing_game_records(tmp_path, monkeypatch):
    results = tmp_path / "results.jsonl"
    lines = [_line(1, [0, 1], [400, 300]), _line(1, [1, 0], [300, 400])]
    monkeypatch.setattr(selfplay.subprocess, "run", _fake_play_game(results, lines, expect_games=4))
    with pytest.raises(RuntimeError, match="expected 4 game records"):
        harness.play_round("a", "b", 2, threads=1, seed=1, results_file=results)


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
        "elapsed_s": 60.0,
    }
    record.update(overrides)
    return record


def test_match_eval_db_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "dashboard.db")
    db.write_match_eval(conn, 5, _match_record())
    db.write_match_eval(conn, 10, _match_record(score=0.61))
    # Replaying a generation's match replaces its row.
    db.write_match_eval(conn, 5, _match_record(score=0.5))
    rows = db.read_all_match_eval(conn)
    assert [r["epoch"] for r in rows] == [5, 10]
    assert rows[0]["score"] == 0.5
    assert rows[0]["pair_counts"] == [5, 10, 15, 12, 8]
    assert rows[1]["score"] == 0.61


def test_match_eval_figure_is_serializable(tmp_path):
    conn = db.connect(tmp_path / "dashboard.db")
    assert api.build_figure_item(conn, "match_eval", {}, str(tmp_path)) == (None, None)
    db.write_match_eval(conn, 5, _match_record())
    db.write_match_eval(conn, 10, _match_record(score=0.61))
    item, _structure = api.build_figure_item(conn, "match_eval", {}, str(tmp_path))
    assert item is not None
    assert {"doc", "root_id", "target_id"} <= set(item)


def _ctx(sink=None, **param_overrides) -> WorkerContext:
    return WorkerContext(
        spec=SPEC,
        role=SPEC.role("match_eval"),
        tag="t",
        params=PositionEvalParams(**param_overrides),
        worker_id="w0",
        threads=2,
        max_cycles=1,
        sink=sink,
    )


_MODEL = Path("/models/model_epoch_0003.onnx")


def test_play_match_plays_the_whole_fixed_budget(monkeypatch):
    calls = []

    def fake_round(spec0, spec1, num_pairs, threads, seed, results_file, face_up_leaves):
        calls.append((num_pairs, seed))
        return RoundResult([1.0] * num_pairs, wins=2 * num_pairs, draws=0, losses=0)

    monkeypatch.setattr(runner.harness, "play_round", fake_round)
    outcome = runner._play_match(_ctx(match_pairs=5, match_seed=7), _MODEL)
    assert calls == [(5, 7)]
    assert outcome.pair_counts == [0, 0, 0, 0, 5]
    assert (outcome.wins, outcome.draws, outcome.losses) == (10, 0, 0)
    assert outcome.games == 10


def test_assigned_model_reads_the_inbox(tmp_path):
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    inbox = paths.match_inbox_dir("w0")
    assert runner._assigned_model(paths, "w0") is None  # no inbox yet: idle
    inbox.mkdir(parents=True)
    assert runner._assigned_model(paths, "w0") is None
    (inbox / "lexicon_frozen.bin").touch()  # a sidecar is not an assignment
    assert runner._assigned_model(paths, "w0") is None
    # Nor is a model this worker has already played, still there for the
    # controller's benefit.
    (inbox / f"{paths.onnx_path(3).name}{DONE_SUFFIX}").touch()
    assert runner._assigned_model(paths, "w0") is None
    (inbox / paths.onnx_path(5).name).touch()
    (inbox / paths.onnx_path(10).name).touch()
    assigned = runner._assigned_model(paths, "w0")
    assert paths.onnx_epoch(assigned) == 10  # newest first


def test_run_delivers_the_result_and_marks_the_model_played(tmp_path, monkeypatch):
    paths = TagPaths("t", POSITION_EVAL, mount_root=tmp_path)
    inbox = paths.match_inbox_dir("w0")
    inbox.mkdir(parents=True)
    model = inbox / paths.onnx_path(20).name
    model.touch()
    monkeypatch.setattr(WorkerContext, "tag_paths", lambda self: paths)

    played = []

    def fake_round(spec0, spec1, num_pairs, threads, seed, results_file, face_up_leaves):
        played.append(spec0)
        return RoundResult([1.0] * num_pairs, wins=2 * num_pairs, draws=0, losses=0)

    monkeypatch.setattr(runner.harness, "play_round", fake_round)
    ctx = _ctx(sink=LocalSink(paths.root), match_pairs=4)
    assert runner.run(ctx) == 0

    assert str(model) in played[0]  # the assigned model is what was played
    # Marked only after delivery, and marked rather than removed: the
    # controller still needs the generation to count as spoken for until the
    # result reaches it (match_eval/dispatch.py).
    assert [p.name for p in inbox.iterdir()] == [f"{model.name}{DONE_SUFFIX}"]
    delivered = list(paths.match_results_dir.glob("*.json"))
    assert [p.name for p in delivered] == ["gen_000020-w0.json"]
    record = json.loads(delivered[0].read_text())
    assert record["epoch"] == 20
    assert record["games"] == 8
    assert "positions" not in record  # the controller's column, not the worker's

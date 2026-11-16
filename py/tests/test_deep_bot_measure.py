"""The deep_bot measure role: the controller's assignments and its record of
the matches delivered (scribblez.deep_bot.measure_dispatch), the worker's
matches (measure), and the estimate from kept games (matches)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from cloud.sinks import LocalSink
from scribblez import workloads
from scribblez.dashboard import db, plots
from scribblez.dashboard.slot_files import LocalSlotFiles
from scribblez.deep_bot import matches, measure, measure_dispatch, release
from scribblez.generational import lifecycle
from scribblez.match_eval import harness
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.deep_bot import DeepBotParams
from scripts.deep_bot import match as match_script

SPEC = workloads.get("deep_bot")
PARAMS = DeepBotParams(match_games=4, match_seed=7)
WORKER = "local-9"


def _publish(paths, number: int):
    for rel in release.files(number):
        (paths.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (paths.root / rel).write_text(f"{number}")


def _games(wins: int, pairs: int = 2, seed: int = 7) -> list[dict]:
    """play_game's result lines for `pairs` mirrored pairs in which player 0
    wins the first `wins` games and loses the rest."""
    lines = []
    for g in range(2 * pairs):
        seats = [0, 1] if g % 2 == 0 else [1, 0]
        own, opp = (400, 300) if g < wins else (300, 400)
        scores = [own, opp] if seats[0] == 0 else [opp, own]
        lines.append({"seed": seed + g // 2, "seat_players": seats, "seat_scores": scores})
    return lines


def _deliver(paths, number: int, opponent: str, wins: int):
    """A finished match in the tag's data dirs, as a worker delivers it."""
    for rel, text in (
        (matches.games_rel(number, opponent), "slog"),
        (
            matches.result_rel(number, opponent),
            json.dumps(matches.result_record(number, opponent, _games(wins), 12.5)),
        ),
    ):
        (paths.data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (paths.data_dir / rel).write_text(text)


@pytest.fixture
def paths(tmp_path):
    paths = SPEC.paths("t", tmp_path)
    paths.root.mkdir(parents=True)
    db.connect(paths.dashboard_db).close()
    return paths


def _slot(paths) -> LocalSlotFiles:
    return LocalSlotFiles(WORKER, paths.root)


def _inbox(paths) -> Path:
    return paths.match_inbox_dir(WORKER)


def _tick(paths, params=PARAMS, slots=None) -> bool:
    return measure_dispatch.tick(SPEC, paths, params, [_slot(paths)] if slots is None else slots)


def test_the_estimate_is_the_mean_pair_score_of_the_kept_games():
    summary = matches.estimate(_games(wins=3))
    assert (summary["wins"], summary["losses"], summary["games"]) == (3, 1, 4)
    assert summary["score"] == pytest.approx(0.75)
    assert sum(summary["pair_counts"]) == 2
    assert matches.opponents_of(0) == ("hastybot",)
    assert matches.opponents_of(5) == ("hastybot", "release_0")


def test_the_newest_release_is_assigned_with_release_0_beside_it(paths):
    _publish(paths, 0)
    _publish(paths, 1)
    _publish(paths, 2)
    assert _tick(paths)
    inbox = _inbox(paths)
    assert sorted(p.name for p in (inbox / measure.ASSIGNED_DIR).iterdir()) == [
        "release_000002.json"
    ]
    for number in (0, 2):
        assert all((inbox / rel).read_text() == f"{number}" for rel in release.files(number))
    assert measure.assigned_release(inbox) == 2

    # Busy until its results are recorded: nothing more is assigned.
    _tick(paths)
    assert len(list((inbox / measure.ASSIGNED_DIR).iterdir())) == 1


def test_delivered_matches_are_recorded_kept_and_cleared_from_the_inbox(paths):
    _publish(paths, 0)
    _publish(paths, 1)
    _tick(paths)
    inbox = _inbox(paths)
    token = inbox / measure.ASSIGNED_DIR / "release_000001.json"
    token.rename(token.with_name(token.name + ".done"))  # the worker finished it
    _deliver(paths, 1, matches.HASTYBOT, wins=3)

    _tick(paths)
    # One of release 1's two matches is in: it is not yet measured, so its
    # assignment stays and release 0 waits its turn.
    assert token.with_name(token.name + ".done").exists()
    _deliver(paths, 1, matches.RELEASE_0, wins=2)
    assert _tick(paths)  # release 0 is still owed its match

    conn = db.connect(paths.dashboard_db)
    rows = {(r["release"], r["opponent"]): r for r in db.read_release_matches(conn)}
    conn.close()
    assert rows[(1, "hastybot")]["score"] == pytest.approx(0.75)
    assert rows[(1, "release_0")]["wins"] == 2
    kept = matches.kept_dir(paths, 1)
    assert sorted(p.name for p in kept.iterdir()) == [
        "hastybot.json", "hastybot.slog", "release_0.json", "release_0.slog"
    ]  # fmt: skip
    assert len(matches.read_result(kept / "hastybot.json")["games"]) == 4
    assert not any(paths.match_results_dir.iterdir())
    # Release 1's files left the inbox with its assignment; release 0's stay,
    # and release 0 is now assigned.
    assert not (inbox / release.files(1)[0]).exists()
    assert (inbox / release.files(0)[0]).exists()
    assert measure.assigned_release(inbox) == 0


def test_nothing_is_owed_once_every_due_release_is_measured(paths):
    _publish(paths, 0)
    lifecycle.write_train_state(paths, {"generation_index": 0, "rows_trained": 0})
    _deliver(paths, 0, matches.HASTYBOT, wins=2)
    assert not _tick(paths, slots=[])
    # A trainer ahead of the releases here means one is still on its way.
    lifecycle.write_train_state(paths, {"generation_index": 1, "rows_trained": 40})
    assert _tick(paths, slots=[])
    # With measuring off, results still land and nothing is assigned.
    off = replace(PARAMS, match_every_generations=0)
    assert not _tick(paths, params=off)
    assert not _inbox(paths).exists()


def test_an_unreadable_result_is_quarantined(paths):
    _publish(paths, 0)
    paths.match_results_dir.mkdir(parents=True)
    bad = paths.match_results_dir / "release_000000-hastybot.json"
    bad.write_text('{"release": 0}')
    _tick(paths, slots=[])
    assert not bad.exists() and bad.with_suffix(".bad").exists()


def test_the_worker_plays_each_opponent_and_delivers_every_game(paths, monkeypatch):
    _publish(paths, 0)
    _publish(paths, 3)
    _tick(paths)
    played = []

    def play_round(player0, player1, num_pairs, threads, seed, results_file, slog_dir):
        played.append((player0, player1, num_pairs, seed))
        results_file.write_text("\n".join(json.dumps(g) for g in _games(3, num_pairs, seed)))
        slog_dir.mkdir(parents=True)
        (slog_dir / "1.slog").write_text("games")
        return harness.round_result(_games(3, num_pairs, seed))

    monkeypatch.setattr(harness, "play_round", play_round)
    ctx = WorkerContext(
        spec=SPEC,
        role=SPEC.role("measure"),
        tag="t",
        params=PARAMS,
        worker_id=WORKER,
        threads=2,
        max_cycles=1,
        sink=LocalSink(paths.root),
        mount_root=paths.mount_root,
    )
    assert measure.run(ctx) == 0

    inbox = _inbox(paths)
    (hasty, own) = played
    assert f"--leaf-model {inbox / release.export_rel(3, 'leaf')}" in hasty[0]
    assert hasty[1:] == (matches.HASTYBOT_SPEC, 2, 7)
    assert f"--leaf-model {inbox / release.export_rel(0, 'leaf')}" in own[1]
    assert measure.assigned_release(inbox) is None  # marked done, awaiting the controller
    _tick(paths)
    kept = sorted(p.name for p in matches.kept_dir(paths, 3).iterdir())
    assert kept == ["hastybot.json", "hastybot.slog", "release_0.json", "release_0.slog"]


def test_the_chart_draws_a_panel_per_opponent_with_a_trailing_mean(paths):
    assert plots.trailing_mean([1.0, 0.0, 1.0, 1.0], window=2) == [1.0, 0.5, 0.5, 1.0]
    conn = db.connect(paths.dashboard_db)
    assert plots.release_match_grid(conn) is None
    for number in (1, 2):
        for opponent in matches.OPPONENTS:
            db.write_release_match(
                conn, number, opponent, matches.estimate(_games(3)) | {"elapsed_s": 1.0}
            )
    grid = plots.release_match_grid(conn)
    conn.close()
    assert [f.title.text for f in grid.children] == [
        "Win rate vs HastyBot", "Win rate vs release 0"
    ]  # fmt: skip


def test_a_match_on_demand_is_filed_under_its_release(paths, monkeypatch, capsys):
    """scripts.deep_bot.match: a release against a named opponent, delivered
    as a measure slot's match is and recorded by the controller's next pass."""
    _publish(paths, 0)
    _publish(paths, 2)
    (paths.root / "task.json").write_text(json.dumps({"params": {"match_games": 4}}))
    played = []

    def play_round(player0, player1, num_pairs, threads, seed, results_file, slog_dir):
        played.append((player1, num_pairs, seed))
        results_file.write_text("\n".join(json.dumps(g) for g in _games(3, num_pairs, seed)))
        slog_dir.mkdir(parents=True)
        (slog_dir / "1.slog").write_text("games")

    monkeypatch.setattr(harness, "play_round", play_round)
    argv = ["--tag", "t", "--release", "2", "--mount-root", str(paths.mount_root)]
    spec = "--type=bestbot-endgame"
    assert match_script.main([*argv, "--opponent", "bestbot", "--player", spec]) == 0
    assert played == [(spec, 2, 1)]  # the tag's match_games, and its default seed
    assert "vs bestbot: 3/0/1" in capsys.readouterr().out

    _tick(paths, slots=[])
    kept = sorted(p.name for p in matches.kept_dir(paths, 2).iterdir())
    assert kept == ["bestbot.json", "bestbot.slog"]
    conn = db.connect(paths.dashboard_db)
    assert [(r["release"], r["opponent"]) for r in db.read_release_matches(conn)] == [
        (2, "bestbot")
    ]
    conn.close()

    # The loop's own opponents and a missing release are refused.
    with pytest.raises(SystemExit):
        match_script.main([*argv, "--opponent", "hastybot", "--player", spec])
    assert match_script.main([*argv[:3], "9", *argv[4:], "--opponent", "x", "--player", spec]) == 1

"""The leaf_bot_generation workload (scribblez/workloads/leaf_bot_generation.py)
and its measure role: seeds copied at creation, generators on release 0, and
the matches spread over several measure slots (leaf_bot_generation/
measure_dispatch.py) and played by each (measure.py)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from cloud.sinks import LocalSink
from scribblez import workloads
from scribblez.dashboard import api, db, deep_bot_leaf_eval_api, plots
from scribblez.dashboard.slot_files import LocalSlotFiles
from scribblez.deep_bot import LEAF, LEAVE, known_positions, matches, release, seeds
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.generational import lifecycle
from scribblez.leaf_bot_generation import KINDS, LAYOUT, measure, measure_dispatch
from scribblez.match_eval import harness
from scribblez.params import ParamsError
from scribblez.paths import DONE_SUFFIX, TagPaths
from scribblez.workloads import leaf_bot_generation, selfplay_gen
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.leaf_bot_generation import LeafBotGenerationParams
from scribblez.workloads.selfplay_gen import ChunkSidecar

SPEC = workloads.get("leaf_bot_generation")
PARAMS = LeafBotGenerationParams(
    seed_leave_tag="i", seed_leaf_tag="l", match_games=4, match_seed=7, match_every_generations=2
)


def _checkpoint(paths: TagPaths, text: str) -> Path:
    """A stand-in rolling checkpoint of the tag at `paths`."""
    paths.rolling_checkpoint.parent.mkdir(parents=True)
    paths.rolling_checkpoint.write_text(text)
    return paths.rolling_checkpoint


def _publish(paths: TagPaths, number: int):
    """Release `number`, each file holding its release's number."""
    for rel in LAYOUT.files(number):
        (paths.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (paths.root / rel).write_text(f"{number}")


def test_a_tag_takes_one_kind_of_seed():
    with pytest.raises(ParamsError, match="not both"):
        LeafBotGenerationParams(seed_tag="g1", seed_leaf_tag="l")
    with pytest.raises(ParamsError, match="even"):
        replace(PARAMS, match_games=3)
    assert replace(PARAMS, leave_window=2, leaf_window=5).window == 5


def test_creating_a_tag_copies_the_standalone_seeds(tmp_path):
    for kind in KINDS:
        tag = getattr(PARAMS, f"seed_{kind}_tag")
        _checkpoint(TagPaths(tag, seeds.WORKLOADS[kind], tmp_path), kind)
    paths = SPEC.paths("t", tmp_path)

    assert leaf_bot_generation.finalize(SPEC, paths, PARAMS) == PARAMS

    for kind in KINDS:
        assert seeds.seed_path(paths, kind).read_text() == kind
    assert SPEC.role("train").tag_files
    assert workloads.resolve(SPEC.role("train").tag_files)(paths, PARAMS) == [
        "seeds/leave.pt",
        "seeds/leaf.pt",
    ]


def test_a_tag_seeded_from_the_previous_generation_copies_its_checkpoint_for_both(tmp_path):
    _checkpoint(SPEC.paths("g1", tmp_path), "generation 1")
    paths = SPEC.paths("g2", tmp_path)
    params = LeafBotGenerationParams(seed_tag="g1")

    leaf_bot_generation.finalize(SPEC, paths, params)

    assert [seeds.seed_path(paths, kind).read_text() for kind in KINDS] == ["generation 1"] * 2


def test_a_missing_seed_refuses_the_tag_before_anything_is_copied(tmp_path):
    _checkpoint(TagPaths("i", seeds.WORKLOADS[LEAVE], tmp_path), LEAVE)
    paths = SPEC.paths("t", tmp_path)
    with pytest.raises(ParamsError, match="deep_bot_leaf_eval tag 'l' has no checkpoint"):
        leaf_bot_generation.finalize(SPEC, paths, PARAMS)
    with pytest.raises(ParamsError, match="leaf_bot_generation tag 'g0' has no checkpoint"):
        leaf_bot_generation.finalize(SPEC, paths, LeafBotGenerationParams(seed_tag="g0"))
    assert not seeds.seed_path(paths, LEAVE).exists()


def _ctx(tmp_path, role: str, max_cycles: int = 1, worker_id: str = "local-0") -> WorkerContext:
    return WorkerContext(
        spec=SPEC,
        role=SPEC.role(role),
        tag="t",
        params=PARAMS,
        worker_id=worker_id,
        threads=2,
        max_cycles=max_cycles,
        sink=LocalSink(SPEC.data_dir("t", tmp_path)),
        mount_root=tmp_path,
    )


def test_every_cycle_plays_leafbot_on_release_0_with_its_forgone_plays(tmp_path, monkeypatch):
    paths = SPEC.paths("t", tmp_path)
    _publish(paths, 0)
    played = []

    def run_games(out_dir, player_spec, **kwargs):
        played.append(player_spec)
        (out_dir / f"{out_dir.name}.slog").write_bytes(b"x")
        if len(played) == 1:
            _publish(paths, 1)  # the trainer publishes while the first chunk plays
        return 0

    def write_forgone(chunk_dir, threads):
        for slog in chunk_dir.glob("*.slog"):
            slog.with_suffix(".forgone").write_bytes(b"x")

    monkeypatch.setattr(selfplay_gen, "run_games", run_games)
    sidecar = ChunkSidecar((".forgone",), write_forgone)
    monkeypatch.setattr(leaf_bot_generation, "FORGONE_SIDECAR", sidecar)

    assert leaf_bot_generation.run_generate(_ctx(tmp_path, "generate", max_cycles=2)) == 0

    leaf, leave = (release.export_path(paths, 0, kind) for kind in (LEAF, LEAVE))
    want = (
        f"--type=deepbot-endgame --leaf-model={leaf} --leave-inference={leave} "
        "--candidates=10 --opp-leaves=16"
    )
    assert played == [want, want]  # release 1 changes nothing
    staged = sorted(p.suffix for p in (paths.data_dir / "staging").iterdir())
    assert staged == [".forgone", ".forgone", ".slog", ".slog"]
    tag_files = workloads.resolve(SPEC.role("generate").tag_files)
    assert tag_files(paths, PARAMS) == [
        "releases/release_000000/leave.onnx",
        "releases/release_000000/leaf.onnx",
        "releases/release_000000/manifest.json",
    ]


def test_a_generator_without_release_0_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="no release 0"):
        leaf_bot_generation.run_generate(_ctx(tmp_path, "generate"))
    assert workloads.resolve(SPEC.role("generate").tag_files)(SPEC.paths("t", tmp_path), None) == []


@pytest.fixture
def paths(tmp_path):
    paths = SPEC.paths("t", tmp_path)
    paths.root.mkdir(parents=True)
    db.connect(paths.dashboard_db).close()
    return paths


def _slots(paths, n: int) -> list[LocalSlotFiles]:
    return [LocalSlotFiles(f"local-{i}", paths.root) for i in range(n)]


def _assigned(paths, slot) -> list[str]:
    folder = paths.match_inbox_dir(slot.worker_id) / measure.ASSIGNED_DIR
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def _tick(paths, slots, params=PARAMS) -> bool:
    return measure_dispatch.tick(SPEC, paths, params, slots)


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


def _finish(paths, slot, wins: int = 3):
    """What a slot's worker does with its assignment: deliver the match's
    rows and mark the assignment played."""
    inbox = paths.match_inbox_dir(slot.worker_id)
    number, opponent = measure.assigned_match(inbox)
    record = matches.result_record(number, opponent, _games(wins), 1.0)
    path = paths.data_dir / matches.result_rel(number, opponent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    token = inbox / measure.ASSIGNED_DIR / measure.token_name(number, opponent)
    token.rename(token.with_name(token.name + DONE_SUFFIX))


def test_the_due_matches_go_to_different_slots_newest_release_first(paths):
    for number in range(5):
        _publish(paths, number)
    slots = _slots(paths, 4)

    assert _tick(paths, slots)

    # Releases 4, 2 and 0 are due (every 2nd); release 0 plays HastyBot only.
    assert [_assigned(paths, s) for s in slots] == [
        ["release_000004-hastybot.json"],
        ["release_000004-release_0.json"],
        ["release_000002-hastybot.json"],
        ["release_000002-release_0.json"],
    ]
    inbox = paths.match_inbox_dir("local-1")
    for number in (0, 4):
        assert all((inbox / rel).read_text() == f"{number}" for rel in LAYOUT.files(number))
    assert measure.assigned_match(inbox) == (4, "release_0")

    # Every slot is busy until its result is recorded.
    _tick(paths, slots)
    assert sum(len(_assigned(paths, s)) for s in slots) == 4


def test_a_recorded_match_frees_its_slot_for_the_next_one(paths):
    for number in range(3):
        _publish(paths, number)
    slots = _slots(paths, 1)
    _tick(paths, slots)
    _finish(paths, slots[0])

    assert _tick(paths, slots)

    conn = db.connect(paths.dashboard_db)
    rows = db.read_release_matches(conn)
    conn.close()
    assert [(r["release"], r["opponent"], r["score"]) for r in rows] == [(2, "hastybot", 0.75)]
    assert _assigned(paths, slots[0]) == ["release_000002-release_0.json"]
    # Release 0 stays in the inbox; it is every assignment's opponent.
    assert (paths.match_inbox_dir("local-0") / LAYOUT.files(0)[-1]).is_file()


def test_nothing_is_owed_once_every_due_release_is_measured(paths):
    for number in range(3):
        _publish(paths, number)
    slots = _slots(paths, 2)
    params = replace(PARAMS, match_every_generations=2)
    for _ in range(2):  # three matches over two slots
        _tick(paths, slots, params)
        for slot in slots:
            if any(not n.endswith(DONE_SUFFIX) for n in _assigned(paths, slot)):
                _finish(paths, slot)
    lifecycle.write_train_state(paths, {"generation_index": 3})
    assert _tick(paths, slots, params)  # release 3 is still on its way from the trainer
    lifecycle.write_train_state(paths, {"generation_index": 2})

    assert not _tick(paths, slots, params)

    # Spent assignments and their release are cleared lazily, at the next
    # assignment; nothing new is assigned.
    assert all(n.endswith(DONE_SUFFIX) for s in slots for n in _assigned(paths, s))


def test_a_quarantined_result_falls_due_again(paths):
    _publish(paths, 0)
    slots = _slots(paths, 1)
    _tick(paths, slots)
    inbox = paths.match_inbox_dir("local-0")
    path = paths.data_dir / matches.result_rel(0, matches.HASTYBOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json")
    token = inbox / measure.ASSIGNED_DIR / measure.token_name(0, matches.HASTYBOT)
    token.rename(token.with_name(token.name + DONE_SUFFIX))

    assert _tick(paths, slots)

    assert path.with_suffix(".bad").exists()
    assert _assigned(paths, slots[0]) == ["release_000000-hastybot.json"]


def test_the_worker_plays_its_match_and_analyzes_with_hastybots(paths, tmp_path, monkeypatch):
    _publish(paths, 0)
    _publish(paths, 2)
    slot = _slots(paths, 1)[0]
    _tick(paths, [slot])
    played = []

    def play_round(player0, player1, num_pairs, threads, seed, results_file, slog_dir):
        played.append((player0, player1, num_pairs, seed, slog_dir))
        results_file.write_text("\n".join(json.dumps(g) for g in _games(wins=3)))

    analyzed = []

    def ask(positions, player, work, threads):
        analyzed.append(player)
        return [{"pick": "", "values": []} for _ in positions]

    monkeypatch.setattr(harness, "play_round", play_round)
    monkeypatch.setattr(known_positions, "ask", ask)
    ctx = _ctx(tmp_path, "measure", max_cycles=1)

    assert measure.run(ctx) == 0

    inbox = paths.match_inbox_dir("local-0")
    (player0, player1, pairs, seed, slog_dir) = played[0]
    assert f"--leaf-model={inbox / release.export_rel(2, LEAF)}" in player0
    assert "--candidates=10 --opp-leaves=16" in player0
    assert player1 == matches.HASTYBOT_SPEC
    assert (pairs, seed, slog_dir) == (2, 7, None)  # the games are not kept
    (spec,) = analyzed
    assert f"--leave-inference={inbox / release.export_rel(2, LEAVE)}" in spec
    assert "--opp-leaves=128" in spec
    analysis = json.loads((paths.data_dir / leaf_positions.rel(2)).read_text())
    assert analysis["generation"] == 2
    assert measure.assigned_match(inbox) is None  # marked played

    _tick(paths, [slot])
    # The Positions tab's slider reads the release's match against HastyBot.
    (entry,) = deep_bot_leaf_eval_api.generation_list(paths)
    assert (entry["generation"], entry["rows_trained"], entry["score"]) == (2, None, 0.75)
    assert _assigned(paths, slot) == ["release_000002-release_0.json"]


def test_the_match_against_release_0_plays_its_leafbot(paths, tmp_path, monkeypatch):
    _publish(paths, 0)
    _publish(paths, 2)
    slots = _slots(paths, 2)
    _tick(paths, slots)
    played = []

    def play_round(player0, player1, num_pairs, threads, seed, results_file, slog_dir):
        played.append(player1)
        results_file.write_text("\n".join(json.dumps(g) for g in _games(wins=1)))

    monkeypatch.setattr(harness, "play_round", play_round)

    assert measure.run(_ctx(tmp_path, "measure", worker_id="local-1")) == 0

    inbox = paths.match_inbox_dir("local-1")
    assert played == [
        f"--type=deepbot-endgame --leaf-model={inbox / release.export_rel(0, LEAF)} "
        f"--leave-inference={inbox / release.export_rel(0, LEAVE)} "
        "--candidates=10 --opp-leaves=16"
    ]
    assert not (paths.data_dir / leaf_positions.rel(2)).exists()  # analyzed with HastyBot's


def test_the_training_tab_draws_each_model_beside_release_0(paths):
    keys = {k for group in plots.LEAF_BOT_GENERATION_QUALITY for k in group[1]}
    assert {"leave_eval_nll_vs_uniform", "r0_leaf_eval_wld_nll", "loss_leaf"} <= keys
    conn = db.connect(paths.dashboard_db)
    assert api.FIGURES["leaf_bot_generation_metrics"](conn, {}, str(paths.mount_root)) is None
    db.write_metrics(conn, 0, {"positions": 1, **dict.fromkeys(keys, 0.5)})
    assert api.FIGURES["leaf_bot_generation_metrics"](conn, {}, str(paths.mount_root)) is not None
    conn.close()

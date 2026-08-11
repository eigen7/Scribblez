"""Tests for the match_arms workload: the arms mini-format, params validation,
the runner's per-arm cycle discipline, and the db/figure plumbing."""

import pytest
from scribblez import params as params_mod
from scribblez.dashboard import api, db
from scribblez.match_eval import arms as arms_runner
from scribblez.match_eval.harness import RoundResult
from scribblez.params import ParamsError
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.match_arms import SPEC, MatchArmsParams, parse_arms

# ---------------------------------------------------------------------------
# The arms mini-format
# ---------------------------------------------------------------------------


def test_parse_arms_splits_on_the_first_equals_only():
    # Player specs are full of '='; only the first one separates the name.
    text = "k5=--type=neural-sim --model=m.onnx --sim-top-k=5; base=--type=sim --top-k=10"
    parsed = parse_arms(text)
    assert [(a.name, a.player_spec) for a in parsed] == [
        ("k5", "--type=neural-sim --model=m.onnx --sim-top-k=5"),
        ("base", "--type=sim --top-k=10"),
    ]


def test_parse_arms_tolerates_whitespace_and_trailing_semicolon():
    parsed = parse_arms("  a = --type=greedy ;  b = --type=sim ; ")
    assert [a.name for a in parsed] == ["a", "b"]


def test_parse_arms_allows_the_default_empty_experiment():
    # Params must stay default-constructible (the registry-wide contract test
    # constructs every workload's params from defaults), so "" parses to an
    # empty experiment rather than raising.
    assert parse_arms("") == []


@pytest.mark.parametrize(
    "text",
    [
        "a=--type=greedy; a=--type=sim",  # duplicate name
        "--type=greedy",  # forgotten name: would parse as name "--type"
        "a=",  # empty spec
        "=--type=greedy",  # empty name
    ],
)
def test_parse_arms_rejects_malformed_input(text):
    with pytest.raises(ParamsError):
        parse_arms(text)


# ---------------------------------------------------------------------------
# Params validation (at task creation, not in the runner)
# ---------------------------------------------------------------------------


def test_params_validate_accepts_a_good_experiment():
    p = params_mod.validate(MatchArmsParams, {"arms": "a=--type=greedy; b=--type=sim", "seed": 7})
    assert [a.name for a in parse_arms(p.arms)] == ["a", "b"]


@pytest.mark.parametrize(
    "raw",
    [
        {"arms": "a=--type=greedy; a=--type=sim"},
        {"arms": "a=--type=greedy", "seed": 0},
        {"arms": "a=--type=greedy", "pairs_per_arm": 0},
        {"arms": "a=--type=greedy", "round_pairs": 0},
    ],
)
def test_params_validate_rejects_bad_experiments(raw):
    with pytest.raises(ParamsError):
        params_mod.validate(MatchArmsParams, raw)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

_ARMS = "a=--type=greedy --name=A; b=--type=sim --top-k=4"


def _ctx(tmp_path, monkeypatch, max_cycles=0, **param_overrides) -> WorkerContext:
    from cloud.sinks import LocalSink

    monkeypatch.setattr(
        WorkerContext, "tag_paths", lambda self: SPEC.paths(self.tag, mount_root=tmp_path)
    )
    params = MatchArmsParams(
        **{"arms": _ARMS, "pairs_per_arm": 4, "round_pairs": 2, **param_overrides}
    )
    paths = SPEC.paths("t", mount_root=tmp_path)
    return WorkerContext(
        spec=SPEC,
        role=SPEC.role("arms"),
        tag="t",
        params=params,
        worker_id="w0",
        threads=2,
        max_cycles=max_cycles,
        sink=LocalSink(paths.root),
    )


def _scripted_round(calls, score_by_spec):
    """A play_round stand-in recording (player0_spec, seed) and returning each
    pair at a fixed score for that spec."""

    def fake_round(spec0, spec1, num_pairs, threads, seed, results_file, face_up_leaves):
        calls.append((spec0, seed))
        score = score_by_spec[spec0]
        wins = num_pairs * 2 if score == 1.0 else 0
        losses = num_pairs * 2 if score == 0.0 else 0
        draws = num_pairs * 2 - wins - losses
        return RoundResult([score] * num_pairs, wins=wins, draws=draws, losses=losses)

    return fake_round


def test_runner_measures_every_arm_with_shared_seeds(tmp_path, monkeypatch):
    calls = []
    scores = {"--type=greedy --name=A": 1.0, "--type=sim --top-k=4": 0.5}
    monkeypatch.setattr(arms_runner.harness, "play_round", _scripted_round(calls, scores))
    ctx = _ctx(tmp_path, monkeypatch)

    assert arms_runner.run(ctx) == 0

    # Both arms played their full budget, in declared order, from the SAME
    # seed schedule -- the cross-arm CRN discipline.
    assert [c[0] for c in calls[:2]] == ["--type=greedy --name=A"] * 2
    assert [c[1] for c in calls[:2]] == [1, 3]
    assert [c[1] for c in calls[2:]] == [1, 3]

    rows = db.read_all_match_arms(db.connect(ctx.tag_paths().dashboard_db))
    assert [r["arm"] for r in rows] == ["a", "b"]
    assert rows[0]["score"] == 1.0 and rows[0]["games"] == 8  # 4 pairs = 8 games
    assert rows[1]["score"] == 0.5
    assert rows[1]["pair_counts"] == [0, 0, 4, 0, 0]


def test_runner_skips_measured_arms_and_exits_when_done(tmp_path, monkeypatch):
    calls = []
    scores = {"--type=greedy --name=A": 1.0, "--type=sim --top-k=4": 0.5}
    monkeypatch.setattr(arms_runner.harness, "play_round", _scripted_round(calls, scores))

    # First run measures only arm 'a' (one cycle); the rerun -- the
    # reconciler's respawn -- resumes at 'b', then a third run is a no-op.
    ctx = _ctx(tmp_path, monkeypatch, max_cycles=1)
    assert arms_runner.run(ctx) == 0
    assert {c[0] for c in calls} == {"--type=greedy --name=A"}

    ctx = _ctx(tmp_path, monkeypatch)
    assert arms_runner.run(ctx) == 0
    rows = db.read_all_match_arms(db.connect(ctx.tag_paths().dashboard_db))
    assert [r["arm"] for r in rows] == ["a", "b"]

    before = len(calls)
    assert arms_runner.run(_ctx(tmp_path, monkeypatch)) == 0
    assert len(calls) == before  # nothing left: no matches played


# ---------------------------------------------------------------------------
# DB + figure plumbing
# ---------------------------------------------------------------------------


def _arm_record(**overrides) -> dict:
    record = {
        "player_spec": "--type=neural-sim --sim-top-k=5",
        "opponent": "--type=sim",
        "games": 80,
        "wins": 44,
        "draws": 2,
        "losses": 34,
        "pair_counts": [4, 8, 12, 10, 6],
        "score": 0.56,
        "ci_half_width": 0.05,
        "elapsed_s": 120.0,
    }
    record.update(overrides)
    return record


def test_match_arm_db_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "dashboard.db")
    db.write_match_arm(conn, "k5", _arm_record())
    db.write_match_arm(conn, "k10", _arm_record(score=0.61))
    # Replaying an arm replaces its row and keeps the original order.
    db.write_match_arm(conn, "k5", _arm_record(score=0.50))
    rows = db.read_all_match_arms(conn)
    assert [r["arm"] for r in rows] == ["k5", "k10"]
    assert rows[0]["score"] == 0.50
    assert rows[0]["pair_counts"] == [4, 8, 12, 10, 6]


def test_match_arms_figure_is_serializable(tmp_path):
    conn = db.connect(tmp_path / "dashboard.db")
    assert api.build_figure_item(conn, "match_arms", {}, str(tmp_path)) is None
    db.write_match_arm(conn, "k5", _arm_record())
    db.write_match_arm(conn, "k10", _arm_record(score=0.61))
    item = api.build_figure_item(conn, "match_arms", {}, str(tmp_path))
    assert item is not None
    assert {"doc", "root_id", "target_id"} <= set(item)

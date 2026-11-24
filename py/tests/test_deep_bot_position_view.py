"""deep_bot's Positions tab: PolicyBot's candidates and opponent-rack input
at a known position through the FFI, the analysis a release of tiny seeds
publishes (deep_bot/release_positions.py), and the tab's payload of it
(deep_bot/position_view.py, dashboard/deep_bot_api.py)."""

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

from scribblez import ffi  # noqa: E402
from scribblez.dashboard import db, deep_bot_api  # noqa: E402
from scribblez.deep_bot import (  # noqa: E402
    KINDS,
    known_positions,
    position_view,
    release,
    release_positions,
    trainer,
    units,
)
from scribblez.deep_bot.release_positions import KnownPositionsReplay, ReleaseModels  # noqa: E402
from scribblez.leave_inference.beliefs import NUM_PARAMS  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.sim_evidence.sobs import move_notation  # noqa: E402
from tests.test_deep_bot_trainer import CPU, PARAMS, save_tiny_seed  # noqa: E402

CLINKER = [p.id for p in known_positions.load()].index("CLINKER")  # its manifest index


@pytest.fixture(scope="module")
def replay() -> KnownPositionsReplay:
    return release_positions._replay()


@pytest.fixture(scope="module")
def tag(tmp_path_factory) -> TagPaths:
    """A tag holding release 0 of tiny seeds as the trainer publishes it: one
    ONNX export of each model and their analysis, a few seconds the module
    shares."""
    paths = TagPaths("t", "deep_bot", tmp_path_factory.mktemp("mount"))
    torch.manual_seed(0)
    for kind in KINDS:
        save_tiny_seed(paths, kind)
    with pytest.MonkeyPatch.context() as m:
        # Compiling the tiny models would take most of the setup's time.
        m.setattr(torch, "compile", lambda model: model)
        by_kind = units.build_units(paths, PARAMS, CPU, KINDS)
    trainer.ensure_newest_release(by_kind, paths, "t", False, trainer.DeepBotState())
    return paths


def _view(replay: KnownPositionsReplay, i: int) -> dict:
    return position_view.board_view(replay.positions[i])


def test_the_candidates_are_the_pool_ranked_by_static_equity(replay):
    view = _view(replay, CLINKER)
    moves, ranks = ffi.policy_candidates(replay.slog, CLINKER, view["turn"] - 1, 40)
    names = [move_notation(m, view["board"]) for m in moves]
    p = replay.positions[CLINKER]
    assert names[0] == p.static_equity_best and ranks[0] == 0
    assert names[ranks.tolist().index(p.reference_equity_rank - 1)] in p.reference
    # The pool ends with the pass, outside the ranking.
    assert names[-1] == "-" and ranks[-1] == -1
    assert len(set(names)) == len(names) and (np.diff(ranks[:-1]) > 0).all()


def test_the_opp_rack_input_is_the_engines_summary_of_the_belief(replay, tag):
    """The engine's summary reads the same rack the Python belief does: under
    the model's parameters and under its no-information prior."""
    models = ReleaseModels.load(tag, 0)
    b = release_positions.belief_at(models.beliefs, replay, CLINKER, _view(replay, CLINKER))
    assert b.modelled
    table = models.beliefs.table
    inferred = table.opp_rack_summary(b.unseen, b.params, b.leave_length, b.leave_pool)
    prior = table.opp_rack_summary(
        b.unseen, np.zeros(NUM_PARAMS, dtype=np.float32), b.leave_length, b.leave_pool
    )
    np.testing.assert_allclose(inferred["presence"], models.beliefs.rack_presence(b), atol=1e-5)
    np.testing.assert_allclose(
        prior["presence"], models.beliefs.rack_presence(b, uniform=True), atol=1e-5
    )
    assert inferred["modelled"] == 1 and inferred["leave_value_sd"] > 0
    uniform = models.opp_rack(dataclasses.replace(b, modelled=False))
    assert uniform["modelled"] == 0 and uniform["leave_value_mean"] == 0


def test_a_release_publishes_its_analysis_of_every_position(replay, tag):
    doc = json.loads(release_positions.path(tag, 0).read_text())
    assert doc["inferred"] is True  # the release's policy reads its own beliefs
    assert list(doc["positions"]) == [p.id for p in replay.positions]
    clinker = doc["positions"][replay.positions[CLINKER].id]
    probs = [c["prob"] for c in clinker["candidates"]]
    assert probs == sorted(probs, reverse=True) and sum(probs) == pytest.approx(1, abs=1e-4)
    moves = [c["move"] for c in clinker["candidates"]]
    assert replay.positions[CLINKER].static_equity_best in moves
    assert clinker["belief"]["modelled"] and len(clinker["belief"]["presence"]) == 27


def _payload(tag: TagPaths, number: int, index: int) -> dict:
    key = (tag.tag, tag.task, str(tag.mount_root))
    return json.loads(json.dumps(deep_bot_api.position_payload(key, number, index)))


def test_a_positions_payload_under_a_release(replay, tag):
    p = replay.positions[CLINKER]
    conn = db.connect(tag.dashboard_db)
    db.write_release_picks(
        conn, 0, {"hasty": [{"id": p.id, "pick": "F6 JILTE.", "reading": "equity", "seconds": 1.0}]}
    )
    conn.close()
    out = _payload(tag, 0, CLINKER)

    assert (out["id"], out["release"], out["rack"]) == (p.id, 0, "EIJLRST")
    assert out["picks"]["hasty"]["display"] == "F6 JILTE(R)"
    assert out["belief"]["model"] == "release 0"
    policy = out["policy"]
    listed = [c["move"] for c in policy["candidates"]]
    # The likeliest candidates, then the ones the answers name, in policy order.
    assert len(listed) < policy["num_candidates"]
    assert {*p.reference, p.static_equity_best} <= set(listed)
    ranks = [c["policy_rank"] for c in policy["candidates"]]
    assert ranks[: position_view.SHOWN_CANDIDATES] == list(
        range(1, position_view.SHOWN_CANDIDATES + 1)
    )
    assert ranks == sorted(ranks)
    assert all(0 <= c["win_pct"] <= 100 for c in policy["candidates"])
    rows = release.read_manifest(tag, 0)["rows_trained"]
    assert position_view.release_list(tag, {0}) == [
        {"release": 0, "rows_trained": rows, "picked": True}
    ]


def test_a_release_without_an_analysis_shows_no_model_outputs(tag):
    """A release published before the trainer wrote its analysis."""
    shutil.copytree(release.release_dir(tag, 0), release.release_dir(tag, 1))
    release_positions.path(tag, 1).unlink()
    out = _payload(tag, 1, CLINKER)
    assert out["release"] == 1 and out["belief"] is None and out["policy"] is None
    assert out["board"] == _payload(tag, 0, CLINKER)["board"]

"""deep_bot's Positions tab (scribblez.deep_bot.position_view,
dashboard/deep_bot_api.py): PolicyBot's candidates and opponent-rack input
at a known position through the FFI, and one position's payload under a
release of tiny seeds."""

import dataclasses
import json
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
from scribblez.deep_bot import KINDS, position_view, release, trainer, units  # noqa: E402
from scribblez.deep_bot.position_view import KnownPositionsReplay, ReleaseModels  # noqa: E402
from scribblez.leave_inference.beliefs import NUM_PARAMS  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.sim_evidence.sobs import move_notation  # noqa: E402
from tests.test_deep_bot_trainer import CPU, PARAMS, save_tiny_seed  # noqa: E402

CLINKER = 2  # the manifest's third position


@pytest.fixture(scope="module")
def replay() -> KnownPositionsReplay:
    return KnownPositionsReplay()


@pytest.fixture(scope="module")
def tag(tmp_path_factory) -> TagPaths:
    """A tag holding release 0 of tiny seeds, the trainer's own export: one
    ONNX export of each model, a few seconds the module shares."""
    paths = TagPaths("t", "deep_bot", tmp_path_factory.mktemp("mount"))
    torch.manual_seed(0)
    for kind in KINDS:
        save_tiny_seed(paths, kind)
    with pytest.MonkeyPatch.context() as m:
        # Compiling the tiny models would take most of the setup's time.
        m.setattr(torch, "compile", lambda model: model)
        by_kind = units.build_units(paths, PARAMS, CPU)
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
    b = position_view.belief_at(models, replay, CLINKER, _view(replay, CLINKER))
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


def test_a_positions_payload_under_a_release(replay, tag, monkeypatch):
    monkeypatch.setattr(deep_bot_api, "_replay", lambda: replay)
    p = replay.positions[CLINKER]
    conn = db.connect(tag.dashboard_db)
    db.write_release_picks(
        conn, 0, {"hasty": [{"id": p.id, "pick": "F6 JILTE.", "reading": "equity", "seconds": 1.0}]}
    )
    conn.close()
    key = (tag.tag, tag.task, str(tag.mount_root))
    out = json.loads(json.dumps(deep_bot_api.position_payload(key, 0, CLINKER)))

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

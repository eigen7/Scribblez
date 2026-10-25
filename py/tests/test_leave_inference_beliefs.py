"""The leave inference model's beliefs at .slog positions, on the CPU
(scribblez/leave_inference/beliefs.py): rows decoded from seeded HastyBot games
and their .forgone sidecars, a tiny exported model, and the engine's sampler."""

from math import comb
from pathlib import Path

import numpy as np
import pytest
import torch

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "forgone_play_tool").is_file():
    pytest.skip("engine not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

from scribblez.ffi import task_input_shapes  # noqa: E402
from scribblez.leave_inference import forgone  # noqa: E402
from scribblez.leave_inference.beliefs import (  # noqa: E402
    DISTRIBUTION,
    LeaveBeliefs,
    _target,
    unseen_pool,
)
from scribblez.leave_inference.heads import HEAD_TILTED  # noqa: E402
from scribblez.leave_inference.model import LeaveInferenceModel  # noqa: E402
from scribblez.leave_inference.onnx_export import export_onnx  # noqa: E402
from scribblez.selfplay import run_games  # noqa: E402

SHAPES = {s.name: s.dims for s in task_input_shapes("leave_inference")}
RACK = 7


@pytest.fixture(scope="module")
def model(tmp_path_factory) -> LeaveBeliefs:
    torch.manual_seed(0)
    net = LeaveInferenceModel(
        SHAPES["input_spatial"][0], SHAPES["input_scalar"][0], 8, 1, head=HEAD_TILTED
    ).eval()
    path = tmp_path_factory.mktemp("model") / "model.onnx"
    export_onnx(
        net,
        path,
        SHAPES["input_spatial"][0],
        SHAPES["input_scalar"][0],
        leave_values="test",
        public=False,
        provenance={},
    )
    return LeaveBeliefs(path)


@pytest.fixture(scope="module")
def slog(tmp_path_factory) -> Path:
    games = tmp_path_factory.mktemp("games")
    assert run_games(games, num_games=1, threads=1, player_spec="--type=hastybot", seed=3) == 0
    forgone.write_sidecars(games, threads=2)
    (path,) = games.glob("*.slog")
    return path


def beliefs_at(model: LeaveBeliefs, slog: Path, turns: list[int]):
    """Beliefs at game 0's `turns`, each unseen pool taken as its leave pool
    (equal unless the observer exchanged just before the opponent's move)."""
    rows = model.rows(slog, [(0, t) for t in turns])
    pools = [_target(r, "leave_pool").astype(np.int64) for r in rows]
    return model.beliefs(slog, [(0, t) for t in turns], pools, [RACK] * len(turns))


def modelled(model: LeaveBeliefs, slog: Path, count: int):
    """The first `count` beliefs among game 0's turns 1..12 that the model
    covers (a bingo or a pass before a turn leaves it uncovered)."""
    out = [b for b in beliefs_at(model, slog, list(range(1, 13))) if b.modelled]
    assert len(out) >= count
    return out[:count]


def test_turn_zero_has_no_belief(model, slog):
    (b,) = beliefs_at(model, slog, [0])
    assert not b.modelled and b.leave_length == 0 and not b.params.any()


def test_no_information_is_a_uniform_rack(model, slog):
    """Under zero parameters the leave is a uniform draw from the pool, so the
    rack is a uniform draw of 7: P(holds a tile) is hypergeometric."""
    for b in modelled(model, slog, 2):
        assert 1 <= b.leave_length <= 6
        n = int(b.unseen.sum())
        want = [1 - comb(n - int(u), RACK) / comb(n, RACK) for u in b.unseen]
        np.testing.assert_allclose(model.rack_presence(b, uniform=True), want, atol=1e-9)


def test_sampled_leaves_match_the_presence_marginals(model, slog):
    (b,) = modelled(model, slog, 1)
    probs = model.leave_probs(b.params, b)
    exact = probs @ model._leave_presence_matrix(b.leave_length)
    leaves = model._sample_leaves(b, 4000, np.random.default_rng(0))
    held = np.array([[t in leave for t in range(27)] for leave in leaves]).mean(axis=0)
    np.testing.assert_allclose(held, exact, atol=0.03)
    # The model's random weights do tilt it off the no-information prior.
    assert not np.allclose(model.rack_presence(b), model.rack_presence(b, uniform=True))


def test_sample_racks_are_distinct_and_drawn_from_the_unseen_pool(model, slog):
    b = modelled(model, slog, 2)[1]
    racks = model.sample_racks(b, 10, np.random.default_rng(1))
    assert len(racks) == 10
    assert len({(r["leave"], r["refill"]) for r in racks}) == 10
    for r in racks:
        assert len(r["leave"]) == b.leave_length and len(r["refill"]) == RACK - b.leave_length
        tiles = r["leave"] + r["refill"]
        counts = np.bincount([26 if c == "?" else ord(c) - 65 for c in tiles], minlength=27)
        assert (counts <= b.unseen).all()
    assert model.sample_racks(b, 10, np.random.default_rng(1)) == racks  # seeded


def test_unseen_pool_counts_the_board_and_the_rack():
    board = [[None] * 15 for _ in range(15)]
    board[7][7:10] = ["C", "a", "T"]  # 'a' is a blank
    u = unseen_pool(board, "EE?")
    assert u.sum() == 100 - 6
    assert u[2] == DISTRIBUTION[2] - 1 and u[19] == DISTRIBUTION[19] - 1
    assert u[4] == DISTRIBUTION[4] - 2 and u[26] == 0 and u[0] == DISTRIBUTION[0]

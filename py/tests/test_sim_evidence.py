"""Tests for the sim-evidence data path: sim_obs_tool -> .sobs parsing ->
row decoding by identity -> evidence features -> the evidence-conditioned
model. The end-to-end pieces run the real C++ tool on freshly generated
.slog fixtures and are skipped when the binary or the HastyBot leave values
(mount macondo strategy data) are unavailable.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from scribblez.dataset import row_layout
from scribblez.ffi import decode_rows, get_input_shapes, row_size_floats
from scribblez.sim_evidence.model import EvidencePostMoveModel
from scribblez.sim_evidence.sobs import (
    MOVE_PLAY,
    NUM_EVIDENCE_SCALARS,
    evidence_features,
    move_footprint,
    read_sobs,
)

SIM_OBS_TOOL = Path("/workspace/repo/target/engine/sim_obs_tool")
SLOG_WRITER = Path("/workspace/repo/target/engine/test_slog_writer")
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

ROLLOUTS = 8
TOP_K = 3


@pytest.fixture(scope="module")
def sobs_dir(tmp_path_factory) -> Path:
    """A directory of .slog files with freshly generated .sobs sidecars."""
    if not SIM_OBS_TOOL.exists() or not SLOG_WRITER.exists():
        pytest.skip("engine binaries not built")
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    d = tmp_path_factory.mktemp("sim_evidence")
    subprocess.run([str(SLOG_WRITER), str(d), "12", "6"], check=True, capture_output=True)
    result = subprocess.run(
        [
            str(SIM_OBS_TOOL),
            f"--slog-dir={d}",
            f"--rollouts={ROLLOUTS}",
            f"--top-k={TOP_K}",
            "--positions-per-game=2",
            "--threads=4",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"sim_obs_tool failed: {result.stderr}"
    assert sorted(d.glob("*.sobs")), "no .sobs produced"
    return d


def test_sobs_positions_parse_and_hold_invariants(sobs_dir):
    total_positions = 0
    for sobs in sorted(sobs_dir.glob("*.sobs")):
        positions = read_sobs(sobs)
        total_positions += len(positions)
        for pos in positions:
            assert pos.rollouts == ROLLOUTS
            assert 1 <= len(pos.moves) <= TOP_K
            for move, obs in zip(pos.moves, pos.obs, strict=True):
                assert int(move["type"]) in (0, 1, 2)
                assert int(obs["n"]) == ROLLOUTS
                assert int(obs["wins"]) + int(obs["draws"]) + int(obs["losses"]) == ROLLOUTS
                assert (obs["opp_win_count"] <= obs["opp_next_count"]).all()
                assert (obs["self_win_count"] <= obs["self_next_count"]).all()
                assert (obs["opp_next_count"] <= ROLLOUTS).all()
    assert total_positions > 0


def test_sobs_positions_decode_as_training_rows(sobs_dir):
    slog = sorted(sobs_dir.glob("*.slog"))[0]
    positions = read_sobs(slog.with_suffix(".sobs"))
    games = np.array([p.game_index for p in positions], dtype=np.int64)
    turns = np.array([p.turn_index for p in positions], dtype=np.int64)
    rows = decode_rows(slog, games, turns, post_move=True)
    assert rows.shape == (len(positions), row_size_floats())
    # The WLD one-hot (located via the row layout): exactly one of 3 set per row.
    _, targets = row_layout()
    start, end = next((s, e) for name, s, e, _ in targets if name == "wld")
    wld = rows[:, start:end]
    assert np.all(wld.sum(axis=1) == 1.0)


def test_move_footprint_and_features(sobs_dir):
    slog = sorted(sobs_dir.glob("*.slog"))[0]
    positions = read_sobs(slog.with_suffix(".sobs"))
    pos = positions[0]
    planes, scalars, mask = evidence_features(pos, max_k=TOP_K + 2)
    assert planes.shape == (TOP_K + 2, 5, 15, 15)
    assert scalars.shape == (TOP_K + 2, NUM_EVIDENCE_SCALARS)
    assert mask.sum() == len(pos.moves)
    for i, move in enumerate(pos.moves):
        foot = move_footprint(move)
        if int(move["type"]) == MOVE_PLAY:
            assert foot.sum() == int(move["num_played"])
        else:
            assert foot.sum() == 0
        assert np.array_equal(planes[i, 4].astype(np.float32), foot)
        # Frequencies are per-rollout fractions in [0, 1].
        assert float(planes[i, :4].max()) <= 1.0
    # W/D/L frequencies sum to 1 for real candidates.
    assert np.allclose(scalars[mask][:, :3].sum(axis=1), 1.0)


def _tiny_model() -> EvidencePostMoveModel:
    shapes = {s.name: s.dims for s in get_input_shapes()}
    torch.manual_seed(0)
    return EvidencePostMoveModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=16,
        num_blocks=2,
    )


def test_model_ignores_masked_evidence():
    model = _tiny_model().eval()
    shapes = {s.name: s.dims for s in get_input_shapes()}
    b, k = 2, 4
    spatial = torch.randn(b, *shapes["input_spatial"])
    scalar = torch.randn(b, shapes["input_scalar"][0])
    mask = torch.zeros(b, k, dtype=torch.bool)
    with torch.no_grad():
        out_zero = model(
            spatial,
            scalar,
            torch.zeros(b, k, 5, 15, 15),
            torch.zeros(b, k, NUM_EVIDENCE_SCALARS),
            mask,
        )
        out_junk = model(
            spatial,
            scalar,
            torch.randn(b, k, 5, 15, 15),
            torch.randn(b, k, NUM_EVIDENCE_SCALARS),
            mask,
        )
    for key in out_zero:
        assert torch.equal(out_zero[key], out_junk[key]), key


def test_model_trains_on_evidence():
    model = _tiny_model()
    shapes = {s.name: s.dims for s in get_input_shapes()}
    b, k = 3, 4
    spatial = torch.randn(b, *shapes["input_spatial"])
    scalar = torch.randn(b, shapes["input_scalar"][0])
    planes = torch.rand(b, k, 5, 15, 15)
    scalars = torch.randn(b, k, NUM_EVIDENCE_SCALARS)
    mask = torch.ones(b, k, dtype=torch.bool)
    mask[0, 2:] = False  # a padded set in the batch
    out = model(spatial, scalar, planes, scalars, mask)
    out["wld"].sum().backward()
    # The fusion projections are zero-initialized, so at init they (not the
    # encoder behind them) are the learning conduits: their weight gradients
    # must be nonzero, which is what pulls them off zero and opens gradient
    # flow into the rest of the evidence encoder.
    for param in (model.evidence.spatial_out.weight, model.value_proj.weight):
        assert param.grad is not None and param.grad.abs().sum() > 0


def test_empty_evidence_is_finite():
    # An all-padding evidence set exercises the empty-attention guard.
    model = _tiny_model().eval()
    shapes = {s.name: s.dims for s in get_input_shapes()}
    spatial = torch.randn(1, *shapes["input_spatial"])
    scalar = torch.randn(1, shapes["input_scalar"][0])
    mask = torch.zeros(1, 3, dtype=torch.bool)
    with torch.no_grad():
        out = model(
            spatial,
            scalar,
            torch.zeros(1, 3, 5, 15, 15),
            torch.zeros(1, 3, NUM_EVIDENCE_SCALARS),
            mask,
        )
    for value in out.values():
        assert torch.isfinite(value).all()

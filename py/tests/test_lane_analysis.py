"""Tests for the max-move-per-lane lane-analysis Python layer: the prediction
store (dashboard DB) and decoding a model's outputs over the dataset."""

import numpy as np
import pytest
import torch
from scribblez import lane_analysis
from scribblez.dashboard import db


def _fake_preds(n: int) -> dict:
    rng = np.random.default_rng(0)
    return {
        "occ": (rng.random((n, 30, 15, 27)) > 0.7).astype(np.uint8),
        "score_pmf": rng.random((n, 30, 100)).astype(np.float32),
        "has_move": rng.random((n, 30)).astype(np.float32),
    }


def test_db_lane_pred_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "d.db")
    preds = _fake_preds(3)
    db.write_lane_preds(conn, generation=2, positions=4096, preds=preds)

    assert db.read_lane_generations(conn) == [{"generation": 2, "positions": 4096}]
    p1 = db.read_lane_pred(conn, 2, 1)
    assert p1["occ"].shape == (30, 15, 27)
    assert p1["score_pmf"].shape == (30, 100)
    assert p1["has_move"].shape == (30,)
    np.testing.assert_array_equal(p1["occ"], preds["occ"][1])
    np.testing.assert_allclose(p1["score_pmf"], preds["score_pmf"][1])
    assert db.read_lane_pred(conn, 2, 99) is None


def test_db_lane_pred_replace_on_resume(tmp_path):
    conn = db.connect(tmp_path / "d.db")
    db.write_lane_preds(conn, 0, 100, _fake_preds(2))
    db.write_lane_preds(conn, 0, 200, _fake_preds(2))  # same generation -> replaced
    assert db.read_lane_generations(conn) == [{"generation": 0, "positions": 200}]


def test_predict_decode_shapes():
    from scribblez.max_move_per_lane.model import MaxMovePerLaneModel

    spatial_planes, scalar = 31, 27
    model = MaxMovePerLaneModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar,
        trunk_channels=8,
        num_blocks=1,
        lane_layers=1,
        lane_heads=2,
        ffn_mult=1,
        n_rack_tokens=2,
    )
    n = 4
    inputs = np.random.default_rng(1).random((n, spatial_planes * 225 + scalar)).astype(np.float32)
    preds = lane_analysis.predict(model, inputs, spatial_planes, torch.device("cpu"))

    assert preds["occ"].shape == (n, 30, 15, 27)
    assert preds["occ"].dtype == np.uint8
    assert preds["score_pmf"].shape == (n, 30, 100)
    assert preds["has_move"].shape == (n, 30)
    np.testing.assert_allclose(preds["score_pmf"].sum(-1), 1.0, atol=1e-4)  # a PMF per lane
    assert ((preds["has_move"] >= 0) & (preds["has_move"] <= 1)).all()  # a probability


def test_load_inputs_real_dataset():
    """The shipped GCG dataset loads in natural order with full-width inputs."""
    try:
        names, inputs = lane_analysis.load_inputs(lane_analysis.DEFAULT_DATASET)
    except OSError:
        pytest.skip("lexicon unavailable")
    # natural order, not pos-1,pos-10
    assert names == [f"pos-{i}" for i in range(1, len(names) + 1)]
    assert inputs.shape == (len(names), 7002)

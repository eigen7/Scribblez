"""Tests for the move set evaluation model training side: the move-feature encoder
(pure CPU) and the end-to-end dataset -> train step -> recall eval path over a
tiny generated .mset/.slog corpus (GPU, since the target generator builds a
TensorRT engine).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from scribblez.move_set_eval import moves as move_enc
from scribblez.sim_evidence.sobs import MOVE_DTYPE, MOVE_PLAY, move_footprint

TARGET_GENERATOR = Path("/workspace/repo/target/engine/move_set_eval_target_generator")
SLOG_WRITER = Path("/workspace/repo/target/engine/test_slog_writer")
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

QUOTAS = {"top": 3, "mid": 2, "tail": 2, "exchange": 1}


def test_encode_moves_matches_footprint():
    """The move encoder's placed squares agree with the .sobs footprint walk,
    and the scalar block reflects the resultant differential / tile-count /
    is-play."""
    move = np.zeros(1, dtype=MOVE_DTYPE)[0]
    move["type"] = MOVE_PLAY
    move["horizontal"] = 1
    move["start"] = 7
    move["square_mask"] = (1 << 7) | (1 << 9) | (1 << 10)
    move["num_played"] = 3
    move["glyphs"][:3] = [18, 1, 4]  # R, A, D (natural letters)
    move["score"] = 20

    # Pre-move differential of 30; resultant = 30 + 20 = 50 -> 50/100 = 0.5.
    enc = move_enc.encode_moves(np.array([move], dtype=MOVE_DTYPE), np.array([30], dtype=np.int32))
    mask = enc["tile_mask"][0]
    assert mask.sum() == 3
    squares = enc["squares"][0][mask]
    got = {(int(s) // move_enc.BOARD, int(s) % move_enc.BOARD) for s in squares}
    footprint = move_footprint(move)
    expected = set(zip(*np.nonzero(footprint), strict=True))
    assert got == expected
    # Natural letters: id == A..Z index + 1, so R/A/D -> 18/1/4; no blanks.
    assert list(enc["letters"][0][mask]) == [18, 1, 4]
    assert not enc["blanks"][0][mask].any()
    np.testing.assert_allclose(enc["scalars"][0], [0.50, 3 / 7, 1.0], atol=1e-6)


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory) -> Path:
    """A tiny .slog corpus labeled with .mset targets from a small teacher."""
    if not TARGET_GENERATOR.exists() or not SLOG_WRITER.exists():
        pytest.skip("engine binaries not built")
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from scribblez.ffi import get_input_shapes
    from scribblez.position_eval.model import PositionEvalModel
    from scribblez.position_eval.onnx_export import export_onnx

    d = tmp_path_factory.mktemp("move_set_eval_train")
    subprocess.run([str(SLOG_WRITER), str(d), "12", "4"], check=True, capture_output=True)

    shapes = {s.name: s.dims for s in get_input_shapes()}
    torch.manual_seed(0)
    teacher = PositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=8,
        num_blocks=3,
    ).eval()
    onnx_path = d / "teacher.onnx"
    export_onnx(
        teacher,
        onnx_path,
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        contingent_features=True,
    )
    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-dir={d}",
            f"--model={onnx_path}",
            "--fast-build",
            f"--quota-top={QUOTAS['top']}",
            f"--quota-mid={QUOTAS['mid']}",
            f"--quota-tail={QUOTAS['tail']}",
            f"--quota-exchange={QUOTAS['exchange']}",
            "--positions-per-game=2",
            "--threads=4",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"target generator failed: {result.stderr}"
    assert sorted(d.glob("*.mset")), "no .mset produced"
    return d


def test_dataset_batches_flatten_candidates(corpus_dir):
    from scribblez.move_set_eval.dataset import MsetDataset

    ds = MsetDataset(corpus_dir)
    assert ds.num_positions > 0
    assert ds.num_candidates >= ds.num_positions  # >= 1 candidate each

    seen_positions = 0
    for batch in ds.iter_batches(positions_per_batch=8, seed=0):
        p = batch["input_spatial"].shape[0]
        m = batch["move_pos_id"].shape[0]
        assert batch["input_spatial"].shape[1] == ds.spatial_planes
        assert batch["input_scalar"].shape == (p, ds.scalar_size)
        # Flattened move tensors are all length M and map into [0, P).
        move_keys = ("move_letters", "move_blanks", "move_squares", "move_tile_mask")
        for key in (*move_keys, "move_scalars"):
            assert batch[key].shape[0] == m
        assert int(batch["move_pos_id"].max()) < p
        assert batch["target_wld"].shape == (m, 3)
        assert batch["target_score_diff"].shape == (m, 2)
        seen_positions += p
    assert seen_positions == ds.num_positions


def test_train_step_and_eval(corpus_dir):
    from scribblez.move_set_eval.dataset import MsetDataset
    from scribblez.move_set_eval.eval import evaluate
    from scribblez.move_set_eval.model import MoveSetEvalModel
    from scribblez.move_set_eval.train_loop import LossConfig, run_epoch

    device = torch.device("cpu")
    ds = MsetDataset(corpus_dir)
    torch.manual_seed(0)
    model = MoveSetEvalModel(
        spatial_planes=ds.spatial_planes,
        scalar_size=ds.scalar_size,
        trunk_channels=8,
        num_blocks=2,
        num_heads=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_cfg = LossConfig(lambda_sd=0.004, huber_delta_mean=10.0, huber_delta_std=10.0)

    rows = 0
    first = last = None
    for epoch in range(3):
        result = run_epoch(
            model,
            optimizer,
            ds.iter_batches(positions_per_batch=8, seed=0, epoch_index=epoch),
            device,
            loss_cfg,
            rows_trained=rows,
        )
        rows = result.rows_trained
        assert np.isfinite(result.losses["total"])
        first = result.losses["total"] if first is None else first
        last = result.losses["total"]
    assert rows == 3 * ds.num_candidates
    assert last <= first * 2.0  # a few steps should not diverge

    metrics = evaluate(model, ds, device, positions_per_batch=8)
    assert metrics["positions"] == ds.num_positions
    for k in (1, 3, 5):
        assert 0.0 <= metrics[f"recall@{k}"] <= 1.0
    assert -1.0 <= metrics["spearman"] <= 1.0

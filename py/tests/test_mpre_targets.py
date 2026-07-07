"""Tests for the M_pre distillation-target data path: mpre_target_tool over
.slog fixtures with a tiny teacher ONNX, parsed by the .mpt reader. Skipped
when the engine binaries, HastyBot leave values, or torch/TensorRT are
unavailable (the tool builds a TensorRT engine, so this is a GPU test).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from scribblez.mpre.targets import TARGET_NAMES_V1, read_mpt
from scribblez.post_move_value.model import PostMoveValueModel
from scribblez.post_move_value.onnx_export import export_onnx
from scribblez.sim_evidence.slog_meta import game_metas, move_at, read_slog_bytes

MPRE_TOOL = Path("/workspace/repo/target/engine/mpre_target_tool")
SLOG_WRITER = Path("/workspace/repo/target/engine/test_slog_writer")
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

QUOTAS = {"top": 3, "mid": 2, "tail": 2, "exchange": 1}


@pytest.fixture(scope="module")
def mpt_dir(tmp_path_factory) -> Path:
    """A directory of .slog files labeled by the tool with a tiny teacher."""
    if not MPRE_TOOL.exists() or not SLOG_WRITER.exists():
        pytest.skip("engine binaries not built")
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    d = tmp_path_factory.mktemp("mpre_targets")
    subprocess.run([str(SLOG_WRITER), str(d), "8", "4"], check=True, capture_output=True)

    from scribblez.ffi import get_input_shapes

    shapes = {s.name: s.dims for s in get_input_shapes()}
    torch.manual_seed(0)
    model = PostMoveValueModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=8,
        num_blocks=3,
    ).eval()
    onnx_path = d / "teacher.onnx"
    export_onnx(
        model,
        onnx_path,
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        contingent_features=True,
    )

    result = subprocess.run(
        [
            str(MPRE_TOOL),
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
    assert result.returncode == 0, f"mpre_target_tool failed: {result.stderr}"
    assert sorted(d.glob("*.mpt")), "no .mpt produced"
    return d


def test_mpt_positions_parse_and_hold_invariants(mpt_dir):
    max_candidates = 1 + sum(QUOTAS.values())  # the played move + the strata
    total = 0
    for path in sorted(mpt_dir.glob("*.mpt")):
        parsed = read_mpt(path)
        assert parsed.record_floats == len(TARGET_NAMES_V1)
        assert len(parsed.model_hash) > 0
        assert parsed.flags == 0
        for pos in parsed.positions:
            total += 1
            assert 1 <= len(pos.moves) <= max_candidates
            assert pos.targets.shape == (len(pos.moves), len(TARGET_NAMES_V1))
            # WLD probabilities: each a distribution summing to ~1.
            wld = pos.targets[:, :3]
            assert np.all(wld >= 0) and np.all(wld <= 1)
            assert np.allclose(wld.sum(axis=1), 1.0, atol=1e-3)
            # The std head is softplus-floored positive.
            assert np.all(pos.targets[:, 4] > 0)
    assert total > 0


def test_mpt_includes_the_played_move(mpt_dir):
    slog = sorted(mpt_dir.glob("*.slog"))[0]
    buf = read_slog_bytes(slog)
    metas = game_metas(buf)
    parsed = read_mpt(slog.with_suffix(".mpt"))
    for pos in parsed.positions:
        played = move_at(buf, metas[pos.game_index], pos.turn_index).tobytes()
        assert any(pos.moves[i].tobytes() == played for i in range(len(pos.moves)))


def test_mpt_model_hash_consistent_across_files(mpt_dir):
    hashes = {read_mpt(p).model_hash for p in mpt_dir.glob("*.mpt")}
    assert len(hashes) == 1

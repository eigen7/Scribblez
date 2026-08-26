"""Tests for the FP16-safety export gate and the activation-magnitude recipe
terms (docs/fp16_safe_serving.md): the probe catches over-range intermediates
that never reach a graph output, an overflowing checkpoint fails the export
atomically (no ONNX lands), and the restoring-force loss terms (pooled-FC
penalty, WLD z-loss) are live in both families' objectives."""

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from scribblez.ffi import get_input_shapes
from scribblez.fp16_gate import Fp16HeadroomError, check_fp16_headroom, intermediate_peaks
from scribblez.move_set_eval.model import compute_loss as mset_compute_loss
from scribblez.position_eval.model import MASK_HEAD_NAMES, PositionEvalModel
from scribblez.position_eval.model import compute_loss as position_compute_loss
from scribblez.position_eval.onnx_export import export_onnx
from scribblez.spatial_trunk import PoolFcPenalty, SpatialTrunk

_input_shapes = {s.name: s.dims for s in get_input_shapes()}
SPATIAL_PLANES, BOARD_SIZE, _ = _input_shapes["input_spatial"]
SCALAR_SIZE = _input_shapes["input_scalar"][0]


# ---------------------------------------------------------------------------
# The gate machinery, on a hand-built graph
# ---------------------------------------------------------------------------


def _hidden_peak_model(tmp_path, scale: float):
    """x (N,4) -> Gemm(scale*I) -> Relu -> Gemm(I/scale) -> y: the first Gemm's
    output peaks at ~scale but the graph output stays ~1, so only a probe that
    exposes intermediates can see it."""
    up = numpy_helper.from_array(np.eye(4, dtype=np.float32) * scale, "w_up")
    down = numpy_helper.from_array(np.eye(4, dtype=np.float32) / scale, "w_down")
    nodes = [
        helper.make_node("MatMul", ["x", "w_up"], ["hidden"]),
        helper.make_node("Relu", ["hidden"], ["hidden_relu"]),
        helper.make_node("MatMul", ["hidden_relu", "w_down"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "hidden_peak",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 4])],
        initializer=[up, down],
    )
    path = tmp_path / f"hidden_{scale}.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    return path


_ONES_FEED = [{"x": np.ones((2, 4), dtype=np.float32)}]


def test_probe_sees_hidden_intermediates(tmp_path):
    path = _hidden_peak_model(tmp_path, scale=30000.0)
    peaks = dict(intermediate_peaks(path, _ONES_FEED))
    assert peaks["hidden"] == pytest.approx(30000.0)
    assert peaks["y"] == pytest.approx(1.0)


def test_gate_fails_over_threshold(tmp_path):
    path = _hidden_peak_model(tmp_path, scale=30000.0)
    with pytest.raises(Fp16HeadroomError) as err:
        check_fp16_headroom(path, _ONES_FEED)
    assert any(name == "hidden" for name, _ in err.value.peaks)


def test_gate_passes_under_threshold(tmp_path):
    path = _hidden_peak_model(tmp_path, scale=1000.0)
    assert check_fp16_headroom(path, _ONES_FEED) == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# The gated export
# ---------------------------------------------------------------------------


def _random_model(seed: int = 0) -> PositionEvalModel:
    torch.manual_seed(seed)
    model = PositionEvalModel(
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        trunk_channels=16,
        num_blocks=3,  # includes one global-pooling block
        board_size=BOARD_SIZE,
    )
    model.eval()
    return model


def _random_feeds(batch: int = 4, seed: int = 1) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {
            "input_spatial": rng.standard_normal(
                (batch, SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
            ),
            "input_scalar": rng.standard_normal((batch, SCALAR_SIZE), dtype=np.float32),
        }
    ]


def _export(model, path, probe_feeds):
    return export_onnx(
        model,
        path,
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        contingent_features=True,
        opp_leave_input=False,
        board_size=BOARD_SIZE,
        probe_feeds=probe_feeds,
    )


def test_export_gate_passes_and_reports_peak(tmp_path):
    path = tmp_path / "model.onnx"
    peak = _export(_random_model(), path, _random_feeds())
    assert path.exists()
    assert peak is not None and 0.0 < peak < 16384.0


def test_export_gate_blocks_overflowing_checkpoint(tmp_path):
    model = _random_model()
    # The stem conv feeds a fresh BatchNorm whose running stats are (0, 1), so
    # inflating its weights inflates the whole trunk: the checkpoint overflows
    # exactly the way a magnitude-grown trained one does.
    with torch.no_grad():
        model.trunk.stem[0].weight.mul_(1e5)
    path = tmp_path / "model.onnx"
    with pytest.raises(Fp16HeadroomError):
        _export(model, path, _random_feeds())
    assert not path.exists(), "a failed gate must not land an ONNX file"


# ---------------------------------------------------------------------------
# The recipe terms
# ---------------------------------------------------------------------------


def test_pool_fc_penalty_records_and_closes():
    torch.manual_seed(0)
    trunk = SpatialTrunk(spatial_planes=4, scalar_size=3, trunk_channels=8, num_blocks=3)
    recorder = PoolFcPenalty(trunk)
    spatial, scalar = torch.randn(2, 4, 15, 15), torch.randn(2, 3)

    trunk(spatial, scalar)
    penalty = recorder.penalty()
    assert penalty is not None and penalty.item() > 0.0
    assert penalty.requires_grad
    assert recorder.penalty() is None, "penalty() pops its collection"

    recorder.close()
    trunk(spatial, scalar)
    assert recorder.penalty() is None, "a closed recorder must not collect"


def _position_loss_args(wld_scale: float):
    torch.manual_seed(0)
    outputs = {
        "wld": torch.randn(2, 3) * wld_scale,
        "score_diff": torch.stack([torch.randn(2), torch.rand(2) + 0.5], dim=1),
        **{name: torch.randn(2, 15, 15) for name in MASK_HEAD_NAMES},
    }
    targets = {
        "wld": torch.eye(3)[[0, 1]],
        "score_diff": torch.randn(2, 1),
        **{name: torch.rand(2, 15, 15).round() for name in MASK_HEAD_NAMES},
    }
    return outputs, targets


def test_wld_z_loss_restrains_logit_growth():
    outputs, targets = _position_loss_args(wld_scale=1.0)
    small = position_compute_loss(outputs, targets, lambda_wld_z=1e-4)
    outputs_big = dict(outputs, wld=outputs["wld"] + 100.0)
    big = position_compute_loss(outputs_big, targets, lambda_wld_z=1e-4)
    # A uniform logit shift leaves the cross-entropy untouched; only the
    # z-loss opposes it.
    assert big["wld"].item() == pytest.approx(small["wld"].item(), abs=1e-4)
    assert big["wld_z"].item() > small["wld_z"].item()
    assert big["total"].item() > small["total"].item()
    expected = torch.logsumexp(outputs["wld"], dim=1).square().mean()
    assert small["wld_z"].item() == pytest.approx(expected.item())


def test_mset_wld_z_loss_in_total():
    torch.manual_seed(0)
    outputs = {
        "wld": torch.randn(4, 3) * 50.0,
        "score_diff": torch.stack([torch.randn(4), torch.rand(4) + 0.5], dim=1),
        "planes": torch.randn(4, 4, 225),
    }
    targets = {
        "target_wld": torch.softmax(torch.randn(4, 3), dim=1),
        "target_score_diff": torch.randn(4, 2),
    }
    zero = mset_compute_loss(outputs, targets, lambda_wld_z=0.0)
    on = mset_compute_loss(outputs, targets, lambda_wld_z=1e-4)
    assert on["wld_z"].item() > 0.0
    assert on["total"].item() == pytest.approx(
        zero["total"].item() + 1e-4 * on["wld_z"].item(), rel=1e-5
    )

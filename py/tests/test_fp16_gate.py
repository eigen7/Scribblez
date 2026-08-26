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
from scribblez.ffi import get_input_shapes, score_diff_input_layout
from scribblez.fp16_gate import (
    PROBE_LEADS,
    Fp16HeadroomError,
    check_fp16_headroom,
    intermediate_peaks,
)
from scribblez.move_set_eval.model import compute_loss as mset_compute_loss
from scribblez.move_set_eval.onnx_export import fp16_probe_feeds_from_batch
from scribblez.position_eval.model import MASK_HEAD_NAMES, PositionEvalModel
from scribblez.position_eval.model import compute_loss as position_compute_loss
from scribblez.position_eval.onnx_export import export_onnx, fp16_probe_feeds
from scribblez.position_eval.trainer import FP16_PROBE_LARGE_ROWS, load_fp16_probe
from scribblez.spatial_trunk import PoolFcPenalty, SpatialTrunk, apply_pool_penalty

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
    tmp = path.with_name(path.name + ".tmp")
    assert not tmp.exists(), "a failed gate must clean up its temp file"


# ---------------------------------------------------------------------------
# The probe-feed builders
# ---------------------------------------------------------------------------


def test_position_probe_feeds_stamping_and_pairing():
    rng = np.random.default_rng(2)
    n, cells = 5, BOARD_SIZE * BOARD_SIZE
    flat = rng.standard_normal((n, SPATIAL_PLANES * cells + SCALAR_SIZE), dtype=np.float32)
    leads = (-200, 150)
    feeds = fp16_probe_feeds(flat, SPATIAL_PLANES, leads=leads, rows_per_feed=4)

    spatial = np.concatenate([f["input_spatial"] for f in feeds])
    scalar = np.concatenate([f["input_scalar"] for f in feeds])
    assert spatial.shape[0] == scalar.shape[0] == n * (1 + len(leads))

    orig_spatial = flat[:, : SPATIAL_PLANES * cells].reshape(n, SPATIAL_PLANES, 15, 15)
    orig_scalar = flat[:, SPATIAL_PLANES * cells :]
    sd_index, sd_scale = score_diff_input_layout()
    others = np.arange(SCALAR_SIZE) != sd_index

    # Block 0 is the rows as encoded; block b >= 1 is the rows with the
    # score-diff scalar stamped to leads[b-1], everything else untouched, and
    # the spatial half still paired with its own row.
    np.testing.assert_array_equal(scalar[:n], orig_scalar)
    for b, lead in enumerate(leads, start=1):
        block = scalar[b * n : (b + 1) * n]
        np.testing.assert_allclose(block[:, sd_index], np.float32(lead / sd_scale))
        np.testing.assert_array_equal(block[:, others], orig_scalar[:, others])
        np.testing.assert_array_equal(spatial[b * n : (b + 1) * n], orig_spatial)


def test_mset_probe_feeds_from_batch():
    rng = np.random.default_rng(3)
    t = 7
    batch = {
        "input_spatial": torch.from_numpy(
            rng.standard_normal((2, SPATIAL_PLANES, 15, 15), dtype=np.float32)
        ),
        "input_scalar": torch.from_numpy(rng.standard_normal((2, SCALAR_SIZE), dtype=np.float32)),
        "move_letters": torch.randint(0, 26, (3, t)),
        "move_blanks": torch.zeros(3, t, dtype=torch.bool),
        "move_squares": torch.randint(0, 225, (3, t)),
        "move_tile_mask": torch.ones(3, t, dtype=torch.bool),
        "move_scalars": torch.randn(3, 3),
        "move_pos_id": torch.tensor([0, 0, 1]),
    }
    feeds = fp16_probe_feeds_from_batch(batch, leads=(100,))
    assert len(feeds) == 2 * 2  # 2 positions x (as-encoded + 1 lead)

    sd_index, sd_scale = score_diff_input_layout()
    others = np.arange(SCALAR_SIZE) != sd_index
    for p, (plain, stamped) in enumerate([feeds[0:2], feeds[2:4]]):
        np.testing.assert_array_equal(plain["input_scalar"][0], batch["input_scalar"][p].numpy())
        assert stamped["input_scalar"][0, sd_index] == pytest.approx(100 / sd_scale)
        # The stamp touches exactly the score-diff scalar: everything else in
        # the stamped feed is the row as encoded.
        np.testing.assert_array_equal(
            stamped["input_scalar"][0, others], plain["input_scalar"][0, others]
        )
        np.testing.assert_array_equal(stamped["input_spatial"], plain["input_spatial"])
        np.testing.assert_array_equal(plain["input_spatial"][0], batch["input_spatial"][p].numpy())
        # Each feed carries only its own position's move rows, in the export
        # graph's dtypes.
        expected_moves = int((batch["move_pos_id"] == p).sum())
        assert plain["move_letters"].shape[0] == expected_moves
        assert plain["move_letters"].dtype == np.int32
        assert plain["move_blanks"].dtype == np.uint8
        assert plain["move_scalars"].dtype == np.float32


def test_load_fp16_probe_sources():
    rng = np.random.default_rng(4)
    cells = BOARD_SIZE * BOARD_SIZE
    variants = 1 + len(PROBE_LEADS)

    def rows(n):
        return rng.standard_normal((n, SPATIAL_PLANES * cells + SCALAR_SIZE), dtype=np.float32)

    def probe_rows(feeds):
        return sum(f["input_spatial"].shape[0] for f in feeds)

    frozen, quality = {"inputs": rows(3)}, {"inputs": rows(FP16_PROBE_LARGE_ROWS + 16)}
    sliced = FP16_PROBE_LARGE_ROWS
    assert probe_rows(load_fp16_probe(frozen, None, SPATIAL_PLANES)) == 3 * variants
    # The quality set contributes only its FP16_PROBE_LARGE_ROWS slice.
    assert probe_rows(load_fp16_probe(None, quality, SPATIAL_PLANES)) == sliced * variants
    assert probe_rows(load_fp16_probe(frozen, quality, SPATIAL_PLANES)) == (3 + sliced) * variants
    assert load_fp16_probe(None, None, SPATIAL_PLANES) is None


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


def test_apply_pool_penalty_weights_total():
    torch.manual_seed(0)
    trunk = SpatialTrunk(spatial_planes=4, scalar_size=3, trunk_channels=8, num_blocks=3)
    recorder = PoolFcPenalty(trunk)
    trunk(torch.randn(2, 4, 15, 15), torch.randn(2, 3))
    losses = {"total": torch.tensor(1.5)}
    apply_pool_penalty(losses, recorder, 0.25)
    assert losses["pool_act"].item() > 0.0
    assert losses["total"].item() == pytest.approx(1.5 + 0.25 * losses["pool_act"].item())
    recorder.close()

    # A model without pooled blocks contributes a constant zero.
    losses = {"total": torch.tensor(1.5)}
    recorder = PoolFcPenalty(torch.nn.Linear(3, 3))
    apply_pool_penalty(losses, recorder, 0.25)
    assert losses["pool_act"].item() == 0.0 and losses["total"].item() == pytest.approx(1.5)
    recorder.close()


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

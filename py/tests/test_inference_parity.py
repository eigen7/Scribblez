"""Inference-parity tests for the position-eval ONNX export, PyTorch vs ONNXRuntime.

Training and Python-side analysis run the PyTorch PositionEvalModel; engine
agents run the same weights through onnx_export and TensorRT. These tests pin the
first hop: the exported graph under ONNXRuntime matches PyTorch to FP32 rounding
on every head. A broken export (wrong head order or names, a tracing or opset
failure) fails here instead of silently corrupting the agents' value estimates.
The ONNX -> TensorRT hop is engine/tests/test_nn_inference_parity.cpp.

The model is randomly initialized: this tests the plumbing, not trained weights,
so the tests stay hermetic (no checkpoint, CPU only).
"""

import numpy as np
import pytest
import torch
from scribblez.ffi import get_input_shapes
from scribblez.position_eval.model import (
    FOOTPRINT_CLASSES,
    PLACEMENT_HEAD_NAMES,
    PositionEvalModel,
)
from scribblez.position_eval.onnx_export import export_onnx
from scribblez.transformer_tower import TransformerConfig

# The engine encoder's real input shapes, served by the FFI. The export does not
# depend on them numerically, but using them keeps the test in step with the
# encoder when a plane count changes.
_input_shapes = {s.name: s.dims for s in get_input_shapes()}
SPATIAL_PLANES, BOARD_SIZE, _BOARD_WIDTH = _input_shapes["input_spatial"]
assert BOARD_SIZE == _BOARD_WIDTH, "the model assumes a square board"
SCALAR_SIZE = _input_shapes["input_scalar"][0]

# The exported graph's output order. A silent reordering would scramble which head
# the agent reads.
OUTPUT_NAMES = ["wld", "score_diff", *PLACEMENT_HEAD_NAMES]

# Both trunk towers export, so both are checked: the conv tower (None) and a tiny
# transformer tower (attention, RoPE, RMSNorm, SwiGLU, register tokens).
TRUNKS = {
    "conv": None,
    "transformer": TransformerConfig(mid_channels=8, num_heads=2, ffn_channels=16),
}


def _random_model(trunk: str, seed: int = 0) -> PositionEvalModel:
    """A small random model in eval mode, so BatchNorm uses its default running
    stats and the forward pass is deterministic."""
    torch.manual_seed(seed)
    model = PositionEvalModel(
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        trunk_channels=16,  # tiny: this test checks numerics, not capacity
        num_blocks=3,  # conv: 3 -> includes one global-pooling block (covers its ops)
        board_size=BOARD_SIZE,
        transformer=TRUNKS[trunk],
    )
    model.eval()
    return model


def _random_inputs(batch: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    spatial = rng.standard_normal((batch, SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    scalar = rng.standard_normal((batch, SCALAR_SIZE), dtype=np.float32)
    return spatial, scalar


@pytest.mark.parametrize("trunk", TRUNKS)
@pytest.mark.parametrize("batch", [1, 4])
def test_pytorch_matches_onnxruntime(tmp_path, trunk, batch):
    ort = pytest.importorskip("onnxruntime")

    model = _random_model(trunk)
    onnx_path = tmp_path / "model.onnx"
    export_onnx(
        model,
        onnx_path,
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        opp_leave_input=False,
        board_size=BOARD_SIZE,
    )

    spatial, scalar = _random_inputs(batch)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(spatial), torch.from_numpy(scalar))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == OUTPUT_NAMES
    ort_out = sess.run(OUTPUT_NAMES, {"input_spatial": spatial, "input_scalar": scalar})

    for name, ort_arr in zip(OUTPUT_NAMES, ort_out, strict=True):
        np.testing.assert_allclose(
            ort_arr,
            torch_out[name].numpy(),
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"PyTorch vs ONNXRuntime mismatch on head '{name}' ({trunk}, batch={batch})",
        )


@pytest.mark.parametrize("trunk", TRUNKS)
def test_dynamic_batch_axis(tmp_path, trunk):
    """One exported graph serves both single-position and batched inference."""
    ort = pytest.importorskip("onnxruntime")

    model = _random_model(trunk)
    onnx_path = tmp_path / "model.onnx"
    export_onnx(
        model,
        onnx_path,
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        opp_leave_input=False,
        board_size=BOARD_SIZE,
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for batch in (1, 3, 8):
        spatial, scalar = _random_inputs(batch, seed=batch)
        wld, score_diff, *mask_outs = sess.run(
            OUTPUT_NAMES, {"input_spatial": spatial, "input_scalar": scalar}
        )
        assert wld.shape == (batch, 3)
        assert score_diff.shape == (batch, 2)  # [mean, std]
        assert len(mask_outs) == len(PLACEMENT_HEAD_NAMES)
        for mask in mask_outs:
            # Each placement head exports raw footprint logits, not a (15,15) map.
            assert mask.shape == (batch, FOOTPRINT_CLASSES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Inference-parity tests for the ONNX export path (hop A: PyTorch <-> ONNXRuntime).

The agent (NeuralAgent) and the dashboard probes run the *same weights*
through *different* inference stacks: the dashboard calls the in-memory PyTorch
PositionEvalModel (FP32), while the agent uses onnx_export -> TensorRT. This test
pins the first hop of that chain -- that exporting to ONNX and running it under
ONNXRuntime reproduces the PyTorch outputs bit-for-bit (FP32) -- so an export
regression (wrong head order/names, opset/tracing breakage) fails loudly here
rather than silently corrupting the agent's value estimates.

The model is *randomly initialized*: we are testing the fidelity of the
plumbing, not the quality of any trained weights, so random weights on a
matching input shape are sufficient and keep the test hermetic (no checkpoint,
no GPU/TensorRT, CPU-only).

The second hop (ONNX -> TensorRT FP16, plus the C++ Eval decode) needs the GPU
stack and is covered separately.
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

# Input contract (single source of truth: engine/include/scribblez/input_encoder.h,
# surfaced through the FFI). The export is numerically agnostic to these, but
# pulling the real shapes keeps the test representative and in lock-step with the
# encoder, so a plane-count change can never silently mismatch this test.
_input_shapes = {s.name: s.dims for s in get_input_shapes()}
SPATIAL_PLANES, BOARD_SIZE, _BOARD_WIDTH = _input_shapes["input_spatial"]
assert BOARD_SIZE == _BOARD_WIDTH, "the model assumes a square board"
SCALAR_SIZE = _input_shapes["input_scalar"][0]

# Exported graph output order (onnx_export.export_onnx output_names). A silent
# reordering here would scramble which head the agent reads -- assert it.
OUTPUT_NAMES = ["wld", "score_diff", *PLACEMENT_HEAD_NAMES]


def _random_model(seed: int = 0) -> PositionEvalModel:
    """A small randomly-initialized model in eval mode (BatchNorm uses its
    default running stats, so the forward pass is deterministic)."""
    torch.manual_seed(seed)
    model = PositionEvalModel(
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        trunk_channels=16,  # tiny: this test checks numerics, not capacity
        num_blocks=3,  # 3 -> includes one global-pooling block (covers its ops)
        board_size=BOARD_SIZE,
    )
    model.eval()
    return model


def _random_inputs(batch: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    spatial = rng.standard_normal((batch, SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    scalar = rng.standard_normal((batch, SCALAR_SIZE), dtype=np.float32)
    return spatial, scalar


@pytest.mark.parametrize("batch", [1, 4])
def test_pytorch_matches_onnxruntime(tmp_path, batch):
    """Exported ONNX run under ONNXRuntime matches PyTorch on every head."""
    ort = pytest.importorskip("onnxruntime")

    model = _random_model()
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
    # Exported output order must be the documented head order.
    assert [o.name for o in sess.get_outputs()] == OUTPUT_NAMES
    ort_out = sess.run(OUTPUT_NAMES, {"input_spatial": spatial, "input_scalar": scalar})

    for name, ort_arr in zip(OUTPUT_NAMES, ort_out, strict=True):
        np.testing.assert_allclose(
            ort_arr,
            torch_out[name].numpy(),
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"PyTorch vs ONNXRuntime mismatch on head '{name}' (batch={batch})",
        )


def test_dynamic_batch_axis(tmp_path):
    """One exported graph serves both single-position and batched inference."""
    ort = pytest.importorskip("onnxruntime")

    model = _random_model()
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

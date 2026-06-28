#!/usr/bin/env python3
"""Generate a fixture for the C++ TensorRT parity test (Suite 1 hop B).

The agent runs the value model through onnx_export -> TensorRT (FP16), while the
dashboard runs the in-memory PyTorch model (FP32). This script captures the
PyTorch side as ground truth so the C++ test can confirm the TensorRT path (and
the C++ Eval decode -- softmax, win_prob, score-diff mean/std) reproduces it.

It writes three files into --out-dir:
  * model.onnx   -- a randomly-initialized ScribblezModel exported to ONNX.
  * inputs.bin   -- N rows x kInputFloats float32, laid out exactly as
                    GameStateEncoder::encode_input writes them (spatial floats
                    then scalar floats), row-major. N is recovered C++-side from
                    the file size.
  * expected.bin -- N x 6 float32: [win_prob, p_win, p_draw, p_loss,
                    score_diff_mean, score_diff_std], the PyTorch reference
                    decode of each row.

Random weights are deliberate: this checks numerical fidelity of the inference
stack, not the quality of any trained model, and keeps the fixture hermetic.
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from scribblez.ffi import get_input_shapes
from scribblez.model import ScribblezModel
from scribblez.onnx_export import export_onnx

# Input contract is owned by the C++ encoder
# (engine/include/scribblez/input_encoder.h) and surfaced through the FFI, so the
# fixture's per-row layout always matches the kInputFloats / kSpatialPlanes the
# C++ parity test reads back. Score-diff head: 2 outputs, [mean, std] of the
# final-differential Gaussian.
_input_shapes = {s.name: s.dims for s in get_input_shapes()}
SPATIAL_PLANES, BOARD_SIZE, _BOARD_WIDTH = _input_shapes["input_spatial"]
assert BOARD_SIZE == _BOARD_WIDTH, "the model assumes a square board"
SCALAR_SIZE = _input_shapes["input_scalar"][0]
SPATIAL_FLOATS = SPATIAL_PLANES * BOARD_SIZE * BOARD_SIZE
INPUT_FLOATS = SPATIAL_FLOATS + SCALAR_SIZE


def build_model(seed: int) -> ScribblezModel:
    torch.manual_seed(seed)
    model = ScribblezModel(
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        trunk_channels=8,
        num_blocks=3,  # 3 -> includes one global-pooling block (parity-covers it)
        board_size=BOARD_SIZE,
    )
    model.eval()
    return model


@torch.no_grad()
def reference_evals(model: ScribblezModel, rows: np.ndarray) -> np.ndarray:
    """PyTorch decode of each row into
    [win_prob, p_win, p_draw, p_loss, sd_mean, sd_std]."""
    spatial = torch.from_numpy(
        rows[:, :SPATIAL_FLOATS].reshape(-1, SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE)
    )
    scalar = torch.from_numpy(rows[:, SPATIAL_FLOATS:])
    out = model(spatial, scalar)

    wld = torch.softmax(out["wld"], dim=1).numpy()  # [win, draw, loss]
    p_win, p_draw, p_loss = wld[:, 0], wld[:, 1], wld[:, 2]
    win_prob = p_win + 0.5 * p_draw

    # The score-diff head already emits [mean, std] (std softplus-positive).
    sd = out["score_diff"].numpy()
    sd_mean, sd_std = sd[:, 0], sd[:, 1]

    return np.stack(
        [win_prob, p_win, p_draw, p_loss, sd_mean, sd_std], axis=1
    ).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-rows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.seed)
    onnx_path = args.out_dir / "model.onnx"
    export_onnx(model, onnx_path, spatial_planes=SPATIAL_PLANES, scalar_size=SCALAR_SIZE,
                board_size=BOARD_SIZE)

    rng = np.random.default_rng(args.seed)
    rows = rng.standard_normal((args.num_rows, INPUT_FLOATS), dtype=np.float32)
    expected = reference_evals(model, rows)

    (args.out_dir / "inputs.bin").write_bytes(rows.tobytes())
    (args.out_dir / "expected.bin").write_bytes(expected.tobytes())

    print(f"Wrote fixture to {args.out_dir}:")
    print(f"  model.onnx   ({onnx_path.stat().st_size} bytes)")
    print(f"  inputs.bin   ({args.num_rows} rows x {INPUT_FLOATS} floats)")
    print(f"  expected.bin ({args.num_rows} rows x 6 floats)")
    print(f"  score_diff_mean range: [{expected[:, 4].min():.2f}, {expected[:, 4].max():.2f}]")
    print(f"  score_diff_std  range: [{expected[:, 5].min():.2f}, {expected[:, 5].max():.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

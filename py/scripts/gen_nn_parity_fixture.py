#!/usr/bin/env python3
"""Generate a fixture for the C++ TensorRT parity test (Suite 1 hop B).

The agent runs the value model through onnx_export -> TensorRT (FP16), while the
dashboard runs the in-memory PyTorch model (FP32). This script captures the
PyTorch side as ground truth so the C++ test can confirm the TensorRT path (and
the C++ Eval decode -- softmax, win_prob, score-diff mean) reproduces it.

It writes three files into --out-dir:
  * model.onnx   -- a randomly-initialized ScribblezModel exported to ONNX.
  * inputs.bin   -- N rows x kInputFloats float32, laid out exactly as
                    GameStateEncoder::encode_input writes them (spatial floats
                    then scalar floats), row-major. N is recovered C++-side from
                    the file size.
  * expected.bin -- N x 5 float32: [win_prob, p_win, p_draw, p_loss,
                    score_diff_mean], the PyTorch reference decode of each row.

Random weights are deliberate: this checks numerical fidelity of the inference
stack, not the quality of any trained model, and keeps the fixture hermetic.
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from scribblez.model import ScribblezModel
from scribblez.onnx_export import export_onnx

# Input contract: 33 spatial planes on a 15x15 board + 936 scalars
# (engine/include/scribblez/input_encoder.h). Score-diff head: 801 symmetric
# bins, so bin i carries the differential value (i - 400).
SPATIAL_PLANES = 33
BOARD_SIZE = 15
SCALAR_SIZE = 936
SPATIAL_FLOATS = SPATIAL_PLANES * BOARD_SIZE * BOARD_SIZE
INPUT_FLOATS = SPATIAL_FLOATS + SCALAR_SIZE
SCORE_DIFF_BINS = 801


def build_model(seed: int) -> ScribblezModel:
    torch.manual_seed(seed)
    model = ScribblezModel(
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        trunk_channels=16,
        num_blocks=3,  # 3 -> includes one global-pooling block (parity-covers it)
        score_diff_bins=SCORE_DIFF_BINS,
        board_size=BOARD_SIZE,
    )
    model.eval()
    return model


@torch.no_grad()
def reference_evals(model: ScribblezModel, rows: np.ndarray) -> np.ndarray:
    """PyTorch decode of each row into [win_prob, p_win, p_draw, p_loss, sd_mean]."""
    spatial = torch.from_numpy(
        rows[:, :SPATIAL_FLOATS].reshape(-1, SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE)
    )
    scalar = torch.from_numpy(rows[:, SPATIAL_FLOATS:])
    out = model(spatial, scalar)

    wld = torch.softmax(out["wld"], dim=1).numpy()  # [win, draw, loss]
    p_win, p_draw, p_loss = wld[:, 0], wld[:, 1], wld[:, 2]
    win_prob = p_win + 0.5 * p_draw

    sd_pdf = torch.softmax(out["score_diff"], dim=1).numpy()
    bin_values = np.arange(SCORE_DIFF_BINS, dtype=np.float32) - (SCORE_DIFF_BINS - 1) // 2
    sd_mean = sd_pdf @ bin_values

    return np.stack([win_prob, p_win, p_draw, p_loss, sd_mean], axis=1).astype(np.float32)


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
    print(f"  expected.bin ({args.num_rows} rows x 5 floats)")
    print(f"  score_diff_mean range: [{expected[:, 4].min():.2f}, {expected[:, 4].max():.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

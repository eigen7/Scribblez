#!/usr/bin/env python3
"""Generate the fixtures for the C++ TensorRT parity test (Suite 1 hop B).

The agent runs the value model through onnx_export -> TensorRT (FP16), while the
dashboard runs the in-memory PyTorch model (FP32). This script captures the
PyTorch side as ground truth so the C++ test can confirm the TensorRT path (and
the C++ Eval decode -- softmax, win_prob, score-diff mean/std) reproduces it.

One fixture per trunk tower (conv and transformer, the two graph shapes the
engine may be handed) is written under --out-dir/<trunk>/, three files each:
  * model.onnx   -- a randomly-initialized PositionEvalModel exported to ONNX.
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
from pathlib import Path

import numpy as np
import torch
from scribblez.ffi import get_input_shapes
from scribblez.position_eval.model import PositionEvalModel
from scribblez.position_eval.onnx_export import export_onnx
from scribblez.transformer_tower import TransformerConfig
from scribblez.workloads.position_eval import TRUNK_CONV, TRUNK_TRANSFORMER

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

# The tower each fixture exercises: the conv tower (None) and a tiny transformer
# tower, both at the smallest widths that still cover every op the real ones use.
TRUNK_FIXTURES = {
    TRUNK_CONV: None,
    TRUNK_TRANSFORMER: TransformerConfig(mid_channels=8, num_heads=2, ffn_channels=16),
}


def build_model(seed: int, transformer: TransformerConfig | None) -> PositionEvalModel:
    torch.manual_seed(seed)
    model = PositionEvalModel(
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        trunk_channels=8,
        num_blocks=3,  # conv: 3 -> includes one global-pooling block (parity-covers it)
        board_size=BOARD_SIZE,
        transformer=transformer,
    )
    model.eval()
    return model


@torch.no_grad()
def reference_evals(model: PositionEvalModel, rows: np.ndarray) -> np.ndarray:
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

    return np.stack([win_prob, p_win, p_draw, p_loss, sd_mean, sd_std], axis=1).astype(np.float32)


def write_fixture(out_dir: Path, transformer: TransformerConfig | None, num_rows: int, seed: int):
    """Export one tower's model and its reference decode of `num_rows` random rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(seed, transformer)
    onnx_path = out_dir / "model.onnx"
    export_onnx(
        model,
        onnx_path,
        spatial_planes=SPATIAL_PLANES,
        scalar_size=SCALAR_SIZE,
        opp_leave_input=False,
        board_size=BOARD_SIZE,
    )

    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((num_rows, INPUT_FLOATS), dtype=np.float32)
    expected = reference_evals(model, rows)

    (out_dir / "inputs.bin").write_bytes(rows.tobytes())
    (out_dir / "expected.bin").write_bytes(expected.tobytes())

    print(f"Wrote fixture to {out_dir}:")
    print(f"  model.onnx   ({onnx_path.stat().st_size} bytes)")
    print(f"  inputs.bin   ({num_rows} rows x {INPUT_FLOATS} floats)")
    print(f"  expected.bin ({num_rows} rows x 6 floats)")
    print(f"  score_diff_mean range: [{expected[:, 4].min():.2f}, {expected[:, 4].max():.2f}]")
    print(f"  score_diff_std  range: [{expected[:, 5].min():.2f}, {expected[:, 5].max():.2f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-rows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for trunk, transformer in TRUNK_FIXTURES.items():
        write_fixture(args.out_dir / trunk, transformer, args.num_rows, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the fixture for the leaf eval TensorRT parity test.

The engine serves the DeepBot leaf eval model from its ONNX export through
TensorRT (nn::LeafEvalSpec), in GPU rollouts (gpu/gpu_rollout_values.h). This
script records the PyTorch outputs as ground truth for
engine/tests/test_leaf_eval_parity.cpp. py/tools/gpu_parity.py runs it; to
run it by hand, from py/:

    python3 -m scripts.deep_bot_leaf_eval.gen_parity_fixture --out-dir /tmp/leaf_fixture

Files written into --out-dir:
  * model.onnx      -- a randomly initialized conv model with join planes,
                       stamped with LEAVE_VALUES as its leave values.
  * real.onnx       -- the same export stamped with the engine's own leave
                       values, so gpu_rollout_demo accepts it as --leaf-model.
  * leave_values.txt -- LEAVE_VALUES.
  * inputs.bin      -- N rows x the row's floats, spatial then scalar, as
                       LeafEvalInputEncoder::encode writes them.
  * expected.bin    -- N x 5 float32: each row's win/draw/loss logits and its
                       score-difference mean and spread, as PyTorch computes
                       them.

Random weights and rows are deliberate: the check is the inference stack's
fidelity, not a trained model's.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scribblez.deep_bot_leaf_eval.model import LAYOUT, LeafEvalModel, LeafEvalShape
from scribblez.deep_bot_leaf_eval.onnx_export import OUTPUTS, export_onnx
from scribblez.ffi import LeaveValueArrays

BOARD_SIZE = 15
SPATIAL_PLANES = LAYOUT["join_plane0"] + LAYOUT["join_planes"]
SCALAR_SIZE = LAYOUT["pool_size"] + 1
SPATIAL_FLOATS = SPATIAL_PLANES * BOARD_SIZE * BOARD_SIZE
LEAVE_VALUES = "parity-fixture-leave-values"


def build_model(seed: int) -> LeafEvalModel:
    torch.manual_seed(seed)
    shape = LeafEvalShape(trunk="conv", join_planes=True, width=8, depth=2, num_heads=1)
    model = LeafEvalModel(shape)
    with torch.no_grad():
        for p in model.heads.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model.eval()


@torch.no_grad()
def reference(model: LeafEvalModel, rows: np.ndarray) -> np.ndarray:
    spatial = torch.from_numpy(
        rows[:, :SPATIAL_FLOATS].reshape(-1, SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE)
    )
    out = model(spatial, torch.from_numpy(rows[:, SPATIAL_FLOATS:]))
    wld, score = (out[name].reshape(len(rows), -1) for name in OUTPUTS)
    return torch.cat([wld, score], dim=1).numpy().astype(np.float32)


def random_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    """Rows shaped like the encoder's: 0/1 planes, small non-negative scalars."""
    spatial = (rng.random((n, SPATIAL_FLOATS)) < 0.1).astype(np.float32)
    scalar = rng.integers(0, 4, (n, SCALAR_SIZE)).astype(np.float32)
    scalar[:, LAYOUT["score_diff"] :] = rng.standard_normal((n, 4), dtype=np.float32)
    return np.concatenate([spatial, scalar], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-rows", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.seed)
    provenance = {"tag": "parity-fixture", "generation": 0}
    export_onnx(model, out / "model.onnx", leave_values=LEAVE_VALUES, provenance=provenance)
    export_onnx(
        model,
        out / "real.onnx",
        leave_values=LeaveValueArrays().content_hash,
        provenance=provenance,
    )
    (out / "leave_values.txt").write_text(LEAVE_VALUES)

    rows = random_rows(np.random.default_rng(args.seed), args.num_rows)
    (out / "inputs.bin").write_bytes(rows.tobytes())
    (out / "expected.bin").write_bytes(reference(model, rows).tobytes())
    print(f"Wrote the leaf eval fixture ({args.num_rows} rows) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the fixture for the leave inference TensorRT parity test.

The engine serves the leave inference model's tilted head from its ONNX export
through TensorRT (engine/include/belief/leave_inference_net.h). This script
records the PyTorch outputs as ground truth for
engine/tests/test_leave_inference_parity.cpp. py/tools/gpu_parity.py runs it;
to run it by hand, from py/:

    python3 -m scripts.leave_inference.gen_parity_fixture --out-dir /tmp/li_fixture

Files written into --out-dir:
  * model.onnx        -- a randomly initialized tilted-head model, its head
                         perturbed off zero-init so the output depends on the
                         row, stamped with LEAVE_VALUES as its leave values.
  * stale_format.onnx -- the same export stamped with the previous format
                         version, which the loader must refuse.
  * public.onnx       -- the same export stamped as a public model, which the
                         loader must refuse too.
  * leave_values.txt  -- LEAVE_VALUES.
  * inputs.bin        -- N rows x kInputFloats float32, spatial then scalar, as
                         LeaveInferenceInputEncoder::encode writes them.
  * expected.bin      -- N x 29 float32, the exported `leave_params` of each
                         row as PyTorch computes it.

Random weights and rows are deliberate: the check is the inference stack's
fidelity, not a trained model's.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from scribblez.ffi import task_input_shapes
from scribblez.leave_inference.model import LeaveInferenceModel
from scribblez.leave_inference.onnx_export import (
    FORMAT_KEY,
    PUBLIC_KEY,
    export_onnx,
    params_in_points,
)

_shapes = {s.name: s.dims for s in task_input_shapes("leave_inference")}
SPATIAL_PLANES, BOARD_SIZE, _ = _shapes["input_spatial"]
SCALAR_SIZE = _shapes["input_scalar"][0]
SPATIAL_FLOATS = SPATIAL_PLANES * BOARD_SIZE * BOARD_SIZE
LEAVE_VALUES = "parity-fixture-leave-values"


def build_model(seed: int) -> LeaveInferenceModel:
    torch.manual_seed(seed)
    model = LeaveInferenceModel(SPATIAL_PLANES, SCALAR_SIZE, trunk_channels=8, num_blocks=2)
    with torch.no_grad():
        for p in model.head.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model.eval()


@torch.no_grad()
def reference(model: LeaveInferenceModel, rows: np.ndarray) -> np.ndarray:
    spatial = torch.from_numpy(
        rows[:, :SPATIAL_FLOATS].reshape(-1, SPATIAL_PLANES, BOARD_SIZE, BOARD_SIZE)
    )
    out = model(spatial, torch.from_numpy(rows[:, SPATIAL_FLOATS:]))
    return params_in_points(out["theta"], out["a"], out["b"]).numpy().astype(np.float32)


def onnx_format_version(path: Path) -> int:
    meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    return int(meta[FORMAT_KEY])


def write_restamped(src: Path, dst: Path, key: str, value: str):
    """Copy `src` to `dst` with metadata entry `key` set to `value`."""
    m = onnx.load(str(src))
    for entry in m.metadata_props:
        if entry.key == key:
            entry.value = value
    onnx.save(m, str(dst))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-rows", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args.seed)
    export_onnx(
        model,
        out / "model.onnx",
        SPATIAL_PLANES,
        SCALAR_SIZE,
        leave_values=LEAVE_VALUES,
        public=False,
        provenance={"tag": "parity-fixture", "generation": 0, "policy": "random"},
        board_size=BOARD_SIZE,
    )
    model_path = out / "model.onnx"
    stale_version = str(onnx_format_version(model_path) - 1)
    write_restamped(model_path, out / "stale_format.onnx", FORMAT_KEY, stale_version)
    write_restamped(model_path, out / "public.onnx", PUBLIC_KEY, "1")
    (out / "leave_values.txt").write_text(LEAVE_VALUES)

    rng = np.random.default_rng(args.seed)
    rows = rng.standard_normal((args.num_rows, SPATIAL_FLOATS + SCALAR_SIZE), dtype=np.float32)
    (out / "inputs.bin").write_bytes(rows.tobytes())
    (out / "expected.bin").write_bytes(reference(model, rows).tobytes())
    print(f"Wrote the leave inference fixture ({args.num_rows} rows) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

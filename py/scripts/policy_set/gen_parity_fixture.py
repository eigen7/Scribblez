#!/usr/bin/env python3
"""Generate the fixture for the policy-set TensorRT parity test.

The engine serves the policy-set model's policy form from its ONNX export
through TensorRT (nn::PolicySetSpec), for PolicyBot. This script records the
PyTorch logits as ground truth for
engine/tests/test_policy_set_inference_parity.cpp. py/tools/gpu_parity.py runs
it; to run it by hand, from py/:

    python3 -m scripts.policy_set.gen_parity_fixture --out-dir /tmp/policy_fixture

Files written into --out-dir:
  * model_a.onnx / model_b.onnx -- randomly initialized PolicyModels of one
    architecture. The engine refits one cached plan per architecture, and only
    B's own reference shows that loading B after A served B's weights.
  * model_stale.onnx -- model A stamped with policy-set format version 0, for
    the version gate.
  * board.bin, opp_rack.bin -- one position's board row and opponent-rack
    input, float32.
  * move_*.bin -- the candidate set, as the move-set fixture writes it
    (gen_mset_parity_fixture.random_candidates), plus move_equity.bin, M
    float32 equity inputs.
  * expected_a.bin / expected_b.bin -- M float32: each candidate's logit, by
    the training model's own forward (PolicyModel), not the export wrapper.

Random weights and inputs are deliberate: the check is the inference stack's
fidelity, not a trained model's.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scribblez.policy_set import onnx_export, trainer
from scribblez.policy_set.dataset import OPP_RACK_FLOATS
from scribblez.policy_set.model import PolicySetEncoder
from scribblez.policy_set.policy import PolicyModel
from scribblez.workloads.policy_set import PolicySetParams
from scripts.move_set_eval.gen_mset_parity_fixture import input_widths, random_candidates

# A tiny architecture keeps the TensorRT builds to seconds; three blocks
# include a global-pooling block.
PARAMS = PolicySetParams(trunk_channels=8, num_blocks=3, num_heads=2)


def build_model(seed: int, spatial_planes: int, scalar_size: int) -> PolicyModel:
    torch.manual_seed(seed)
    p = PARAMS
    model = PolicyModel(
        PolicySetEncoder(
            spatial_planes, scalar_size, p.trunk_channels, p.num_blocks, p.num_heads,
            p.normalize_scalar,
        )
    )  # fmt: skip
    # The cross-check encoder's output starts at zero; perturbed, so the
    # served cross-check inputs are read.
    with torch.no_grad():
        for param in model.encoder.cross_check_encoder.out.parameters():
            param.add_(0.1 * torch.randn_like(param))
    return model.eval()


@torch.no_grad()
def reference_logits(
    model: PolicyModel, board_row: np.ndarray, opp_rack: np.ndarray, moves: dict, shape
) -> np.ndarray:
    """The training forward's logits over the one position's candidates."""
    spatial_planes, board_size, _ = shape
    spatial_floats = spatial_planes * board_size * board_size
    m = moves["move_scalars"].shape[0]
    batch = {
        "input_spatial": torch.from_numpy(board_row[:spatial_floats]).reshape(
            1, spatial_planes, board_size, board_size
        ),
        "input_scalar": torch.from_numpy(board_row[spatial_floats:]).reshape(1, -1),
        "opp_rack": torch.from_numpy(opp_rack).reshape(1, -1),
        **{name: torch.from_numpy(a) for name, a in moves.items()},
        "move_pos_id": torch.zeros(m, dtype=torch.long),
    }
    for name in ("move_letters", "move_squares", "move_cross_cells"):
        batch[name] = batch[name].long()
    for name in ("move_blanks", "move_tile_mask"):
        batch[name] = batch[name].bool()
    logits, _ = model(batch)
    return logits[0].numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path, help="directory to write the fixture")
    ap.add_argument("--num-moves", type=int, default=37, help="candidate moves in the set")
    ap.add_argument("--seed", type=int, default=0, help="weight and input seed (B uses seed + 1)")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    spatial_planes, board_size, scalar_size = shape = input_widths()
    config = {**asdict(PARAMS), **trainer.versions()}
    provenance = {"tag": "parity", "generation": 0}
    model_a = build_model(args.seed, spatial_planes, scalar_size)
    model_b = build_model(args.seed + 1, spatial_planes, scalar_size)
    onnx_export.export_onnx(model_a, out / "model_a.onnx", config, provenance)
    onnx_export.export_onnx(model_b, out / "model_b.onnx", config, provenance)
    stale = {**config, "policy_set_format_version": 0}
    onnx_export.export_onnx(model_a, out / "model_stale.onnx", stale, provenance)

    rng = np.random.default_rng(args.seed)
    board_row = rng.standard_normal(
        spatial_planes * board_size * board_size + scalar_size, dtype=np.float32
    )
    opp_rack = rng.random(OPP_RACK_FLOATS, dtype=np.float32)
    moves = random_candidates(args.num_moves, args.seed)
    moves["move_equity"] = -rng.random(args.num_moves, dtype=np.float32)

    for name, array in {"board": board_row, "opp_rack": opp_rack, **moves}.items():
        (out / f"{name}.bin").write_bytes(array.tobytes())
    for tag, model in (("a", model_a), ("b", model_b)):
        logits = reference_logits(model, board_row, opp_rack, moves, shape)
        (out / f"expected_{tag}.bin").write_bytes(logits.tobytes())
    print(f"Wrote fixture to {out}: models a/b/stale, {args.num_moves} candidate moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

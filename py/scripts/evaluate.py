#!/usr/bin/env python3
"""Run the evaluation suite over a trained checkpoint.

Usage:
    python -m scripts.evaluate -t mytag                # latest checkpoint
    python -m scripts.evaluate -t mytag --epoch 30

Currently runs the structural monotonicity probe (roadmap 3.1) and writes a
report image plus a JSON summary. This is the Phase 3.4 harness entry point
that the agent-eval (3.2) and calibration (3.3) suites will extend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Importable both as a module (python -m scripts.evaluate) and directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scribblez.eval.monotonicity import render_report, run_probe
from scribblez.eval.probe_bank import ProbeBank, build_probe_bank
from scribblez.ffi import get_input_shapes
from scribblez.model import ScribblezModel
from scribblez.paths import TagPaths


def latest_checkpoint(paths: TagPaths) -> Path:
    ckpts = sorted(paths.checkpoints_dir.glob("model_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {paths.checkpoints_dir}")
    return ckpts[-1]


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> ScribblezModel:
    shapes = {s.name: s.dims for s in get_input_shapes()}
    targs = ckpt.get("args", {})
    model = ScribblezModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        num_blocks=targs.get("num_blocks", 8),
        trunk_channels=targs.get("trunk_channels", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a Scribblez checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch (default: latest).")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    parser.add_argument("--mount-root", type=str, default="/workspace/mount", help="tags/ root.")
    parser.add_argument(
        "--num-probe-positions", type=int, default=12, help="Positions in the monotonicity bank."
    )
    parser.add_argument(
        "--rebuild-bank", action="store_true", help="Rebuild the probe bank from the test split."
    )
    args = parser.parse_args()

    paths = TagPaths(args.tag, args.mount_root)
    device = torch.device(args.device)

    ckpt_path = paths.checkpoint_path(args.epoch) if args.epoch else latest_checkpoint(paths)
    print(f"Loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", 0)
    model = build_model_from_checkpoint(ckpt, device)

    if args.rebuild_bank or not paths.probe_bank_path.exists():
        print(f"Building probe bank from {paths.test_dir} ...")
        bank = build_probe_bank(paths.test_dir, num_positions=args.num_probe_positions)
        bank.save(paths.probe_bank_path)
    else:
        bank = ProbeBank.load(paths.probe_bank_path)

    report = run_probe(model, bank, device)
    img_path = paths.probe_image_path(epoch)
    render_report(bank, report, img_path, title=f"{args.tag} gen-{epoch:04d} (eval)")

    summary = {
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "probe_mean_structural_score": report.mean_structural_score,
        "probe_mean_sigmoid_r2": report.mean_sigmoid_r2,
        "probe_total_violations": report.total_violations,
        "probe_image": str(img_path),
        "per_position": [
            {
                "structural_score": s.structural_score,
                "sigmoid_r2": s.sigmoid_r2,
                "monotonicity_violations": s.monotonicity_violations,
                "total_variation": s.total_variation,
            }
            for s in report.scores
        ],
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_position"}, indent=2))
    print(f"Wrote probe image: {img_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

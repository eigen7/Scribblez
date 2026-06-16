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

from scribblez.eval.monotonicity import render_monotonicity, score_monotonicity
from scribblez.eval.probe_eval import evaluate_subset, write_position_dumps
from scribblez.eval.sampling import build_test_subset
from scribblez.eval.score_belief import render_score_belief
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
    parser.add_argument(
        "--num-probe-positions", type=int, default=12, help="Positions in the evaluation subset."
    )
    parser.add_argument(
        "--probe-diff-range", type=int, default=100, help="Score-diff sweep half-width (±range)."
    )
    parser.add_argument(
        "--rebuild-subset", action="store_true", help="Resample the evaluation subset .slog."
    )
    args = parser.parse_args()

    paths = TagPaths(args.tag)
    device = torch.device(args.device)

    ckpt_path = paths.checkpoint_path(args.epoch) if args.epoch else latest_checkpoint(paths)
    print(f"Loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", 0)
    model = build_model_from_checkpoint(ckpt, device)

    if args.rebuild_subset or not paths.test_subset_slog.exists():
        print(f"Sampling evaluation subset from {paths.test_dir} ...")
        build_test_subset(paths.test_dir, paths.test_subset_slog, num_positions=args.num_probe_positions)
    write_position_dumps(paths.test_subset_slog, paths.test_subset_dir)

    lo, hi = -args.probe_diff_range, args.probe_diff_range
    outs = evaluate_subset(model, paths.test_subset_slog, device, diff_lo=lo, diff_hi=hi)
    report = score_monotonicity(outs.score_diffs, outs.win_rate)
    mono_img = paths.probe_image_path(epoch)
    render_monotonicity(outs.score_diffs, outs.win_rate, report, mono_img, title=f"{args.tag} gen-{epoch:04d} (eval)")
    belief_img = paths.score_belief_image_path(epoch)
    render_score_belief(outs.score_diffs, outs.score_pdf, belief_img, title=f"{args.tag} gen-{epoch:04d} (eval)")

    summary = {
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "probe_mean_structural_score": report.mean_structural_score,
        "probe_mean_sigmoid_r2": report.mean_sigmoid_r2,
        "probe_total_violations": report.total_violations,
        "monotonicity_image": str(mono_img),
        "score_belief_image": str(belief_img),
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
    print(f"Wrote {mono_img} and {belief_img}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

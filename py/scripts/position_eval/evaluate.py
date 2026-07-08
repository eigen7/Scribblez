#!/usr/bin/env python3
"""Run the evaluation suite over a trained checkpoint and store it in the dashboard DB.

Usage:
    ./py/scripts/position_eval/evaluate.py -t mytag                # latest checkpoint
    ./py/scripts/position_eval/evaluate.py -t mytag --epoch 30

Runs the structural probes (3.1) and full-test-set calibration (3.3) for one
checkpoint, writing all results to tags/<tag>/dashboard.db keyed by that epoch.
The Bokeh dashboard renders the plots; this prints the scalar summary.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from scribblez.dashboard import db
from scribblez.dataset import SlogDataset
from scribblez.ffi import get_input_shapes, set_contingent_features
from scribblez.paths import POSITION_EVAL, TagPaths
from scribblez.position_eval.eval.runner import render_boards, run_calibration, run_probes
from scribblez.position_eval.eval.sampling import build_test_subset
from scribblez.position_eval.model import PositionEvalModel
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def latest_checkpoint(paths: TagPaths) -> Path:
    ckpts = sorted(paths.checkpoints_dir.glob("model_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {paths.checkpoints_dir}")
    return ckpts[-1]


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> PositionEvalModel:
    targs = ckpt.get("args", {})
    # The checkpoint records the run's experiment arm; configuring the session
    # from it makes every FFI shape/encode in this process match the model.
    set_contingent_features(targs["contingent_features"])
    shapes = {s.name: s.dims for s in get_input_shapes()}
    model = PositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        num_blocks=targs.get("num_blocks", 10),
        trunk_channels=targs.get("trunk_channels", 192),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a Scribblez checkpoint.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    parser.add_argument(
        "--epoch", type=int, default=None, help="Checkpoint epoch (default: latest)."
    )
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
    parser.add_argument(
        "--no-calibration", action="store_true", help="Skip the full-test-set calibration eval."
    )
    args = parser.parse_args()

    paths = TagPaths(args.tag, POSITION_EVAL)
    device = torch.device(args.device)

    ckpt_path = paths.checkpoint_path(args.epoch) if args.epoch else latest_checkpoint(paths)
    print(f"Loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", 0)
    model = build_model_from_checkpoint(ckpt, device)

    if args.rebuild_subset or not paths.test_subset_slog.exists():
        print(f"Sampling evaluation subset from {paths.test_dir} ...")
        build_test_subset(
            paths.test_dir, paths.test_subset_slog, num_positions=args.num_probe_positions
        )

    conn = db.connect(paths.dashboard_db)
    if not paths.position_dump_path(0).with_suffix(".png").exists():
        render_boards(paths.test_subset_slog, paths.test_subset_dir)

    lo, hi = -args.probe_diff_range, args.probe_diff_range
    record = {"epoch": epoch}
    record.update(run_probes(model, paths.test_subset_slog, device, conn, epoch, lo, hi))

    has_test = paths.test_dir.exists() and any(paths.test_dir.glob("*.slog"))
    if not args.no_calibration and has_test:
        test_ds = SlogDataset(paths.test_dir, post_move=True, apply_symmetry=False)
        record.update(run_calibration(model, test_ds, device, conn, epoch))

    db.write_metrics(conn, epoch, record)
    print(json.dumps(record, indent=2))
    print(f"Wrote results for epoch {epoch} to {paths.dashboard_db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

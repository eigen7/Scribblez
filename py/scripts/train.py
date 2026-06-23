#!/usr/bin/env python3
"""Train a post-move value model from .slog training data.

Usage:
    python -m scripts.train -t mytag --epochs 20 --batch-size 256

Reads the tag's training split (tags/<tag>/data/train), writes .pt checkpoints
and .onnx exports under tags/<tag>/, and after every epoch runs the structural
probes (over a frozen test subset) and full-test-set calibration. All metrics
and eval data are written to tags/<tag>/dashboard.db; the Bokeh dashboard renders
every plot on the fly. Training resumes from the latest checkpoint by default; a
dashboard server is launched alongside unless disabled.
"""

import argparse
import atexit
import shutil
import sys
import time
from pathlib import Path

import torch

from scribblez.dashboard import db, server
from scribblez.dataset import SlogDataset
from scribblez.eval.runner import render_boards, run_calibration, run_probes
from scribblez.eval.sampling import build_test_subset
from scribblez.model import ScribblezModel, compute_loss
from scribblez.onnx_export import export_onnx
from scribblez.paths import TagPaths


def ensure_test_subset(paths: TagPaths, num_positions: int) -> bool:
    """Build the frozen evaluation subset + board images if missing. Returns availability."""
    if paths.test_subset_slog.exists():
        print(f"Using evaluation subset {paths.test_subset_slog}")
    elif not paths.test_dir.exists() or not any(paths.test_dir.glob("*.slog")):
        print(
            f"WARNING: no test split at {paths.test_dir}; skipping structural probes. "
            "Regenerate data with a non-zero --test-ratio to enable them.",
            file=sys.stderr,
        )
        return False
    else:
        print(f"Sampling {num_positions} positions from {paths.test_dir} ...")
        n = build_test_subset(paths.test_dir, paths.test_subset_slog, num_positions=num_positions)
        print(f"  Wrote {n} positions to {paths.test_subset_slog}")
    # Board images are static; render once (when the first one is missing).
    if not paths.position_dump_path(0).with_suffix(".png").exists():
        render_boards(paths.test_subset_slog, paths.test_subset_dir)
    return True


def save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, wld_acc, args, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "wld_acc": wld_acc,
            "args": vars(args),
        },
        path,
    )


def reset_tag(paths: TagPaths):
    """Wipe a tag's prior run artifacts (checkpoints, onnx, dashboard DB) for a fresh start."""
    print(f"--restart: clearing prior run artifacts under {paths.root}", file=sys.stderr)
    shutil.rmtree(paths.checkpoints_dir, ignore_errors=True)
    shutil.rmtree(paths.onnx_dir, ignore_errors=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(paths.dashboard_db) + suffix).unlink(missing_ok=True)


def maybe_resume(paths: TagPaths, model, optimizer, scheduler, device) -> int:
    """Load the latest checkpoint (model/optimizer/scheduler) and return the next epoch."""
    ckpts = sorted(paths.checkpoints_dir.glob("model_epoch_*.pt"))
    if not ckpts:
        return 1
    ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start = int(ckpt["epoch"]) + 1
    print(f"Resuming from {ckpts[-1].name}: next epoch {start} (use --restart for a fresh run)")
    return start


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Scribblez value model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    parser.add_argument("--num-blocks", type=int, default=8, help="Residual blocks.")
    parser.add_argument("--trunk-channels", type=int, default=128, help="Trunk width.")
    parser.add_argument("--lambda-sd", type=float, default=0.05, help="Score-diff loss weight.")
    parser.add_argument("--lambda-opp", type=float, default=0.5, help="Opp-placement loss weight.")
    parser.add_argument(
        "--num-probe-positions", type=int, default=12, help="Positions in the evaluation subset."
    )
    parser.add_argument(
        "--probe-diff-range", type=int, default=100, help="Score-diff sweep half-width (±range)."
    )
    parser.add_argument("--no-probe", action="store_true", help="Disable the structural probes.")
    parser.add_argument(
        "--no-calibration", action="store_true", help="Disable the full-test-set calibration eval."
    )
    parser.add_argument(
        "--calibration-batch-size", type=int, default=512, help="Batch size for calibration."
    )
    parser.add_argument(
        "--restart", action="store_true", help="Ignore existing checkpoints and start fresh."
    )
    parser.add_argument("--no-dashboard", action="store_true", help="Do not launch the dashboard.")
    parser.add_argument(
        "--dashboard-port", type=int, default=server.DEFAULT_PORT, help="Dashboard server port."
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    paths = TagPaths(args.tag)
    device = torch.device(args.device)
    print(f"Device: {device}")

    if args.restart:
        reset_tag(paths)

    # Set up dataset from the training split.
    print(f"Loading data from {paths.train_dir} ...")
    ds = SlogDataset(paths.train_dir, post_move=True, apply_symmetry=True)
    print(f"  {ds.num_samples} samples")

    # Build model. Input widths come from the C++ layout (single source of
    # truth) so they never drift from the encoder.
    spatial_planes = ds.input_shapes["input_spatial"][0]
    scalar_size = ds.input_shapes["input_scalar"][0]
    model = ScribblezModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        num_blocks=args.num_blocks,
        trunk_channels=args.trunk_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = maybe_resume(paths, model, optimizer, scheduler, device)

    # Dashboard DB + record-keeping.
    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, args.tag, vars(args), n_params)

    # Frozen evaluation subset for the structural probes (built once; positions
    # stored in the DB so the dashboard can render the boards).
    probe_enabled = not args.no_probe and ensure_test_subset(paths, args.num_probe_positions)

    # Full held-out test set for per-epoch calibration scoring (loaded once).
    test_ds = None
    has_test = paths.test_dir.exists() and any(paths.test_dir.glob("*.slog"))
    if not args.no_calibration and has_test:
        print(f"Loading calibration test set from {paths.test_dir} ...")
        test_ds = SlogDataset(paths.test_dir, post_move=True, apply_symmetry=False)
        print(f"  {test_ds.num_samples} test positions")
    elif not args.no_calibration:
        print(f"WARNING: no test split at {paths.test_dir}; skipping calibration.", file=sys.stderr)

    # Launch the dashboard alongside training (torn down on exit).
    if not args.no_dashboard:
        proc = server.launch_dashboard(args.dashboard_port, str(paths.mount_root))
        if proc is not None:
            atexit.register(proc.terminate)

    if start_epoch > args.epochs:
        print(f"Already trained through epoch {args.epochs}; nothing to do.")
        return 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses_accum = {"total": 0.0, "wld": 0.0, "score_diff": 0.0, "opp_next_placement": 0.0}
        n_batches = 0
        correct_wld = 0
        total_samples = 0

        # Each epoch gets a unique deterministic seed.
        epoch_seed = epoch * 1000003
        for batch in ds.iter_batches(args.batch_size, seed=epoch_seed):
            input_spatial = batch["input_spatial"].to(device)
            input_scalar = batch["input_scalar"].to(device)
            targets = {
                "wld": batch["wld"].to(device),
                "score_diff": batch["score_diff"].to(device),
                "opp_next_placement": batch["opp_next_placement"].to(device),
            }

            outputs = model(input_spatial, input_scalar)
            losses = compute_loss(
                outputs, targets, lambda_sd=args.lambda_sd, lambda_opp=args.lambda_opp
            )

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            bs = input_spatial.shape[0]
            n_batches += 1
            total_samples += bs
            for k in losses_accum:
                losses_accum[k] += losses[k].item()

            pred = outputs["wld"].argmax(dim=1)
            target_idx = targets["wld"].argmax(dim=1)
            correct_wld += (pred == target_idx).sum().item()

        scheduler.step()
        elapsed = time.time() - t0

        avg = {k: v / max(n_batches, 1) for k, v in losses_accum.items()}
        wld_acc = correct_wld / max(total_samples, 1)
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg['total']:.4f} (wld={avg['wld']:.4f} sd={avg['score_diff']:.4f} "
            f"opp={avg['opp_next_placement']:.4f}) | "
            f"wld_acc={wld_acc:.3f} | lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        # Per-epoch scalar metrics (written to the DB's metrics table).
        record = {
            "epoch": epoch,
            "loss": avg["total"],
            "loss_wld": avg["wld"],
            "loss_score_diff": avg["score_diff"],
            "loss_opp_next_placement": avg["opp_next_placement"],
            "wld_acc": wld_acc,
            "lr": lr_now,
            "elapsed_s": elapsed,
        }

        # Structural probes (every epoch, for a smooth scroll-through).
        if probe_enabled:
            record.update(
                run_probes(
                    model, paths.test_subset_slog, device, conn, epoch,
                    diff_lo=-args.probe_diff_range, diff_hi=args.probe_diff_range,
                )
            )

        # Calibration over the full held-out test set (every epoch).
        if test_ds is not None:
            record.update(
                run_calibration(model, test_ds, device, conn, epoch, args.calibration_batch_size)
            )

        db.write_metrics(conn, epoch, record)

        # Checkpoint (.pt + .onnx).
        ckpt_path = paths.checkpoint_path(epoch)
        save_checkpoint(model, optimizer, scheduler, epoch, avg["total"], wld_acc, args, ckpt_path)
        onnx_path = paths.onnx_path(epoch)
        export_onnx(model, onnx_path, spatial_planes, scalar_size)
        print(f"  -> Saved {ckpt_path} and {onnx_path}")

    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Train a post-move value model from .slog training data.

Usage:
    python -m scripts.train -t mytag --epochs 50 --batch-size 256

Reads the tag's training split (tags/<tag>/data/train), writes .pt checkpoints
and .onnx exports under tags/<tag>/, and after every epoch runs the structural
monotonicity probe over a frozen bank drawn from the held-out test split,
emitting tags/<tag>/monotonicity-probe-analysis/gen-XXXX.png and appending
metrics to tags/<tag>/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Importable both as a module (python -m scripts.train) and directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scribblez.dataset import SlogDataset
from scribblez.eval.monotonicity import render_report, run_probe
from scribblez.eval.probe_bank import ProbeBank, build_probe_bank
from scribblez.model import ScribblezModel, compute_loss
from scribblez.onnx_export import export_onnx
from scribblez.paths import TagPaths


def load_or_build_probe_bank(paths: TagPaths, num_positions: int) -> ProbeBank | None:
    """Load the cached probe bank, or build it from the held-out test split."""
    if paths.probe_bank_path.exists():
        print(f"Loading probe bank from {paths.probe_bank_path}")
        return ProbeBank.load(paths.probe_bank_path)
    if not paths.test_dir.exists() or not any(paths.test_dir.glob("*.slog")):
        print(
            f"WARNING: no test split at {paths.test_dir}; skipping monotonicity probe. "
            "Regenerate data with a non-zero --test-ratio to enable it.",
            file=sys.stderr,
        )
        return None
    print(f"Building probe bank ({num_positions} positions) from {paths.test_dir} ...")
    bank = build_probe_bank(paths.test_dir, num_positions=num_positions)
    bank.save(paths.probe_bank_path)
    print(f"  Saved probe bank to {paths.probe_bank_path}")
    return bank


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Scribblez value model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    parser.add_argument(
        "--mount-root", type=str, default="/workspace/mount", help="Root under which tags/ lives."
    )
    parser.add_argument("--num-blocks", type=int, default=8, help="Residual blocks.")
    parser.add_argument("--trunk-channels", type=int, default=128, help="Trunk width.")
    parser.add_argument("--lambda-sd", type=float, default=0.05, help="Score-diff loss weight.")
    parser.add_argument("--lambda-opp", type=float, default=0.5, help="Opp-placement loss weight.")
    parser.add_argument(
        "--checkpoint-every", type=int, default=10, help="Epochs between .pt/.onnx saves."
    )
    parser.add_argument(
        "--num-probe-positions", type=int, default=12, help="Positions in the monotonicity bank."
    )
    parser.add_argument(
        "--no-probe", action="store_true", help="Disable the per-epoch monotonicity probe."
    )
    args = parser.parse_args()

    paths = TagPaths(args.tag, args.mount_root)
    device = torch.device(args.device)
    print(f"Device: {device}")

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

    # Monotonicity probe bank (frozen, drawn from the held-out test split).
    probe_bank = None if args.no_probe else load_or_build_probe_bank(paths, args.num_probe_positions)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    paths.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
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

        # Per-epoch metrics record.
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

        # Structural monotonicity probe (every epoch, for a smooth scroll-through).
        if probe_bank is not None:
            report = run_probe(model, probe_bank, device)
            img_path = paths.probe_image_path(epoch)
            render_report(probe_bank, report, img_path, title=f"{args.tag} gen-{epoch:04d}")
            record.update(
                probe_mean_structural_score=report.mean_structural_score,
                probe_mean_sigmoid_r2=report.mean_sigmoid_r2,
                probe_total_violations=report.total_violations,
            )
            print(
                f"  probe: mean struct={report.mean_structural_score:.3f} "
                f"mean R²={report.mean_sigmoid_r2:.3f} viol={report.total_violations} "
                f"-> {img_path}"
            )

        with paths.metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        # Checkpoint (.pt + .onnx) on the configured cadence and the final epoch.
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            ckpt_path = paths.checkpoint_path(epoch)
            save_checkpoint(model, optimizer, scheduler, epoch, avg["total"], wld_acc, args, ckpt_path)
            onnx_path = paths.onnx_path(epoch)
            export_onnx(model, onnx_path, spatial_planes, scalar_size)
            print(f"  -> Saved {ckpt_path} and {onnx_path}")

    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

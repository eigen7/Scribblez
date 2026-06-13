#!/usr/bin/env python3
"""Train a post-move value model from .slog training data.

Usage:
    python -m scripts.train -t mytag --epochs 50 --batch-size 256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scribblez.dataset import SlogDataset
from scribblez.model import ScribblezModel, compute_loss


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Scribblez value model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="Data subdirectory tag.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/workspace/mount/data",
        help="Root directory for data.",
    )
    parser.add_argument("--num-blocks", type=int, default=8, help="Residual blocks.")
    parser.add_argument("--trunk-channels", type=int, default=128, help="Trunk width.")
    parser.add_argument("--lambda-sd", type=float, default=0.05, help="Score-diff loss weight.")
    parser.add_argument("--lambda-opp", type=float, default=0.5, help="Opp-placement loss weight.")
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.tag
    ckpt_dir = data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Set up dataset.
    print(f"Loading data from {data_dir} ...")
    ds = SlogDataset(data_dir, post_move=True, apply_symmetry=True)
    print(f"  {ds.num_samples} samples")

    # Build model. Input widths come from the C++ layout (single source of
    # truth) so they never drift from the encoder.
    model = ScribblezModel(
        spatial_planes=ds.input_shapes["input_spatial"][0],
        scalar_size=ds.input_shapes["input_scalar"][0],
        num_blocks=args.num_blocks,
        trunk_channels=args.trunk_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop.
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
            # Move to device.
            input_spatial = batch["input_spatial"].to(device)
            input_scalar = batch["input_scalar"].to(device)
            targets = {
                "wld": batch["wld"].to(device),
                "score_diff": batch["score_diff"].to(device),
                "opp_next_placement": batch["opp_next_placement"].to(device),
            }

            # Forward.
            outputs = model(input_spatial, input_scalar)
            losses = compute_loss(
                outputs, targets, lambda_sd=args.lambda_sd, lambda_opp=args.lambda_opp
            )

            # Backward.
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            # Accumulate stats.
            bs = input_spatial.shape[0]
            n_batches += 1
            total_samples += bs
            for k in losses_accum:
                losses_accum[k] += losses[k].item()

            # WLD accuracy.
            pred = outputs["wld"].argmax(dim=1)
            target_idx = targets["wld"].argmax(dim=1)
            correct_wld += (pred == target_idx).sum().item()

        scheduler.step()
        elapsed = time.time() - t0

        # Report.
        avg = {k: v / max(n_batches, 1) for k, v in losses_accum.items()}
        wld_acc = correct_wld / max(total_samples, 1)
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg['total']:.4f} (wld={avg['wld']:.4f} sd={avg['score_diff']:.4f} "
            f"opp={avg['opp_next_placement']:.4f}) | "
            f"wld_acc={wld_acc:.3f} | lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        # Checkpoint every 10 epochs + final.
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg["total"],
                    "wld_acc": wld_acc,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  -> Saved {ckpt_path}")

    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

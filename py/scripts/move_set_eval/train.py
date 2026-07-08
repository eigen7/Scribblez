#!/usr/bin/env python3
"""Minimal trainer for the move set evaluation model.

Trains the model to distill the position evaluation model over the candidate
sets stored in a directory of .mset/.slog pairs (produced by the
move_set_eval_target_generator), reporting the top-K recall / rank-correlation
metric against the teacher on a held-out split. This is a lean, single-window
loop for iterating on the model and the metric; the generational
generate->train pipeline and dashboard integration (docs/roadmap2.md track A)
are separate, later work.

Usage:
    ./py/scripts/move_set_eval/train.py --data-dir DIR [--holdout-dir DIR] \
        --epochs 20 --batch-positions 64
"""

import argparse
import sys

import torch
from scribblez.ffi import set_contingent_features, set_opp_leave_input
from scribblez.move_set_eval.dataset import MsetDataset
from scribblez.move_set_eval.eval import evaluate
from scribblez.move_set_eval.model import MoveSetEvalModel
from scribblez.move_set_eval.train_loop import LossConfig, run_epoch
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the move set evaluation model from .mset/.slog pairs.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, help="Directory of .mset/.slog training pairs.")
    p.add_argument(
        "--holdout-dir",
        default=None,
        help="Directory of held-out .mset/.slog pairs for the recall metric. "
        "Defaults to evaluating on the training set (a smoke check, not a real held-out score).",
    )
    p.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    p.add_argument("--batch-positions", type=int, default=64, help="Positions per batch.")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    p.add_argument("--device", default="cuda", help="Device (cpu or cuda).")
    p.add_argument("--num-blocks", type=int, default=10, help="Trunk residual blocks.")
    p.add_argument("--trunk-channels", type=int, default=192, help="Trunk width.")
    p.add_argument("--num-heads", type=int, default=4, help="Cross-attention heads.")
    p.add_argument(
        "--contingent-features",
        action="store_true",
        help="Encode the full input layout (with contingent-draw features) for this model's "
        "board trunk. Selects the process-wide engine session's input arm; must be consistent "
        "across a run but is independent of the teacher's arm.",
    )
    p.add_argument("--lambda-sd", type=float, default=0.004, help="Score-diff loss weight.")
    p.add_argument("--huber-delta-mean", type=float, default=10.0, help="Huber delta, mean head.")
    p.add_argument("--huber-delta-std", type=float, default=10.0, help="Huber delta, std head.")
    p.add_argument("--seed", type=int, default=0, help="Shuffle/init seed.")
    p.add_argument("--out", default=None, help="Optional path to save the model state_dict.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    set_contingent_features(args.contingent_features)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_ds = MsetDataset(args.data_dir)
    holdout_ds = MsetDataset(args.holdout_dir) if args.holdout_dir else train_ds
    # The board input arm must carry the opponent-leave block iff the targets
    # were generated under the open-leaves condition.
    if train_ds.open_leaves:
        set_opp_leave_input(True)
    print(
        f"train: {train_ds.num_positions} positions / {train_ds.num_candidates} candidates; "
        f"eval: {holdout_ds.num_positions} positions (open_leaves={train_ds.open_leaves})"
    )

    model = MoveSetEvalModel(
        spatial_planes=train_ds.spatial_planes,
        scalar_size=train_ds.scalar_size,
        trunk_channels=args.trunk_channels,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
    ).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_cfg = LossConfig.from_args(args)

    rows_trained = 0
    for epoch in range(args.epochs):
        batches = train_ds.iter_batches(args.batch_positions, seed=args.seed, epoch_index=epoch)
        result = run_epoch(model, optimizer, batches, device, loss_cfg, rows_trained=rows_trained)
        rows_trained = result.rows_trained
        metrics = evaluate(model, holdout_ds, device, positions_per_batch=args.batch_positions)
        recall = " ".join(f"r@{k}={metrics[f'recall@{k}']:.3f}" for k in (1, 3, 5))
        print(
            f"[epoch {epoch}] loss={result.losses['total']:.4f} "
            f"wld={result.losses['wld']:.4f} sd={result.losses['score_diff']:.2f} | "
            f"{recall} spearman={metrics['spearman']:.3f}"
        )

    if args.out:
        torch.save(model.state_dict(), args.out)
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

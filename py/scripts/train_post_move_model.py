#!/usr/bin/env python3
"""Streaming trainer for the post-move value model.

Unlike the disk pipeline (generate_data.py + train.py), this script generates
HastyBot self-play games on C++ threads and feeds the sampled training rows
straight into the GPU training loop through an in-process ring buffer -- no
.slog files are written. Because game generation is fast and we sample one
position per game (no shuffle needed), fresh games are produced continuously
rather than recycled across epochs.

Usage:
    python -m scripts.train_post_move_model -t mytag --batch-size 256

Phase A scope: prove end-to-end streaming throughput. Validation eval, rolling
checkpoints, ONNX export, and the dashboard are added in a later pass.
"""

import argparse
import sys
import time

import torch

from scribblez.dataset import row_layout, slice_row_batch
from scribblez.ffi import StreamingTrainSource, get_input_shapes
from scribblez.model import ScribblezModel, compute_loss
from scribblez.paths import TagPaths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stream HastyBot self-play directly into post-move model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    p.add_argument("--batch-size", type=int, default=256, help="Minibatch / ring-slot size.")
    p.add_argument("--num-slots", type=int, default=2, help="Ring-buffer slots (double-buffering).")
    p.add_argument("--gen-threads", type=int, default=8, help="C++ game-generation threads.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    p.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    p.add_argument("--num-blocks", type=int, default=8, help="Residual blocks.")
    p.add_argument("--trunk-channels", type=int, default=128, help="Trunk width.")
    p.add_argument("--lambda-sd", type=float, default=0.05, help="Score-diff loss weight.")
    p.add_argument("--lambda-opp", type=float, default=0.5, help="Opp-placement loss weight.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for game generation.")
    p.add_argument("--handicap-max", type=int, default=100, help="Random head-start max (0=off).")
    p.add_argument(
        "--max-positions", type=int, default=0, help="Stop after this many positions (0=run forever)."
    )
    p.add_argument(
        "--log-every", type=int, default=25600, help="Log throughput every this many positions."
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    paths = TagPaths(args.tag)
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Tag root: {paths.root}")
    print(f"Device: {device}")

    # Input widths come from the C++ row layout (single source of truth).
    in_shapes = {s.name: s.dims for s in get_input_shapes()}
    spatial_planes = in_shapes["input_spatial"][0]
    scalar_size = in_shapes["input_scalar"][0]

    model = ScribblezModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        num_blocks=args.num_blocks,
        trunk_channels=args.trunk_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    input_layout, targets = row_layout()
    source = StreamingTrainSource(
        batch_size=args.batch_size,
        num_slots=args.num_slots,
        num_threads=args.gen_threads,
        post_move=True,
        apply_symmetry=True,
        seed=args.seed,
        handicap_max=args.handicap_max,
    )
    source.start()
    print(f"Streaming {args.gen_threads} gen-threads -> {args.num_slots} slots of {args.batch_size}")

    model.train()
    positions = 0
    next_log = args.log_every
    t0 = time.time()
    last_t = t0
    last_positions = 0
    try:
        while args.max_positions == 0 or positions < args.max_positions:
            res = source.next_slot()
            if res is None:
                break
            slot_idx, cpu_tensor = res
            # Copy rows out of the slot (slice_row_batch copies), then release so
            # the producers can refill it while we run forward/backward.
            batch = slice_row_batch(cpu_tensor.numpy(), input_layout, targets)
            input_spatial = batch["input_spatial"].to(device, non_blocking=True)
            input_scalar = batch["input_scalar"].to(device, non_blocking=True)
            tgt = {
                "wld": batch["wld"].to(device, non_blocking=True),
                "score_diff": batch["score_diff"].to(device, non_blocking=True),
                "opp_next_placement": batch["opp_next_placement"].to(device, non_blocking=True),
            }
            source.release(slot_idx)

            outputs = model(input_spatial, input_scalar)
            losses = compute_loss(outputs, tgt, lambda_sd=args.lambda_sd, lambda_opp=args.lambda_opp)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            positions += input_spatial.shape[0]
            if positions >= next_log:
                now = time.time()
                st = source.stats()
                dpos = positions - last_positions
                dt = max(now - last_t, 1e-9)
                # Which side waited more since startup tells us the bottleneck.
                bottleneck = "cpu(gen)" if st["consumer_blocked_ns"] > st["producer_blocked_ns"] else "gpu(train)"
                print(
                    f"pos={positions:>9} | {dpos / dt:8.0f} pos/s | "
                    f"loss={losses['total'].item():.4f} | games={st['games_played']} "
                    f"dropped={st['games_dropped']} | bottleneck={bottleneck} "
                    f"(prod_blk={st['producer_blocked_ns'] / 1e9:.1f}s "
                    f"cons_blk={st['consumer_blocked_ns'] / 1e9:.1f}s)"
                )
                last_t = now
                last_positions = positions
                next_log += args.log_every
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.")
    finally:
        source.stop()

    print(f"Trained on {positions} positions in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

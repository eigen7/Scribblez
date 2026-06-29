#!/usr/bin/env python3
"""Streaming trainer for the max-move-per-lane model.

Sibling to train_post_move_model.py: it generates HastyBot self-play games on
C++ threads and feeds the sampled max-move-per-lane training rows straight into
the GPU training loop through an in-process ring buffer (no .slog files). The
per-lane labels are computed at sample time by enumerating legal moves, so the
streamer needs the lexicon (loaded inside the C++ library).

Unlike the post-move trainer there is no held-out probe/calibration eval or ONNX
export -- this is a representation-learning probe. Eval is the per-lane train
accuracy (does the model get each legal lane's best move and its score right?),
recorded alongside the losses. A single rolling model.pt holds resume state.

Usage:
    python -m scripts.max_move_per_lane.train -t mytag --batch-size 256
"""

import argparse
import atexit
import sys
import time

import torch

from scribblez.dashboard import db, server
from scribblez.dataset import row_layout, slice_row_batch
from scribblez.ffi import (
    StreamingTrainSource,
    get_max_move_per_lane_input_shapes,
    get_max_move_per_lane_target_shapes,
)
from scribblez.max_move_per_lane.model import MaxMovePerLaneModel, compute_loss
from scribblez.paths import MAX_MOVE_PER_LANE, TagPaths
from scribblez.train_common import (
    IntervalStats,
    ThroughputMeter,
    TrainStepWriter,
    add_train_log_args,
    timed_print,
    maybe_resume,
    reset_tag,
    save_rolling_checkpoint,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stream HastyBot self-play directly into max-move-per-lane training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    p.add_argument("--batch-size", type=int, default=256, help="Minibatch / ring-slot size.")
    p.add_argument("--num-slots", type=int, default=2, help="Ring-buffer slots (double-buffering).")
    p.add_argument("--gen-threads", type=int, default=8, help="C++ game-generation threads.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    p.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    p.add_argument("--trunk-channels", type=int, default=128, help="CNN trunk width.")
    p.add_argument("--num-blocks", type=int, default=8, help="Trunk residual blocks.")
    p.add_argument("--lane-layers", type=int, default=4, help="Lane transformer layers.")
    p.add_argument("--lane-heads", type=int, default=4, help="Lane transformer attention heads.")
    p.add_argument("--ffn-mult", type=int, default=4, help="Lane transformer FFN width multiple.")
    p.add_argument("--rack-tokens", type=int, default=4, help="Rack tokens prepended per lane.")
    p.add_argument("--lambda-cdf", type=float, default=1.0, help="Score-CDF (CRPS) loss weight.")
    p.add_argument("--lambda-occ", type=float, default=100.0, help="Occupancy (move) loss weight.")
    p.add_argument("--lambda-has-move", type=float, default=1.0, help="Has-move loss weight.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for game generation.")
    p.add_argument("--handicap-max", type=int, default=100, help="Random head-start max (0=off).")
    p.add_argument(
        "--max-positions", type=int, default=0, help="Stop after this many positions (0=forever)."
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=204800, help="Checkpoint every N positions trained."
    )
    p.add_argument(
        "--log-every", type=int, default=25600, help="Sample throughput every N positions."
    )
    add_train_log_args(p)
    p.add_argument("--restart", action="store_true", help="Clear prior checkpoints/DB.")
    p.add_argument("--no-dashboard", action="store_true", help="Do not launch the dashboard.")
    p.add_argument(
        "--dashboard-port", type=int, default=server.DEFAULT_PORT, help="Dashboard server port."
    )
    return p


def lane_accuracy(outputs: dict, targets: dict) -> dict:
    """Per-legal-lane train accuracy: does the model get each lane's best move and
    score right? Averaged over the lanes that actually have a legal move (plus a
    has-move accuracy over all 30 lanes)."""
    mask = targets["lane_mask"]  # (B, 30)
    legal = mask.sum().clamp_min(1.0)

    pred_bin = outputs["lane_score_logits"].argmax(-1)  # (B, 30)
    score_ok = (((pred_bin == targets["lane_score"].long()).float() * mask).sum() / legal).item()

    # "Move right" == the thresholded occupancy union matches the target exactly.
    pred_occ = (outputs["lane_occupancy_logits"] > 0).float()  # (B, 30, 15, 27)
    lane_match = (pred_occ == targets["lane_occupancy"]).all(dim=-1).all(dim=-1).float()  # (B, 30)
    move_ok = ((lane_match * mask).sum() / legal).item()

    has_pred = (outputs["lane_has_move_logits"] > 0).float()
    has_move_ok = (has_pred == mask).float().mean().item()
    return {"score_acc": score_ok, "move_acc": move_ok, "has_move_acc": has_move_ok}


def _step_metrics(losses: dict, accs: dict) -> dict:
    """One minibatch's dashboard metrics: the total + per-component losses and the
    per-lane accuracies (the names define the dashboard series)."""
    return {
        "loss": losses["total"].item(),
        "loss_score_pdf": losses["score_pdf"].item(),
        "loss_score_cdf": losses["score_cdf"].item(),
        "loss_move": losses["move"].item(),
        "loss_has_move": losses["has_move"].item(),
        **accs,
    }


def run_streaming_training(
    model, optimizer, source, conn, paths, device, args, *, start_ckpt, start_positions, start_step
) -> int:
    """Consume streamed batches; record per-minibatch loss/accuracy, throughput,
    and a rolling checkpoint on cadence. `source` exposes start/next_slot/release/
    stats/stop (the real StreamingTrainSource or a fake in tests)."""
    input_layout, targets_layout = row_layout(
        get_max_move_per_lane_input_shapes(), get_max_move_per_lane_target_shapes()
    )
    source.start()

    model.train()
    positions, step, ckpt_idx = start_positions, start_step, start_ckpt
    interval = IntervalStats()
    writer = TrainStepWriter(conn, args.max_log_points)

    next_log = positions + args.log_every
    next_ckpt = positions + args.checkpoint_every
    t0 = time.time()
    meter = ThroughputMeter(positions, t0)

    try:
        while args.max_positions == 0 or positions < args.max_positions:
            res = source.next_slot()
            if res is None:
                break
            slot_idx, cpu_tensor = res
            batch = slice_row_batch(cpu_tensor.numpy(), input_layout, targets_layout)
            input_spatial = batch["input_spatial"].to(device, non_blocking=True)
            input_scalar = batch["input_scalar"].to(device, non_blocking=True)
            tgt = {
                k: batch[k].to(device, non_blocking=True)
                for k in ("lane_occupancy", "lane_score", "lane_mask")
            }
            source.release(slot_idx)

            outputs = model(input_spatial, input_scalar)
            losses = compute_loss(
                outputs, tgt, lambda_cdf=args.lambda_cdf, lambda_occ=args.lambda_occ,
                lambda_has_move=args.lambda_has_move,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            positions += input_spatial.shape[0]
            step += 1
            with torch.no_grad():
                accs = lane_accuracy(outputs, tgt)
            metrics = _step_metrics(losses, accs)
            interval.update(metrics)
            writer.record(step, positions, metrics)

            if positions >= next_log:
                db.write_throughput(conn, meter.sample(time.time(), positions, source.stats()))
                writer.commit()
                timed_print(
                    f"pos={positions:>9} | loss={metrics['loss']:.4f} "
                    f"move_acc={metrics['move_acc']:.5f} score_acc={metrics['score_acc']:.5f}"
                )
                next_log += args.log_every

            if positions >= next_ckpt:
                ckpt_idx += 1
                writer.commit()
                record = {"epoch": ckpt_idx, "positions": positions, **interval.record()}
                db.write_metrics(conn, ckpt_idx, record)
                save_rolling_checkpoint(
                    paths.rolling_checkpoint, model, optimizer, ckpt_idx, positions, step, args
                )
                timed_print(f"[checkpoint {ckpt_idx}] pos={positions} loss={record['loss']:.4f} "
                    f"-> saved {paths.rolling_checkpoint.name}")
                interval.reset()
                model.train()
                next_ckpt += args.checkpoint_every
    except KeyboardInterrupt:
        timed_print("Interrupted; shutting down.")
    finally:
        writer.close()
        source.stop()

    timed_print(f"Trained on {positions} positions in {time.time() - t0:.1f}s.")
    return positions


def main() -> int:
    args = build_arg_parser().parse_args()

    paths = TagPaths(args.tag, MAX_MOVE_PER_LANE)
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Tag root: {paths.root}\nDevice: {device}")

    if args.restart:
        reset_tag(paths)

    in_shapes = {s.name: s.dims for s in get_max_move_per_lane_input_shapes()}
    spatial_planes = in_shapes["input_spatial"][0]
    scalar_size = in_shapes["input_scalar"][0]
    model = MaxMovePerLaneModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        trunk_channels=args.trunk_channels,
        num_blocks=args.num_blocks,
        lane_layers=args.lane_layers,
        lane_heads=args.lane_heads,
        ffn_mult=args.ffn_mult,
        n_rack_tokens=args.rack_tokens,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, args.tag, vars(args), n_params)
    # Coefficients of each loss term in compute_loss's total (PDF has weight 1),
    # so the dashboard can stack the weighted contributions.
    db.write_loss_weights(conn, {
        "loss_score_pdf": 1.0,
        "loss_score_cdf": args.lambda_cdf,
        "loss_move": args.lambda_occ,
        "loss_has_move": args.lambda_has_move,
    })

    if not args.no_dashboard:
        proc = server.launch_dashboard(args.dashboard_port, str(paths.mount_root), tag=args.tag,
                                       app=server.MAX_MOVE_PER_LANE_APP)
        if proc is not None:
            atexit.register(proc.terminate)

    start_ckpt, start_positions, start_step = maybe_resume(paths, model, optimizer, device)

    source = StreamingTrainSource(
        batch_size=args.batch_size,
        task="max_move_per_lane",
        num_slots=args.num_slots,
        num_threads=args.gen_threads,
        apply_symmetry=True,
        seed=args.seed,
        handicap_max=args.handicap_max,
    )
    print(f"Streaming {args.gen_threads} gen-threads -> {args.num_slots} slots of {args.batch_size}")

    run_streaming_training(
        model, optimizer, source, conn, paths, device, args,
        start_ckpt=start_ckpt, start_positions=start_positions, start_step=start_step,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

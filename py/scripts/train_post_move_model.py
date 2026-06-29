#!/usr/bin/env python3
"""Streaming trainer for the post-move value model.

Unlike the disk pipeline (generate_data.py + train.py), this script generates
HastyBot self-play games on C++ threads and feeds the sampled training rows
straight into the GPU training loop through an in-process ring buffer -- no
.slog training files are written. Because game generation is fast and we sample
one position per game (no shuffle needed), fresh games are produced continuously
rather than recycled across epochs.

A held-out validation set IS written to disk once (so it is stable across
restarts); the model is evaluated against it on a fixed positions-trained cadence
("checkpoints"), and those metrics + the live throughput/backpressure series are
written to the per-tag dashboard DB. A single rolling model.pt holds resume
state; ONNX is exported per checkpoint.

Usage:
    python -m scripts.train_post_move_model -t mytag --batch-size 256
"""

import argparse
import atexit
import sys
import time

import torch

from scribblez.dashboard import db, server
from scribblez.dataset import SlogDataset, row_layout, slice_row_batch
from scribblez.eval.runner import render_boards, run_calibration, run_probes
from scribblez.eval.sampling import build_test_subset
from scribblez.ffi import StreamingTrainSource, get_input_shapes
from scribblez.post_move_value_model import PostMoveValueModel, compute_loss
from scribblez.onnx_export import export_onnx
from scribblez.paths import TagPaths
from scribblez.train_common import (
    ThroughputMeter,
    TrainStepWriter,
    add_train_log_args,
    timed_print,
    maybe_resume,
    reset_tag,
    save_rolling_checkpoint,
)

# Imported lazily-friendly: shelling out to play_game for the one-time val set.
from scripts.generate_data import run_games


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
    p.add_argument("--lambda-sd", type=float, default=0.004, help="Score-diff loss weight.")
    p.add_argument("--lambda-opp", type=float, default=0.5, help="Opp-placement loss weight.")
    p.add_argument(
        "--huber-delta-mean", type=float, default=10.0,
        help="Huber transition point (points) for the score-diff mean head.")
    p.add_argument(
        "--huber-delta-std", type=float, default=10.0,
        help="Huber transition point (points) for the score-diff std head.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for game generation.")
    p.add_argument("--handicap-max", type=int, default=100, help="Random head-start max (0=off).")
    p.add_argument(
        "--max-positions", type=int, default=0, help="Stop after this many positions (0=run forever)."
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=204800,
        help="Eval + checkpoint + ONNX export every this many positions trained.",
    )
    p.add_argument(
        "--log-every", type=int, default=25600, help="Sample throughput every this many positions."
    )
    add_train_log_args(p)
    p.add_argument("--val-games", type=int, default=20000, help="Held-out validation games (first run).")
    p.add_argument(
        "--val-games-per-file", type=int, default=10000, help="Games per validation .slog file."
    )
    p.add_argument(
        "--num-probe-positions", type=int, default=12, help="Positions in the probe subset."
    )
    p.add_argument(
        "--probe-diff-range", type=int, default=100, help="Score-diff sweep half-width (±range)."
    )
    p.add_argument("--no-probe", action="store_true", help="Disable the structural probes.")
    p.add_argument("--no-calibration", action="store_true", help="Disable full-val-set calibration.")
    p.add_argument(
        "--calibration-batch-size", type=int, default=512, help="Batch size for calibration."
    )
    p.add_argument("--restart", action="store_true", help="Clear prior checkpoints/onnx/DB.")
    p.add_argument("--no-dashboard", action="store_true", help="Do not launch the dashboard.")
    p.add_argument(
        "--dashboard-port", type=int, default=server.DEFAULT_PORT, help="Dashboard server port."
    )
    return p


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def ensure_validation_set(paths: TagPaths, args) -> bool:
    """Generate the held-out validation split once (if absent). Returns availability."""
    if paths.test_dir.exists() and any(paths.test_dir.glob("*.slog")):
        return True
    print(f"Generating {args.val_games} validation games to {paths.test_dir} ...")
    rc = run_games(paths.test_dir, args.val_games, args.val_games_per_file, args.gen_threads)
    if rc != 0:
        print(f"WARNING: validation-set generation failed (rc={rc}); eval disabled.", file=sys.stderr)
        return False
    return True


def ensure_probe_subset(paths: TagPaths, num_positions: int) -> bool:
    """Build the frozen probe subset + board images if missing. Returns availability."""
    if paths.test_subset_slog.exists():
        print(f"Using probe subset {paths.test_subset_slog}")
    elif not paths.test_dir.exists() or not any(paths.test_dir.glob("*.slog")):
        return False
    else:
        n = build_test_subset(paths.test_dir, paths.test_subset_slog, num_positions=num_positions)
        print(f"  Wrote {n} probe positions to {paths.test_subset_slog}")
    if not paths.position_dump_path(0).with_suffix(".png").exists():
        render_boards(paths.test_subset_slog, paths.test_subset_dir)
    return True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class _IntervalLoss:
    """Accumulates per-head losses + WLD accuracy over a checkpoint interval."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {
            "total": 0.0, "wld": 0.0, "score_diff": 0.0,
            "score_diff_mean": 0.0, "score_diff_std": 0.0, "opp_next_placement": 0.0,
        }
        self.n_batches = 0
        self.correct = 0
        self.samples = 0

    def update(self, batch_losses: dict, batch_acc: float, n: int):
        for k in self.sums:
            self.sums[k] += batch_losses[k]
        self.n_batches += 1
        self.correct += batch_acc * n
        self.samples += n

    def record(self) -> dict:
        nb = max(self.n_batches, 1)
        return {
            "loss": self.sums["total"] / nb,
            "loss_wld": self.sums["wld"] / nb,
            "loss_score_diff": self.sums["score_diff"] / nb,
            "loss_score_diff_mean": self.sums["score_diff_mean"] / nb,
            "loss_score_diff_std": self.sums["score_diff_std"] / nb,
            "loss_opp_next_placement": self.sums["opp_next_placement"] / nb,
            "wld_acc": self.correct / max(self.samples, 1),
        }


def run_streaming_training(model, optimizer, source, conn, paths, device, args, *, probe_enabled,
                           test_ds, start_ckpt, start_positions, start_step, spatial_planes,
                           scalar_size) -> int:
    """Consume streamed batches; record per-minibatch stats, checkpoint/eval, and
    sample throughput on cadence.

    `source` is any object with start/next_slot/release/stats/stop (the real
    StreamingTrainSource, or a fake in tests). Returns the final positions count.
    """
    input_layout, targets = row_layout()
    source.start()

    model.train()
    positions = start_positions
    step = start_step
    ckpt_idx = start_ckpt
    interval = _IntervalLoss()
    writer = TrainStepWriter(
        conn, args.fine_log_positions, args.coarse_log_window, start_positions=positions
    )

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
            # Copy rows out of the slot (slice_row_batch copies), then release so
            # producers can refill it while we run forward/backward on the GPU.
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
            losses = compute_loss(
                outputs, tgt, lambda_sd=args.lambda_sd, lambda_opp=args.lambda_opp,
                huber_delta_mean=args.huber_delta_mean, huber_delta_std=args.huber_delta_std,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            n = input_spatial.shape[0]
            positions += n
            step += 1
            batch_losses = {k: losses[k].item() for k in interval.sums}
            batch_acc = (outputs["wld"].argmax(1) == tgt["wld"].argmax(1)).float().mean().item()
            interval.update(batch_losses, batch_acc, n)
            writer.record(step, positions, {
                "loss": batch_losses["total"], "loss_wld": batch_losses["wld"],
                "loss_score_diff": batch_losses["score_diff"],
                "loss_score_diff_mean": batch_losses["score_diff_mean"],
                "loss_score_diff_std": batch_losses["score_diff_std"],
                "loss_opp_next_placement": batch_losses["opp_next_placement"],
                "wld_acc": batch_acc,
            })

            if positions >= next_log:
                sample = meter.sample(time.time(), positions, source.stats())
                db.write_throughput(conn, sample)
                writer.commit()
                timed_print(
                    f"pos={positions:>9} | {sample['positions_per_s']:8.0f} pos/s | "
                    f"loss={batch_losses['total']:.4f} | bottleneck={sample['bottleneck']}"
                )
                next_log += args.log_every

            if positions >= next_ckpt:
                ckpt_idx += 1
                writer.commit()
                _checkpoint_and_eval(
                    model, optimizer, conn, paths, device, args, ckpt_idx, positions, step, interval,
                    probe_enabled=probe_enabled, test_ds=test_ds,
                    spatial_planes=spatial_planes, scalar_size=scalar_size,
                )
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


def _checkpoint_and_eval(model, optimizer, conn, paths, device, args, ckpt_idx, positions, step,
                         interval, *, probe_enabled, test_ds, spatial_planes, scalar_size):
    """Run eval against the held-out val set, persist metrics, save .pt + .onnx.

    The dashboard DB is keyed on an integer `epoch`; here it is the monotonic
    checkpoint index (one per `--checkpoint-every` positions), so the entire
    eval/dashboard stack is reused unchanged.
    """
    record = {"epoch": ckpt_idx, "positions": positions, **interval.record()}
    timed_print(
        f"[checkpoint {ckpt_idx}] pos={positions} loss={record['loss']:.4f} "
        f"wld_acc={record['wld_acc']:.5f}"
    )
    if probe_enabled:
        record.update(
            run_probes(
                model, paths.test_subset_slog, device, conn, ckpt_idx,
                diff_lo=-args.probe_diff_range, diff_hi=args.probe_diff_range,
            )
        )
    if test_ds is not None:
        record.update(
            run_calibration(model, test_ds, device, conn, ckpt_idx, args.calibration_batch_size)
        )
    db.write_metrics(conn, ckpt_idx, record)

    save_rolling_checkpoint(paths.rolling_checkpoint, model, optimizer, ckpt_idx, positions, step,
                            args)
    onnx_path = paths.onnx_path(ckpt_idx)
    export_onnx(model, onnx_path, spatial_planes, scalar_size)
    timed_print(f"  -> saved {paths.rolling_checkpoint.name} and {onnx_path.name}")


def main() -> int:
    args = build_arg_parser().parse_args()

    paths = TagPaths(args.tag)
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Tag root: {paths.root}\nDevice: {device}")

    if args.restart:
        reset_tag(paths)

    in_shapes = {s.name: s.dims for s in get_input_shapes()}
    spatial_planes = in_shapes["input_spatial"][0]
    scalar_size = in_shapes["input_scalar"][0]
    model = PostMoveValueModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        num_blocks=args.num_blocks,
        trunk_channels=args.trunk_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, args.tag, vars(args), n_params)
    # Coefficients of each loss term in the optimized total (WLD has weight 1),
    # so the dashboard can stack the weighted contributions.
    db.write_loss_weights(conn, {
        "loss_wld": 1.0,
        "loss_score_diff": args.lambda_sd,
        "loss_opp_next_placement": args.lambda_opp,
    })

    # Held-out validation set (written once) drives the probes + calibration.
    val_ok = ensure_validation_set(paths, args)
    probe_enabled = not args.no_probe and val_ok and ensure_probe_subset(paths, args.num_probe_positions)
    test_ds = None
    if val_ok and not args.no_calibration:
        print(f"Loading calibration val set from {paths.test_dir} ...")
        test_ds = SlogDataset(paths.test_dir, post_move=True, apply_symmetry=False)
        print(f"  {test_ds.num_samples} val positions")

    if not args.no_dashboard:
        proc = server.launch_dashboard(args.dashboard_port, str(paths.mount_root), tag=args.tag)
        if proc is not None:
            atexit.register(proc.terminate)

    start_ckpt, start_positions, start_step = maybe_resume(paths, model, optimizer, device)

    source = StreamingTrainSource(
        batch_size=args.batch_size,
        num_slots=args.num_slots,
        num_threads=args.gen_threads,
        post_move=True,
        apply_symmetry=True,
        seed=args.seed,
        handicap_max=args.handicap_max,
    )
    print(f"Streaming {args.gen_threads} gen-threads -> {args.num_slots} slots of {args.batch_size}")

    run_streaming_training(
        model, optimizer, source, conn, paths, device, args,
        probe_enabled=probe_enabled, test_ds=test_ds,
        start_ckpt=start_ckpt, start_positions=start_positions, start_step=start_step,
        spatial_planes=spatial_planes, scalar_size=scalar_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

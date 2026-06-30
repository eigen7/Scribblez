#!/usr/bin/env python3
"""Streaming trainer for the post-move value model.

Unlike the disk pipeline (generate_data.py + train.py), this script generates
HastyBot self-play games on C++ threads and feeds the sampled training rows
straight into the GPU training loop through an in-process ring buffer -- no
.slog training files are written. Because game generation is fast and we sample
one position per game (no shuffle needed), fresh games are produced continuously
rather than recycled across epochs.

On a fixed positions-trained cadence ("checkpoints"), the model is evaluated over
the committed post-move-value dataset and its predictions are written to the
per-tag dashboard DB, where the Positions tab pairs them with the Monte-Carlo
ground truth; the live throughput/backpressure series are recorded too. A single
rolling model.pt holds resume state; ONNX is exported per checkpoint.

Usage:
    python -m scripts.post_move_value.train -t mytag --batch-size 256
"""

import argparse
import atexit
import sys
import time

import torch
from scribblez.dashboard import db, react_server
from scribblez.dataset import row_layout, slice_row_batch
from scribblez.ffi import StreamingTrainSource, get_input_shapes
from scribblez.paths import POST_MOVE_VALUE, TagPaths
from scribblez.post_move_value import analysis as post_move_analysis
from scribblez.post_move_value.model import PostMoveValueModel, compute_loss
from scribblez.post_move_value.onnx_export import export_onnx
from scribblez.train_common import (
    ThroughputMeter,
    TrainStepWriter,
    add_train_log_args,
    maybe_resume,
    reset_tag,
    save_rolling_checkpoint,
    timed_print,
)


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
        "--huber-delta-mean",
        type=float,
        default=10.0,
        help="Huber transition point (points) for the score-diff mean head.",
    )
    p.add_argument(
        "--huber-delta-std",
        type=float,
        default=10.0,
        help="Huber transition point (points) for the score-diff std head.",
    )
    p.add_argument("--seed", type=int, default=0, help="Base seed for game generation.")
    p.add_argument("--handicap-max", type=int, default=100, help="Random head-start max (0=off).")
    p.add_argument(
        "--max-positions",
        type=int,
        default=0,
        help="Stop after this many positions (0=run forever).",
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
    p.add_argument(
        "--post-move-eval-dataset",
        type=str,
        default=str(post_move_analysis.DEFAULT_DATASET),
        help="GCG dataset the Positions tab evaluates each checkpoint against the "
        "Monte-Carlo ground truth.",
    )
    p.add_argument(
        "--no-post-move-eval",
        action="store_true",
        help="Disable the per-checkpoint post-move-value dataset evaluation.",
    )
    p.add_argument("--restart", action="store_true", help="Clear prior checkpoints/onnx/DB.")
    p.add_argument("--no-dashboard", action="store_true", help="Do not launch the dashboard.")
    p.add_argument(
        "--dashboard-port",
        type=int,
        default=react_server.DEFAULT_DEV_PORT,
        help="React dashboard (Vite) dev-server port; open this in a browser.",
    )
    return p


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def load_post_move_eval(args, spatial_planes: int) -> dict | None:
    """Build the frozen post-move-value input batch once (or None if disabled / the
    dataset is empty / the lexicon is unavailable). At each checkpoint the model is
    run over it and the predictions are written to the dashboard DB, where the
    Positions tab pairs them with the Monte-Carlo ground truth."""
    if args.no_post_move_eval:
        return None
    try:
        names, inputs = post_move_analysis.load_inputs(args.post_move_eval_dataset)
    except Exception as e:  # missing lexicon / unreadable dataset
        timed_print(f"post-move-value eval disabled: {e}")
        return None
    if not names:
        timed_print(
            f"post-move-value eval disabled: no GCG positions in {args.post_move_eval_dataset}"
        )
        return None
    timed_print(f"post-move-value eval: {len(names)} positions from {args.post_move_eval_dataset}")
    return {"inputs": inputs, "spatial_planes": spatial_planes}


def eval_post_move(model, post_move_eval: dict, device, conn, generation: int, positions: int):
    """Run the model over the frozen post-move-value set and store this generation's
    per-position predictions (WLD probabilities + score-delta mean/std)."""
    model.eval()
    preds = post_move_analysis.predict(
        model, post_move_eval["inputs"], post_move_eval["spatial_planes"], device
    )
    db.write_post_move_preds(conn, generation, positions, preds)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class _IntervalLoss:
    """Accumulates per-head losses + WLD accuracy over a checkpoint interval."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {
            "total": 0.0,
            "wld": 0.0,
            "score_diff": 0.0,
            "score_diff_mean": 0.0,
            "score_diff_std": 0.0,
            "opp_next_placement": 0.0,
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


def run_streaming_training(
    model,
    optimizer,
    source,
    conn,
    paths,
    device,
    args,
    *,
    post_move_eval,
    start_ckpt,
    start_positions,
    start_step,
    spatial_planes,
    scalar_size,
) -> int:
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
                outputs,
                tgt,
                lambda_sd=args.lambda_sd,
                lambda_opp=args.lambda_opp,
                huber_delta_mean=args.huber_delta_mean,
                huber_delta_std=args.huber_delta_std,
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
            writer.record(
                step,
                positions,
                {
                    "loss": batch_losses["total"],
                    "loss_wld": batch_losses["wld"],
                    "loss_score_diff": batch_losses["score_diff"],
                    "loss_score_diff_mean": batch_losses["score_diff_mean"],
                    "loss_score_diff_std": batch_losses["score_diff_std"],
                    "loss_opp_next_placement": batch_losses["opp_next_placement"],
                    "wld_acc": batch_acc,
                },
            )

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
                    model,
                    optimizer,
                    conn,
                    paths,
                    device,
                    args,
                    ckpt_idx,
                    positions,
                    step,
                    interval,
                    post_move_eval=post_move_eval,
                    spatial_planes=spatial_planes,
                    scalar_size=scalar_size,
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


def _checkpoint_and_eval(
    model,
    optimizer,
    conn,
    paths,
    device,
    args,
    ckpt_idx,
    positions,
    step,
    interval,
    *,
    post_move_eval,
    spatial_planes,
    scalar_size,
):
    """Persist this checkpoint's train metrics + the post-move-value dataset eval,
    then save the rolling .pt and export ONNX.

    The dashboard DB is keyed on an integer `epoch`; here it is the monotonic
    checkpoint index (one per `--checkpoint-every` positions).
    """
    record = {"epoch": ckpt_idx, "positions": positions, **interval.record()}
    timed_print(
        f"[checkpoint {ckpt_idx}] pos={positions} loss={record['loss']:.4f} "
        f"wld_acc={record['wld_acc']:.5f}"
    )
    db.write_metrics(conn, ckpt_idx, record)
    if post_move_eval is not None:
        eval_post_move(model, post_move_eval, device, conn, ckpt_idx, positions)

    save_rolling_checkpoint(
        paths.rolling_checkpoint, model, optimizer, ckpt_idx, positions, step, args
    )
    onnx_path = paths.onnx_path(ckpt_idx)
    export_onnx(model, onnx_path, spatial_planes, scalar_size)
    timed_print(f"  -> saved {paths.rolling_checkpoint.name} and {onnx_path.name}")


def main() -> int:
    args = build_arg_parser().parse_args()

    paths = TagPaths(args.tag, POST_MOVE_VALUE)
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
    db.write_loss_weights(
        conn,
        {
            "loss_wld": 1.0,
            "loss_score_diff": args.lambda_sd,
            "loss_opp_next_placement": args.lambda_opp,
        },
    )

    post_move_eval = load_post_move_eval(args, spatial_planes)

    if not args.no_dashboard:
        for proc in react_server.spawn(
            "post_move_value", str(paths.mount_root), dev_port=args.dashboard_port, tag=args.tag
        ):
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
    print(
        f"Streaming {args.gen_threads} gen-threads -> {args.num_slots} slots of {args.batch_size}"
    )

    run_streaming_training(
        model,
        optimizer,
        source,
        conn,
        paths,
        device,
        args,
        post_move_eval=post_move_eval,
        start_ckpt=start_ckpt,
        start_positions=start_positions,
        start_step=start_step,
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

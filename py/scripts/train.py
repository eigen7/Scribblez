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
import math
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
from scribblez.util import fmt_duration


def print_epoch_progress(epoch, total_epochs, done_batches, total_batches, samples, t0, final=False):
    """Render an in-place per-epoch progress line: batches done out of the epoch
    total, throughput, and a within-epoch ETA. `final` ends the line with a
    newline so the epoch summary that follows prints cleanly."""
    elapsed = time.time() - t0
    rate = samples / elapsed if elapsed > 0 else 0.0
    pct = 100.0 * done_batches / total_batches if total_batches else 0.0
    remaining = max(total_batches - done_batches, 0)
    eta = elapsed * remaining / done_batches if done_batches else 0.0
    sys.stdout.write(
        f"\rEpoch {epoch:3d}/{total_epochs} | batch {done_batches}/{total_batches} "
        f"({pct:5.1f}%) | {rate / 1000:.1f}k samples/s | ETA {fmt_duration(eta)}    "
    )
    sys.stdout.write("\n" if final else "")
    sys.stdout.flush()


def ensure_test_subset(paths: TagPaths, source_test_dir: Path, num_positions: int) -> bool:
    """Build the frozen evaluation subset + board images if missing. The subset is
    sampled from `source_test_dir` (which may belong to a different tag when data
    is decoupled from the output tag) and written under the output tag's `paths`.
    Returns availability."""
    if paths.test_subset_slog.exists():
        print(f"Using evaluation subset {paths.test_subset_slog}")
    elif not source_test_dir.exists() or not any(source_test_dir.glob("*.slog")):
        print(
            f"WARNING: no test split at {source_test_dir}; skipping structural probes. "
            "Regenerate data with a non-zero --test-ratio to enable them.",
            file=sys.stderr,
        )
        return False
    else:
        print(f"Sampling {num_positions} positions from {source_test_dir} ...")
        n = build_test_subset(source_test_dir, paths.test_subset_slog, num_positions=num_positions)
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


def load_init_weights(path: Path, model, device) -> None:
    """Warm-start: copy ONLY the model weights from another run's checkpoint into
    `model`, for policy iteration (seed iter N+1 from iter N's model).

    The optimizer and LR schedule are left fresh -- only the network parameters
    are seeded. The source checkpoint is opened read-only and never written, so
    the init-from model is left completely untouched; this run's checkpoints go
    to its own (different) tag directory.
    """
    if not path.exists():
        raise FileNotFoundError(f"--init-from checkpoint not found: {path}")
    if path.suffix != ".pt":
        raise ValueError(
            f"--init-from expects a PyTorch .pt checkpoint (under tags/<tag>/checkpoints/), "
            f"not '{path.name}'. The models/ dir holds .onnx exports, which cannot seed training."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    model.load_state_dict(state)
    print(f"Warm-started model weights from {path} (optimizer + LR schedule fresh)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Scribblez value model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True,
                        help="Output tag: checkpoints, .onnx, and the dashboard DB go here.")
    parser.add_argument(
        "--data-tag", action="append", default=None, metavar="TAG",
        help="Tag whose data/{train,test} splits supply the training data. Repeatable "
             "to train on the union of several datasets. Defaults to --tag, i.e. data "
             "and model share a tag (the original behavior). Use a separate data tag to "
             "train many model variants over one dataset without copying it.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument(
        "--turns-per-game", type=int, default=0,
        help="Per-game turn subsampling per epoch. 0 = every eligible turn (the "
             "full expanded epoch). 1 = one turn per game per epoch, so an epoch "
             "is one decorrelated row per game (no two rows share a game) and is "
             "~eligible_turns x smaller; K > 1 = K turns per game. A fresh turn "
             "window is drawn each epoch, so over E epochs a game contributes "
             "min(E*K, eligible_turns) distinct positions.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    parser.add_argument("--num-blocks", type=int, default=10, help="Residual blocks.")
    parser.add_argument("--trunk-channels", type=int, default=192, help="Trunk width.")
    parser.add_argument("--lambda-sd", type=float, default=0.05, help="Score-diff loss weight.")
    parser.add_argument("--lambda-opp", type=float, default=0.5, help="Opp-placement loss weight.")
    parser.add_argument(
        "--huber-delta-mean", type=float, default=1.0,
        help="Huber transition point (points) for the score-diff mean head.")
    parser.add_argument(
        "--huber-delta-var", type=float, default=1.0,
        help="Huber transition point (points^2) for the score-diff variance head.")
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
    parser.add_argument(
        "--init-from", type=str, default="",
        help="Warm-start: load model weights (only) from this .pt checkpoint before "
             "training (policy iteration: seed iter N+1 from iter N's model). The source "
             "file is read-only and a fresh optimizer/LR schedule is used; ignored when "
             "resuming this tag's own checkpoints.",
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

    # Data source(s): the --data-tag list (union), or the output tag itself.
    data_tags = args.data_tag or [args.tag]
    data_paths = [TagPaths(t) for t in data_tags]
    train_dirs = [p.train_dir for p in data_paths]
    test_dirs = [p.test_dir for p in data_paths]
    if data_tags != [args.tag]:
        print(f"Data tags: {', '.join(data_tags)}  ->  model tag: {args.tag}")

    # Set up dataset from the training split(s).
    print(f"Loading data from {', '.join(str(d) for d in train_dirs)} ...")
    ds = SlogDataset(train_dirs, post_move=True, apply_symmetry=True)
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
    # Warm-start only on a genuinely fresh run for this tag; an in-progress resume
    # (start_epoch > 1) already carries this tag's own weights and takes priority.
    if args.init_from and start_epoch == 1:
        load_init_weights(Path(args.init_from), model, device)

    # Dashboard DB + record-keeping.
    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, args.tag, vars(args), n_params)

    # Frozen evaluation subset for the structural probes (built once; positions
    # stored in the DB so the dashboard can render the boards). The probe bank is
    # sampled from the first data tag's test split.
    probe_enabled = not args.no_probe and ensure_test_subset(
        paths, test_dirs[0], args.num_probe_positions
    )

    # Full held-out test set for per-epoch calibration scoring (loaded once),
    # over the union of every data tag's test split.
    test_ds = None
    has_test = any(d.exists() and any(d.glob("*.slog")) for d in test_dirs)
    if not args.no_calibration and has_test:
        print(f"Loading calibration test set from {', '.join(str(d) for d in test_dirs)} ...")
        present = [d for d in test_dirs if d.exists() and any(d.glob("*.slog"))]
        test_ds = SlogDataset(present, post_move=True, apply_symmetry=False)
        print(f"  {test_ds.num_samples} test positions")
    elif not args.no_calibration:
        dirs = ", ".join(str(d) for d in test_dirs)
        print(f"WARNING: no test split at {dirs}; skipping calibration.", file=sys.stderr)

    # Launch the dashboard alongside training (torn down on exit).
    if not args.no_dashboard:
        proc = server.launch_dashboard(args.dashboard_port, str(paths.mount_root), tag=args.tag)
        if proc is not None:
            atexit.register(proc.terminate)

    if start_epoch > args.epochs:
        print(f"Already trained through epoch {args.epochs}; nothing to do.")
        return 0

    # Rows per epoch: the full expanded count, or (with --turns-per-game K) at
    # most K rows per game, capped by the full count. Used to size the progress
    # line / ETA (last batch may be partial).
    if args.turns_per_game > 0:
        epoch_samples = min(args.turns_per_game * ds.num_games, ds.num_samples)
    else:
        epoch_samples = ds.num_samples
    num_batches_est = max(1, math.ceil(epoch_samples / args.batch_size))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        last_progress = 0.0
        losses_accum = {"total": 0.0, "wld": 0.0, "score_diff": 0.0, "opp_next_placement": 0.0}
        n_batches = 0
        correct_wld = 0
        total_samples = 0

        # Each epoch gets a unique deterministic seed. epoch_index selects which
        # per-game turns are drawn when --turns-per-game > 0, so successive epochs
        # cover distinct turns.
        epoch_seed = epoch * 1000003
        for batch in ds.iter_batches(
            args.batch_size, seed=epoch_seed,
            turns_per_game=args.turns_per_game, epoch_index=epoch,
        ):
            input_spatial = batch["input_spatial"].to(device)
            input_scalar = batch["input_scalar"].to(device)
            targets = {
                "wld": batch["wld"].to(device),
                "score_diff": batch["score_diff"].to(device),
                "opp_next_placement": batch["opp_next_placement"].to(device),
            }

            outputs = model(input_spatial, input_scalar)
            losses = compute_loss(
                outputs, targets, lambda_sd=args.lambda_sd, lambda_opp=args.lambda_opp,
                huber_delta_mean=args.huber_delta_mean, huber_delta_var=args.huber_delta_var,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            bs = input_spatial.shape[0]
            n_batches += 1
            total_samples += bs
            for k in losses_accum:
                losses_accum[k] += losses[k].item()

            # Throttled in-place progress line (~1 Hz) so long epochs show life.
            if time.time() - last_progress > 1.0:
                print_epoch_progress(epoch, args.epochs, n_batches, num_batches_est,
                                     total_samples, t0)
                last_progress = time.time()

            pred = outputs["wld"].argmax(dim=1)
            target_idx = targets["wld"].argmax(dim=1)
            correct_wld += (pred == target_idx).sum().item()

        # Finalize the in-place progress line (true batch count) before the
        # epoch summary so they don't collide on the same terminal line.
        print_epoch_progress(epoch, args.epochs, n_batches, n_batches, total_samples, t0,
                             final=True)

        # Read the LR that was actually in effect for the epoch just finished
        # *before* advancing the schedule. Stepping first and reading after would
        # report the next epoch's LR -- and with a single-epoch cosine schedule
        # (T_max == 1) that next value is 0, which looks like no training happened
        # even though this epoch ran at the full base LR.
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - t0

        avg = {k: v / max(n_batches, 1) for k, v in losses_accum.items()}
        wld_acc = correct_wld / max(total_samples, 1)
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

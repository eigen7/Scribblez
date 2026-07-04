#!/usr/bin/env python3
"""Trainer for the post-move value model: an open-ended generate->train loop.

  1. Ensure the current generation's self-play games exist on disk (generating
     them via the play_game binary if missing), then train a few epochs over a
     sliding window of the most recent complete generations -- reusing each
     position across epochs and, with --turns-per-game, sampling several turns
     per game.
  2. Advance to the next generation, evicting generations older than the window.

Reuse across epochs extracts more gradient signal from each generated game than a
one-position-per-game stream, while the on-disk generation lifecycle makes the run
restartable at any point.

Everything is keyed on cumulative rows trained (the rows-clock): the dashboard
x-axis, the warmup learning rate, and the restart cursor. A single rolling
model.pt holds resume state, so stopping and restarting the script continues
exactly where it left off. The base learning rate and the CPU-thread pools
(game-generation threads, DataLoader workers, torch intra-op threads) are live
controls in the per-tag dashboard.db: the dashboard Controls tab retunes them,
and the trainer adopts changes at its next epoch / generation.

Self-play here is HastyBot vs. HastyBot (no GPU contention with training). The
--contingent-features switch selects the input-encoding arm (full layout vs. the
smaller ablation baseline); an optional lexicon module (--lexicon-*) attaches
compiled-dictionary features to the trunk.

Usage:
    python -m scripts.post_move_value.train -t mytag \
        --games-per-generation 20000 --reuse-per-position 2 --window 4
"""

import argparse
import atexit
import functools
import shutil
import sys
import time

import torch
from scribblez.dashboard import db, react_server
from scribblez.dataset import SlogDataset
from scribblez.ffi import get_input_shapes, read_file_header, set_contingent_features
from scribblez.generational import checkpoint, generation, lifecycle
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import (
    CONTROL_BASE_LR,
    CpuController,
    LrController,
    init_controls,
    progress_line,
)
from scribblez.generational.lr import effective_lr
from scribblez.generational.reuse import effective_reuse, epochs_for_reuse
from scribblez.lexical_tool.modules import LexiconArgs
from scribblez.paths import POST_MOVE_VALUE, TagPaths
from scribblez.post_move_value import analysis as post_move_analysis
from scribblez.post_move_value.model import PostMoveValueModel
from scribblez.post_move_value.onnx_export import export_onnx
from scribblez.post_move_value.train_loop import LossConfig, run_epoch
from scribblez.train_common import reset_tag, timed_print
from scripts.generate_data import run_games


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generational generate->train loop for the post-move value model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")

    # Generation lifecycle.
    p.add_argument(
        "--games-per-generation", type=int, default=20000, help="Self-play games per generation."
    )
    p.add_argument(
        "--reuse-per-position",
        type=float,
        default=2.0,
        help="Target gradient passes per unique eligible position over a generation's "
        "lifetime in the window. Epochs-per-generation is derived from this and the "
        "generation's measured average eligible turns/game (num_samples/num_games).",
    )
    p.add_argument(
        "--epochs-per-generation",
        type=int,
        default=0,
        help="Explicit epochs-per-generation, overriding --reuse-per-position. "
        "0 (default) derives it from --reuse-per-position.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=4,
        help="Generations kept and trained over (sliding window). Older ones are "
        "evicted. <=0 keeps every generation (unbounded corpus).",
    )
    p.add_argument(
        "--turns-per-game",
        type=int,
        default=1,
        help="Turns sampled per game per epoch (a fresh window each epoch). 0 = every "
        "eligible turn; K>1 = K distinct turns/game/epoch.",
    )
    p.add_argument("--games-per-file", type=int, default=1000, help="Games per .slog file.")
    p.add_argument(
        "--gen-threads",
        type=int,
        default=8,
        help="Initial parallel game-generation threads (a live control retuned per generation).",
    )
    p.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="Initial C++ DataLoader decode/shuffle workers (a live control, retuned per gen).",
    )
    p.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Initial PyTorch intra-op threads (a live control); 0 seeds torch's own default.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base PRNG seed; generation N uses seed+N. 0 = random per generation.",
    )
    p.add_argument("--hasty-temperature", type=float, default=0.0, help="HastyBot softmax temp.")
    p.add_argument(
        "--hasty-top-k", type=int, default=10, help="HastyBot candidate count if temp>0."
    )
    p.add_argument("--hasty-temp-min-bag", type=int, default=0, help="Sample only above this bag.")

    # Optimization.
    p.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial base learning rate. Becomes a live control kept across restarts "
        "and stepped down by hand from the dashboard Controls tab; --restart resets it.",
    )
    p.add_argument(
        "--warmup-rows",
        type=int,
        default=0,
        help="Linearly warm the LR from 0 over the first this-many rows of the whole run.",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    p.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda).")
    p.add_argument("--num-blocks", type=int, default=10, help="Residual blocks.")
    p.add_argument("--trunk-channels", type=int, default=192, help="Trunk width.")
    p.add_argument(
        "--contingent-features",
        action="store_true",
        help="Encode the full input layout including the contingent-draw potential features. "
        "Off skips their move generation entirely and encodes/trains the smaller base layout "
        "-- the ablation baseline. This one switch is baked into the process-wide engine "
        "session, so the input shapes, the model, the ONNX export, and every eval batch follow it.",
    )
    LexiconArgs.add_arguments(p)
    p.add_argument("--lambda-sd", type=float, default=0.004, help="Score-diff loss weight.")
    p.add_argument("--lambda-opp", type=float, default=0.5, help="Opp-placement loss weight.")
    p.add_argument("--huber-delta-mean", type=float, default=10.0, help="Huber delta, mean head.")
    p.add_argument("--huber-delta-std", type=float, default=10.0, help="Huber delta, std head.")
    p.add_argument(
        "--max-rows", type=int, default=0, help="Stop after this many rows trained (0=forever)."
    )

    # Per-checkpoint evaluation over the committed post-move-value GCG dataset
    # (penultimate-bingo positions with Monte-Carlo ground truth); the Positions
    # tab pairs the model's predictions with that ground truth.
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
    p.add_argument(
        "--post-move-quality-dataset",
        type=str,
        default=str(post_move_analysis.LARGE_DATASET),
        help="Large penultimate-bingo dataset the Loss tab's aggregate model-vs-Monte-Carlo "
        "quality curves are measured over each checkpoint.",
    )
    p.add_argument(
        "--no-post-move-quality",
        action="store_true",
        help="Disable the per-checkpoint aggregate quality evaluation on the large dataset.",
    )

    p.add_argument("--restart", action="store_true", help="Wipe checkpoints/onnx/db/generations.")
    p.add_argument("--no-dashboard", action="store_true", help="Do not launch the dashboard.")
    p.add_argument(
        "--dashboard-port",
        type=int,
        default=react_server.DEFAULT_DEV_PORT,
        help="React dashboard (Vite) dev-server port.",
    )
    return p


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def build_player_spec(args) -> str:
    """The `--player` value for both seats: HastyBot, optionally temperature-sampled
    over equity for exploration."""
    if args.hasty_temperature > 0:
        return (
            f"--type=hastybot --temperature={args.hasty_temperature} "
            f"--top-k={args.hasty_top_k} --temperature-min-bag={args.hasty_temp_min_bag}"
        )
    return "--type=hastybot"


def gen_seed(args, index: int) -> int:
    """PRNG seed for generation `index`: seed+index when a base seed is set, else 0
    (each generation then draws a fresh random seed from the binary)."""
    return args.seed + index if args.seed else 0


def make_generate_fn(args, player_spec: str, cpu):
    """Production generate_fn for ensure_generation: shell out to play_game with
    the live game-generation thread count."""

    def generate_fn(gen_dir, num_games: int, seed: int) -> int:
        return run_games(
            gen_dir, num_games, args.games_per_file, cpu.gen_threads, player_spec, seed
        )

    return generate_fn


def count_games(gen_dir) -> int:
    """Committed game count across a generation's .slog files (from their headers)."""
    return sum(read_file_header(f)[0] for f in sorted(gen_dir.glob("*.slog")))


def restart_run(paths: TagPaths):
    """Wipe prior run artifacts (checkpoints, onnx, dashboard DB) and the train
    generations for a fresh start."""
    reset_tag(paths)
    shutil.rmtree(paths.generations_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _rows_left(args, state: GenerationalState) -> bool:
    return args.max_rows == 0 or state.rows_trained < args.max_rows


def _checkpoint_and_eval(model, optimizer, conn, paths, device, args, state, result, elapsed, ctx):
    """Record this epoch's metrics + eval (keyed on the checkpoint index, with the
    rows-clock stored as `positions`), save the rolling checkpoint, export ONNX."""
    sys.stdout.write("\n")
    avg = result.losses
    ci = state.checkpoint_index
    base_lr = db.read_control(conn, CONTROL_BASE_LR, default=args.lr)
    lr_now = effective_lr(base_lr, state.rows_trained, args.warmup_rows)
    timed_print(
        f"[gen {state.generation_index} epoch {state.epoch_in_generation}] ckpt {ci} "
        f"rows={state.rows_trained} loss={avg['total']:.4f} wld_acc={result.wld_acc:.4f} "
        f"lr={lr_now:.2e} {elapsed:.1f}s"
    )
    record = {
        "epoch": ci,
        "positions": state.rows_trained,
        "generation": state.generation_index,
        "loss": avg["total"],
        "loss_wld": avg["wld"],
        "loss_score_diff": avg["score_diff"],
        "loss_score_diff_mean": avg["score_diff_mean"],
        "loss_score_diff_std": avg["score_diff_std"],
        "loss_opp_next_placement": avg["opp_next_placement"],
        "wld_acc": result.wld_acc,
        "lr": lr_now,
        "elapsed_s": elapsed,
    }
    # Aggregate model-vs-Monte-Carlo quality over the large held-out dataset,
    # folded into the same metrics record so the Loss tab plots it alongside the
    # training curves.
    if ctx["post_move_quality"] is not None:
        record.update(eval_post_move_quality(model, ctx["post_move_quality"], device))
        timed_print(
            f"  quality: win_mae={record['eval_win_mae']:.4f} "
            f"sd_mean_mae={record['eval_sd_mean_mae']:.1f}"
        )
    db.write_metrics(conn, ci, record)
    if ctx["post_move_eval"] is not None:
        eval_post_move(model, ctx["post_move_eval"], device, conn, ci, state.rows_trained)
    checkpoint.save(paths, model, optimizer, state, args)
    onnx_path = paths.onnx_path(ci)
    export_onnx(
        model,
        onnx_path,
        ctx["spatial_planes"],
        ctx["scalar_size"],
        contingent_features=args.contingent_features,
    )


def epochs_this_generation(args, ds) -> int:
    """The epoch count to run over the current window: --epochs-per-generation when
    set explicitly, otherwise derived from --reuse-per-position and the window's
    measured average eligible turns per game."""
    if args.epochs_per_generation > 0:
        return args.epochs_per_generation
    avg_eligible = ds.num_samples / max(ds.num_games, 1)
    return epochs_for_reuse(args.reuse_per_position, args.window, args.turns_per_game, avg_eligible)


def train_one_generation(
    model, optimizer, conn, paths, device, args, state, loss_cfg, lr_controller, cpu, ctx
) -> int:
    """Train `epochs` passes over the current window, resuming at
    state.epoch_in_generation and checkpointing after each epoch. Returns the
    target epoch count (so the caller can tell a --max-rows early stop from a
    completed generation)."""
    window = lifecycle.window_dirs(paths, state.generation_index, args.window)
    ds = SlogDataset(
        window, post_move=True, apply_symmetry=True, num_workers=cpu.dataloader_workers
    )
    epochs = epochs_this_generation(args, ds)
    avg_eligible = ds.num_samples / max(ds.num_games, 1)
    reuse = effective_reuse(epochs, args.window, args.turns_per_game, avg_eligible)
    timed_print(
        f"generation {state.generation_index}: window {[d.name for d in window]} "
        f"({ds.num_games} games, {ds.num_samples} rows, {avg_eligible:.1f} eligible turns/game); "
        f"{epochs} epochs/gen -> ~{reuse:.2f} passes/position (turns/game {args.turns_per_game})"
    )
    while state.epoch_in_generation < epochs and _rows_left(args, state):
        e = state.epoch_in_generation
        # The global epoch index (monotonic across the whole run) seeds the epoch
        # shuffle and the per-game turn rotation, so every pass shuffles differently
        # and draws distinct turns. checkpoint_index counts completed epochs, so it
        # is exactly that index and survives restarts.
        global_epoch = state.checkpoint_index
        batches = ds.iter_batches(
            args.batch_size,
            seed=global_epoch * 1000003,
            turns_per_game=args.turns_per_game,
            epoch_index=global_epoch,
        )
        t0 = time.time()
        result = run_epoch(
            model,
            optimizer,
            batches,
            device,
            loss_cfg,
            lr_fn=lr_controller.epoch_lr_fn(state.rows_trained),
            rows_trained=state.rows_trained,
            on_batch=functools.partial(progress_line, state.generation_index, e),
        )
        state.rows_trained = result.rows_trained
        state.epoch_in_generation = e + 1
        state.checkpoint_index += 1
        _checkpoint_and_eval(
            model, optimizer, conn, paths, device, args, state, result, time.time() - t0, ctx
        )
    return epochs


def run_generational_training(model, optimizer, conn, paths, device, args, state, ctx):
    """The generate->train->advance loop, from the resumed cursor onward."""
    loss_cfg = LossConfig.from_args(args)
    lr_controller = LrController(conn, args)
    cpu = CpuController(conn, args)
    generate_fn = make_generate_fn(args, ctx["player_spec"], cpu)
    try:
        while _rows_left(args, state):
            cpu.refresh(state.rows_trained)
            generation.ensure_generation(
                paths,
                state.generation_index,
                target_games=args.games_per_generation,
                seed=gen_seed(args, state.generation_index),
                player_spec=ctx["player_spec"],
                generate_fn=generate_fn,
                count_fn=count_games,
            )
            epochs = train_one_generation(
                model,
                optimizer,
                conn,
                paths,
                device,
                args,
                state,
                loss_cfg,
                lr_controller,
                cpu,
                ctx,
            )
            if state.epoch_in_generation < epochs:
                break  # stopped mid-generation by --max-rows; resume here next run
            evicted = lifecycle.evict_beyond_window(paths, state.generation_index, args.window)
            if evicted:
                timed_print(f"evicted generations {evicted} (window={args.window})")
            state.generation_index += 1
            state.epoch_in_generation = 0
            checkpoint.save(paths, model, optimizer, state, args)
    except KeyboardInterrupt:
        timed_print("Interrupted; last completed epoch is checkpointed.")
    timed_print(f"Stopped at {state.rows_trained} rows (generation {state.generation_index}).")


# ---------------------------------------------------------------------------
# Setup / main
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
    """Run the model over the frozen post-move-value set and store this checkpoint's
    per-position predictions (WLD probabilities + score-delta mean/std)."""
    model.eval()
    preds = post_move_analysis.predict(
        model, post_move_eval["inputs"], post_move_eval["spatial_planes"], device
    )
    db.write_post_move_preds(conn, generation, positions, preds)


def load_post_move_quality(args, spatial_planes: int) -> dict | None:
    """Build the large-dataset quality-eval batch and its Monte-Carlo ground truth once
    (or None if disabled / the dataset or its ground truth is unavailable). At each
    checkpoint the model is run over it and aggregate quality scalars are recorded for
    the Loss tab."""
    if args.no_post_move_quality:
        return None
    try:
        names, inputs = post_move_analysis.load_inputs(args.post_move_quality_dataset)
        gt = post_move_analysis.load_ground_truth(args.post_move_quality_dataset, names)
    except Exception as e:  # missing lexicon / dataset / ground truth
        timed_print(f"post-move-value quality eval disabled: {e}")
        return None
    if not names:
        timed_print(
            f"post-move-value quality eval disabled: no positions in "
            f"{args.post_move_quality_dataset}"
        )
        return None
    timed_print(
        f"post-move-value quality eval: {len(names)} positions from "
        f"{args.post_move_quality_dataset}"
    )
    return {"inputs": inputs, "spatial_planes": spatial_planes, "gt": gt}


def eval_post_move_quality(model, quality_eval: dict, device) -> dict:
    """Run the model over the large quality set and return the aggregate
    model-vs-Monte-Carlo quality scalars (win-equity/WLD + score-delta mean/std MAE)."""
    model.eval()
    preds = post_move_analysis.predict(
        model, quality_eval["inputs"], quality_eval["spatial_planes"], device
    )
    return post_move_analysis.quality_metrics(preds, quality_eval["gt"])


def main() -> int:
    args = build_arg_parser().parse_args()
    # Pick the experiment arm before any engine call: it is baked into the
    # process-wide FFI session, whose reported input shapes -- and therefore the
    # model, the ONNX export, and the eval batches -- all follow it.
    set_contingent_features(args.contingent_features)

    paths = TagPaths(args.tag, POST_MOVE_VALUE)
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Tag root: {paths.root}\nDevice: {device}")

    if args.restart:
        restart_run(paths)

    in_shapes = {s.name: s.dims for s in get_input_shapes()}
    spatial_planes = in_shapes["input_spatial"][0]
    scalar_size = in_shapes["input_scalar"][0]
    lex = LexiconArgs.from_args(args)
    lexicon_module = lex.build(channels=args.trunk_channels)
    if lexicon_module is not None:
        print(f"Lexicon module: {lex.module}")
    model = PostMoveValueModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        num_blocks=args.num_blocks,
        trunk_channels=args.trunk_channels,
        lexicon_module=lexicon_module,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, args.tag, vars(args), n_params)
    db.write_loss_weights(
        conn,
        {
            "loss_wld": 1.0,
            "loss_score_diff": args.lambda_sd,
            "loss_opp_next_placement": args.lambda_opp,
        },
    )
    init_controls(conn, args)

    ctx = {
        "player_spec": build_player_spec(args),
        "spatial_planes": spatial_planes,
        "scalar_size": scalar_size,
        "post_move_eval": load_post_move_eval(args, spatial_planes),
        "post_move_quality": load_post_move_quality(args, spatial_planes),
    }

    if not args.no_dashboard:
        for proc in react_server.spawn(
            "post_move_value", str(paths.mount_root), dev_port=args.dashboard_port, tag=args.tag
        ):
            atexit.register(proc.terminate)

    state = checkpoint.resume(paths, model, optimizer, device)
    run_generational_training(model, optimizer, conn, paths, device, args, state, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Trainer for the max-move-per-lane model: an open-ended generate->train loop.

Sibling to the post-move value trainer (scripts/post_move_value/train.py). It runs
the same generational lifecycle -- generate a generation of HastyBot self-play
games to disk, train a few epochs over a sliding window of the most recent
generations (reusing each position, and with --turns-per-game sampling several
turns per game), then advance and evict -- but for the "highest-scoring move per
lane" representation probe. The per-lane labels are recomputed by replaying each
.slog game and enumerating legal moves at the position, so every turn of a game
(including endgame turns) is a training row.

Everything is keyed on cumulative rows trained (the rows-clock): the dashboard
x-axis, the warmup learning rate, and the restart cursor. A single rolling
model.pt holds resume state, so stopping and restarting the script continues
exactly where it left off. The base learning rate and the CPU-thread pools
(game-generation threads, DataLoader workers, torch intra-op threads) are live
controls in the per-tag dashboard.db, retuned from the dashboard Controls tab.

There is no held-out probe/calibration eval or ONNX export -- this is a
representation-learning probe. Eval is the per-lane train accuracy (does the model
get each legal lane's best move and its score right?), recorded alongside the
losses, plus a per-checkpoint lane-analysis pass over a frozen GCG dataset for the
dashboard's Lane-analysis tab.

Usage:
    python -m scripts.max_move_per_lane.train -t mytag \
        --games-per-generation 20000 --reuse-per-position 2 --window 4
"""

import argparse
import atexit
import functools
import shutil
import sys
import time

import torch
from scribblez import lane_analysis
from scribblez.dashboard import db, react_server
from scribblez.dataset import SlogDataset
from scribblez.ffi import get_max_move_per_lane_input_shapes, read_file_header
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
from scribblez.max_move_per_lane.model import MaxMovePerLaneModel
from scribblez.max_move_per_lane.train_loop import LossConfig, run_epoch
from scribblez.paths import MAX_MOVE_PER_LANE, TagPaths
from scribblez.train_common import reset_tag, timed_print
from scripts.generate_data import run_games


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generational generate->train loop for the max-move-per-lane model.",
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
        help="Target gradient passes per unique position over a generation's lifetime in "
        "the window. Epochs-per-generation is derived from this and the generation's "
        "measured average turns/game (num_samples/num_games).",
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
        "turn; K>1 = K distinct turns/game/epoch.",
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

    # Model architecture.
    p.add_argument("--trunk-channels", type=int, default=128, help="CNN trunk width.")
    p.add_argument("--num-blocks", type=int, default=8, help="Trunk residual blocks.")
    p.add_argument("--lane-layers", type=int, default=4, help="Lane transformer layers.")
    p.add_argument("--lane-heads", type=int, default=4, help="Lane transformer attention heads.")
    p.add_argument("--ffn-mult", type=int, default=4, help="Lane transformer FFN width multiple.")
    p.add_argument("--rack-tokens", type=int, default=4, help="Rack tokens prepended per lane.")
    LexiconArgs.add_arguments(p)

    # Loss weights.
    p.add_argument("--lambda-cdf", type=float, default=1.0, help="Score-CDF (CRPS) loss weight.")
    p.add_argument("--lambda-occ", type=float, default=100.0, help="Occupancy (move) loss weight.")
    p.add_argument("--lambda-has-move", type=float, default=1.0, help="Has-move loss weight.")
    p.add_argument(
        "--max-rows", type=int, default=0, help="Stop after this many rows trained (0=forever)."
    )

    # Per-checkpoint lane-analysis evaluation over a frozen GCG dataset; the
    # Lane-analysis tab pairs the model's per-lane predictions with the ground truth.
    p.add_argument(
        "--lane-eval-dataset",
        type=str,
        default=str(lane_analysis.DEFAULT_DATASET),
        help="GCG dataset directory the lane-analysis tab evaluates each checkpoint over.",
    )
    p.add_argument(
        "--no-lane-eval",
        action="store_true",
        help="Disable the per-checkpoint lane-analysis evaluation.",
    )

    p.add_argument("--restart", action="store_true", help="Wipe checkpoints/db/generations.")
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
    """Wipe prior run artifacts (checkpoints, db) and the train generations for a
    fresh start."""
    reset_tag(paths)
    shutil.rmtree(paths.generations_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _rows_left(args, state: GenerationalState) -> bool:
    return args.max_rows == 0 or state.rows_trained < args.max_rows


def _checkpoint_and_eval(model, optimizer, conn, paths, device, args, state, result, elapsed, ctx):
    """Record this epoch's metrics + lane accuracies (keyed on the checkpoint
    index, with the rows-clock stored as `positions`), run the lane-analysis eval,
    and save the rolling checkpoint."""
    sys.stdout.write("\n")
    avg = result.losses
    ci = state.checkpoint_index
    base_lr = db.read_control(conn, CONTROL_BASE_LR, default=args.lr)
    lr_now = effective_lr(base_lr, state.rows_trained, args.warmup_rows)
    timed_print(
        f"[gen {state.generation_index} epoch {state.epoch_in_generation}] ckpt {ci} "
        f"rows={state.rows_trained} loss={avg['total']:.4f} move_acc={result.accs['move_acc']:.4f} "
        f"score_acc={result.accs['score_acc']:.4f} lr={lr_now:.2e} {elapsed:.1f}s"
    )
    record = {
        "epoch": ci,
        "positions": state.rows_trained,
        "generation": state.generation_index,
        "loss": avg["total"],
        "loss_score_pdf": avg["score_pdf"],
        "loss_score_cdf": avg["score_cdf"],
        "loss_move": avg["move"],
        "loss_has_move": avg["has_move"],
        **result.accs,
        "lr": lr_now,
        "elapsed_s": elapsed,
    }
    db.write_metrics(conn, ci, record)
    if ctx["lane_eval"] is not None:
        eval_lane_analysis(model, ctx["lane_eval"], device, conn, ci, state.rows_trained)
    checkpoint.save(paths, model, optimizer, state, args)


def epochs_this_generation(args, ds) -> int:
    """The epoch count to run over the current window: --epochs-per-generation when
    set explicitly, otherwise derived from --reuse-per-position and the window's
    measured average turns per game."""
    if args.epochs_per_generation > 0:
        return args.epochs_per_generation
    avg_turns = ds.num_samples / max(ds.num_games, 1)
    return epochs_for_reuse(args.reuse_per_position, args.window, args.turns_per_game, avg_turns)


def train_one_generation(
    model, optimizer, conn, paths, device, args, state, loss_cfg, lr_controller, cpu, ctx
) -> int:
    """Train `epochs` passes over the current window, resuming at
    state.epoch_in_generation and checkpointing after each epoch. Returns the
    target epoch count (so the caller can tell a --max-rows early stop from a
    completed generation)."""
    window = lifecycle.window_dirs(paths, state.generation_index, args.window)
    ds = SlogDataset(
        window, task="max_move_per_lane", apply_symmetry=True, num_workers=cpu.dataloader_workers
    )
    epochs = epochs_this_generation(args, ds)
    avg_turns = ds.num_samples / max(ds.num_games, 1)
    reuse = effective_reuse(epochs, args.window, args.turns_per_game, avg_turns)
    timed_print(
        f"generation {state.generation_index}: window {[d.name for d in window]} "
        f"({ds.num_games} games, {ds.num_samples} rows, {avg_turns:.1f} turns/game); "
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


def load_lane_eval(args, spatial_planes: int) -> dict | None:
    """Build the frozen lane-analysis input batch once (or None if disabled / the
    dataset is empty / the lexicon is unavailable). At each checkpoint the model is
    run over it and the predictions are written to the dashboard DB."""
    if args.no_lane_eval:
        return None
    try:
        names, inputs = lane_analysis.load_inputs(args.lane_eval_dataset)
    except Exception as e:  # missing lexicon / unreadable dataset
        timed_print(f"lane-analysis eval disabled: {e}")
        return None
    if not names:
        timed_print(f"lane-analysis eval disabled: no GCG positions in {args.lane_eval_dataset}")
        return None
    timed_print(f"lane-analysis eval: {len(names)} positions from {args.lane_eval_dataset}")
    return {"inputs": inputs, "spatial_planes": spatial_planes}


def eval_lane_analysis(model, lane_eval: dict, device, conn, checkpoint_index: int, positions: int):
    """Run the model over the frozen lane-analysis set and store this checkpoint's
    per-(position, lane) predictions for the dashboard."""
    model.eval()
    preds = lane_analysis.predict(model, lane_eval["inputs"], lane_eval["spatial_planes"], device)
    db.write_lane_preds(conn, checkpoint_index, positions, preds)


def main() -> int:
    args = build_arg_parser().parse_args()
    paths = TagPaths(args.tag, MAX_MOVE_PER_LANE)
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Tag root: {paths.root}\nDevice: {device}")

    if args.restart:
        restart_run(paths)

    in_shapes = {s.name: s.dims for s in get_max_move_per_lane_input_shapes()}
    spatial_planes = in_shapes["input_spatial"][0]
    scalar_size = in_shapes["input_scalar"][0]
    lex = LexiconArgs.from_args(args)
    lexicon_module = lex.build(channels=args.trunk_channels)
    if lexicon_module is not None:
        print(f"Lexicon module: {lex.module}")
    lane_ffn_mult = lex.lane_ffn_mult(lexicon_module is not None)
    if lane_ffn_mult is not None:
        print(f"Lane FFN width multiple shrunk to {lane_ffn_mult} (replace mode).")
    model = MaxMovePerLaneModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        trunk_channels=args.trunk_channels,
        num_blocks=args.num_blocks,
        lane_layers=args.lane_layers,
        lane_heads=args.lane_heads,
        ffn_mult=args.ffn_mult,
        n_rack_tokens=args.rack_tokens,
        lexicon_module=lexicon_module,
        lane_ffn_mult=lane_ffn_mult,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    conn = db.connect(paths.dashboard_db)
    db.write_meta(conn, args.tag, vars(args), n_params)
    # Coefficients of each loss term in compute_loss's total (PDF has weight 1),
    # so the dashboard can stack the weighted contributions.
    db.write_loss_weights(
        conn,
        {
            "loss_score_pdf": 1.0,
            "loss_score_cdf": args.lambda_cdf,
            "loss_move": args.lambda_occ,
            "loss_has_move": args.lambda_has_move,
        },
    )
    init_controls(conn, args)

    ctx = {
        "player_spec": build_player_spec(args),
        "spatial_planes": spatial_planes,
        "scalar_size": scalar_size,
        "lane_eval": load_lane_eval(args, spatial_planes),
    }

    if not args.no_dashboard:
        # The React dashboard (Training + Lane analysis); it embeds the Bokeh
        # metrics plots and adds the interactive lane-analysis board.
        for proc in react_server.spawn(
            "max_move_per_lane", str(paths.mount_root), dev_port=args.dashboard_port, tag=args.tag
        ):
            atexit.register(proc.terminate)

    state = checkpoint.resume(paths, model, optimizer, device)
    run_generational_training(model, optimizer, conn, paths, device, args, state, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())

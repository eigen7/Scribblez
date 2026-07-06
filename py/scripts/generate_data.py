#!/usr/bin/env python3
"""Generate training data by running self-play games.

Usage:
    # HastyBot self-play (iteration 0):
    python -m scripts.generate_data -t mytag -g 100000 --test-ratio 0.1

    # Policy-iteration self-play with the neural value agent (temperature adds
    # the exploration that pure argmax self-play lacks). --top-k 0 evaluates
    # every legal play (most diverse); K > 0 keeps the top-K by HastyBot equity:
    python -m scripts.generate_data -t mytag_iter1 -g 100000 \
        --model /path/to/model.onnx --top-k 10 --temperature 3.0

This shells out to the C++ `play_game` binary with --binary-log-dir pointed at
the tag's train/ and test/ data directories. The split is partitioned at the
file level by running two independent game batches, so a game never straddles
the train/test boundary. The test set is the frozen held-out split used for
calibration and the monotonicity probe bank; it must never be trained on.

Requires Macondo to be built.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scribblez.ffi import read_file_header
from scribblez.paths import POST_MOVE_VALUE, TagPaths
from util.argparse_ext import ArgumentDefaultsHelpFormatter

PLAY_GAME = "/workspace/repo/target/engine/play_game"


def build_player_spec(args) -> str:
    """The `--player` value for both seats. With no --model: HastyBot, optionally
    temperature-sampled over equity (--hasty-temperature > 0) for exploration.
    With a --model: the neural value agent -- --top-k=0 evaluates every legal
    play (most diverse), K > 0 keeps the top-K by HastyBot equity (faster)."""
    if not args.model:
        if args.hasty_temperature > 0:
            return (
                f"--type=hastybot --temperature={args.hasty_temperature} "
                f"--top-k={args.hasty_top_k} "
                f"--temperature-min-bag={args.hasty_temp_min_bag}"
            )
        return "--type=hastybot"
    return (
        f"--type=neural --model={args.model} --top-k={args.top_k} "
        f"--temperature={args.temperature} --precision={args.precision}"
    )


def run_games(
    out_dir: Path,
    num_games: int,
    games_per_file: int,
    threads: int,
    player_spec: str,
    seed: int = 0,
    random_opening_mean: float = 0.0,
) -> int:
    """Run `num_games` self-play games, logging .slog files to out_dir.

    Both seats use `player_spec` (the value of a `--player` flag); each seat is a
    fresh agent, so two neural seats draw independent sampling seeds. `seed` is
    the PRNG seed passed to play_game; 0 (the default) lets the binary pick one
    from std::random_device, so successive default-seeded runs differ.
    `random_opening_mean` > 0 opens each game with K uniformly-random plies
    (K ~ round(Exp(mean)) per game) before the agents take over; positions
    before the last random ply are excluded from the training-eligible region.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # fmt: off
    cmd = [
        PLAY_GAME,
        "--player", player_spec,
        "--player", player_spec,
        "--binary-log-dir", str(out_dir),
        "--games-per-file", str(games_per_file),
        "--games", str(num_games),
        "--threads", str(threads),
        "--seed", str(seed),
        "--random-opening-mean", str(random_opening_mean),
    ]
    # fmt: on
    cmd_str = " ".join(f'"{t}"' if " " in t else t for t in cmd)
    print(f"Running: {cmd_str}")
    print(f"Output:  {out_dir}")
    return subprocess.run(cmd, capture_output=False).returncode


def count_positions(out_dir: Path) -> int:
    """Sum the per-file position counts from .slog headers; -1 on failure."""
    try:
        total = 0
        for f in sorted(out_dir.glob("*.slog")):
            num_pos, _ = read_file_header(f)
            total += num_pos
        return total
    except Exception as e:  # noqa: BLE001 -- validation is best-effort
        print(f"Warning: could not validate headers in {out_dir}: {e}", file=sys.stderr)
        return -1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate HastyBot self-play data with a train/test split.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="Tag (per-tag artifact root).")
    parser.add_argument("-g", "--num-games", type=int, default=100000, help="Total games.")
    parser.add_argument("-T", "--threads", type=int, default=8, help="Parallel game threads.")
    parser.add_argument(
        "-p", "--games-per-file", type=int, default=10000, help="Games per .slog file."
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of games routed to the held-out test split.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="ONNX model for neural self-play; empty = HastyBot self-play.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Neural candidate set (with --model): 0 = every legal play (most "
        "diverse, slowest); K > 0 = top-K by HastyBot equity (faster).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Neural agent softmax sampling temperature (only used with --model).",
    )
    parser.add_argument(
        "--hasty-temperature",
        type=float,
        default=0.0,
        help="HastyBot softmax temperature for model-free self-play (0 = greedy).",
    )
    parser.add_argument(
        "--hasty-top-k",
        type=int,
        default=10,
        help="HastyBot candidate count when --hasty-temperature > 0.",
    )
    parser.add_argument(
        "--hasty-temp-min-bag",
        type=int,
        default=0,
        help="Confine HastyBot softmax sampling to turns where the bag has >= this "
        "many tiles (greedy below it). 0 = sample the whole game; ~60 reproduces "
        "an opening-only exploratory bot.",
    )
    parser.add_argument(
        "--random-opening-mean",
        type=float,
        default=0.0,
        help="If > 0, open each game with K uniformly-random plies (K ~ round(Exp(mean)) "
        "per game) before the agents take over. Positions before the last random ply "
        "are excluded from the training-eligible region.",
    )
    parser.add_argument("--precision", default="FP16", help="Neural agent TensorRT precision.")
    args = parser.parse_args()

    if not 0.0 <= args.test_ratio < 1.0:
        print("--test-ratio must be in [0, 1).", file=sys.stderr)
        return 2

    paths = TagPaths(args.tag, POST_MOVE_VALUE)
    player_spec = build_player_spec(args)
    test_games = round(args.num_games * args.test_ratio)
    train_games = args.num_games - test_games

    t0 = time.time()
    splits = [("train", paths.train_dir, train_games), ("test", paths.test_dir, test_games)]
    for name, out_dir, n in splits:
        if n <= 0:
            continue
        print(f"\n=== Generating {n} {name} games ===")
        rc = run_games(
            out_dir,
            n,
            args.games_per_file,
            args.threads,
            player_spec,
            random_opening_mean=args.random_opening_mean,
        )
        if rc != 0:
            print(f"play_game exited with code {rc} for {name} split", file=sys.stderr)
            return rc

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    for name, out_dir, n in splits:
        if n <= 0:
            continue
        files = list(out_dir.glob("*.slog"))
        print(f"  {name}: {len(files)} files, {count_positions(out_dir)} games -> {out_dir}")
    print(f"  Rate:  {args.num_games / elapsed:.0f} games/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

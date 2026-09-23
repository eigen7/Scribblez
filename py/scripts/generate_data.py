#!/usr/bin/env python3
"""Generate a one-shot self-play corpus with a train/test split, without the dashboard.

Runs the C++ play_game binary, logging .slog files into a position_eval tag's
data/train/ and data/test/ dirs. The two splits are separate game batches, so
no game straddles them. Nothing in the pipeline reads these dirs; the
position_eval workload generates its corpus through the dashboard's generation
scheduler instead. This is for ad-hoc corpora.

Both seats play HastyBot by default, which reads its leave values from the
Macondo checkout py/build.py clones. With --model they play the neural value
agent instead; --temperature adds the exploration that argmax self-play lacks.

Usage:
    ./py/scripts/generate_data.py -t mytag -g 100000 --test-ratio 0.1
    ./py/scripts/generate_data.py -t mytag_iter1 -g 100000 \
        --model /path/to/model.onnx --top-k 10 --temperature 3.0
"""

import argparse
import sys
import time
from pathlib import Path

from scribblez.ffi import read_file_header
from scribblez.hardware import default_thread_count
from scribblez.paths import POSITION_EVAL, TagPaths
from scribblez.selfplay import hasty_player_spec, run_games
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def build_player_spec(args) -> str:
    """The `--player` value for both seats: HastyBot with no --model, else the
    neural value agent, whose endgames go to the exact solver at the engine's
    default node budget."""
    if not args.model:
        return hasty_player_spec(args.hasty_temperature, args.hasty_top_k)
    return (
        f"--type=neural --model={args.model} --top-k={args.top_k} "
        f"--temperature={args.temperature} --precision={args.precision}"
    )


def count_positions(out_dir: Path) -> int:
    """Total positions across the .slog headers in out_dir; -1 on failure."""
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
        description="Generate a self-play corpus with a train/test split.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="position_eval tag to write under.")
    parser.add_argument("-g", "--num-games", type=int, default=100000, help="Total games.")
    parser.add_argument(
        "-T",
        "--threads",
        type=int,
        default=default_thread_count(),
        help="Parallel game threads (default: all logical processors).",
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

    paths = TagPaths(args.tag, POSITION_EVAL)
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

#!/usr/bin/env python3
"""Generate training data by running HastyBot-vs-HastyBot games.

Usage:
    python -m scripts.generate_data -t mytag -g 100000 --test-ratio 0.1

This shells out to the C++ `play_game` binary with --binary-log-dir pointed at
the tag's train/ and test/ data directories. The split is partitioned at the
file level by running two independent game batches, so a game never straddles
the train/test boundary. The test set is the frozen held-out split used for
calibration and the monotonicity probe bank; it must never be trained on.

Requires Macondo to be built.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scribblez.paths import TagPaths

PLAY_GAME = '/workspace/repo/target/engine/play_game'


def run_games(out_dir: Path, num_games: int, games_per_file: int, threads: int) -> int:
    """Run `num_games` HastyBot self-play games, logging .slog files to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PLAY_GAME,
        "--player", "--type=hastybot",
        "--player", "--type=hastybot",
        "--binary-log-dir", str(out_dir),
        "--games-per-file", str(games_per_file),
        "--games", str(num_games),
        "--threads", str(threads),
        "--random-handicap-max", "100",
    ]
    cmd_str = " ".join(f'"{t}"' if " " in t else t for t in cmd)
    print(f"Running: {cmd_str}")
    print(f"Output:  {out_dir}")
    return subprocess.run(cmd, capture_output=False).returncode


def count_positions(out_dir: Path) -> int:
    """Sum the per-file position counts from .slog headers; -1 on failure."""
    try:
        from scribblez.ffi import read_file_header

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    args = parser.parse_args()

    if not 0.0 <= args.test_ratio < 1.0:
        print("--test-ratio must be in [0, 1).", file=sys.stderr)
        return 2

    paths = TagPaths(args.tag)
    test_games = round(args.num_games * args.test_ratio)
    train_games = args.num_games - test_games

    t0 = time.time()
    splits = [("train", paths.train_dir, train_games), ("test", paths.test_dir, test_games)]
    for name, out_dir, n in splits:
        if n <= 0:
            continue
        print(f"\n=== Generating {n} {name} games ===")
        rc = run_games(out_dir, n, args.games_per_file, args.threads)
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

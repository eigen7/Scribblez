#!/usr/bin/env python3
"""Generate training data by running HastyBot-vs-HastyBot games.

Usage:
    python -m scripts.generate_data -t mytag -n 1000

This shells out to the C++ `play_game` binary with --binary-log-dir pointed at
/workspace/mount/data/<tag>/. Requires Macondo to be built.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PLAY_GAME = '/workspace/repo/target/engine/play_game'


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate HastyBot self-play data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--tag", required=True, help="Data subdirectory tag.")
    parser.add_argument("-g", "--num-games", type=int, default=100000, help="Number of games.")
    parser.add_argument("-T", "--threads", type=int, default=8, help="Parallel game threads.")
    parser.add_argument(
        "-p", "--games-per-file", type=int, default=10000, help="Games per .slog file."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/workspace/mount/data",
        help="Root directory for data output.",
    )
    args = parser.parse_args()

    play_game = PLAY_GAME

    out_dir = Path(args.data_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        play_game,
        "--player", "--type=hastybot",
        "--player", "--type=hastybot",
        "--binary-log-dir", str(out_dir),
        "--games-per-file", str(args.games_per_file),
        "--games", str(args.num_games),
        "--threads", str(args.threads),
        "--random-handicap-max", "100",
    ]

    cmd_str = " ".join(f'"{t}"' if " " in t else t for t in cmd)
    print(f"Running: {cmd_str}")
    print(f"Output:  {out_dir}")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"play_game exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    elapsed = time.time() - t0
    slog_files = list(out_dir.glob("*.slog"))

    # Validate headers via FFI.
    total_games = 0
    try:
        from scribblez.ffi import read_file_header

        for f in sorted(slog_files):
            num_pos, _ = read_file_header(f)
            total_games += num_pos
    except Exception as e:
        print(f"Warning: could not validate headers: {e}", file=sys.stderr)
        total_games = -1

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Files:  {len(slog_files)}")
    print(f"  Games:  {total_games}")
    print(f"  Rate:   {args.num_games / elapsed:.0f} games/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

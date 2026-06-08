#!/usr/bin/env python3
"""Generate training data by running HastyBot-vs-HastyBot games.

Usage:
    python -m scripts.generate_data -t mytag -n 1000

This shells out to the C++ `play_game` binary with --binary-log-dir pointed at
/workspace/mount/data/<tag>/. Requires Macondo to be built.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Repo root (two levels up from python/scripts/).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def find_play_game() -> Path:
    """Locate the play_game binary."""
    candidates = [
        REPO_ROOT / "build" / "engine" / "play_game",
        Path(os.environ.get("PLAY_GAME_BIN", "")),
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        "Cannot find play_game binary. Build the project first:\n"
        "  python build.py"
    )


def find_macondo() -> Path:
    """Locate the Macondo binary."""
    candidates = [
        REPO_ROOT / "build" / "macondo" / "macondo",
        Path("/workspace/mount/macondo/bin/macondo"),
        Path(os.environ.get("MACONDO_BIN", "")),
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        "Cannot find macondo binary. Build it first:\n"
        "  python build.py --skip-web\n"
        "Or set MACONDO_BIN=/path/to/macondo"
    )


def find_lexicon() -> Path:
    """Locate the NWL23.kwg lexicon file."""
    candidates = [
        REPO_ROOT / "data" / "lexica" / "NWL23.kwg",
        Path("/workspace/mount/lexica/NWL23.kwg"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "Cannot find NWL23.kwg lexicon. Download it or symlink into data/lexica/."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HastyBot self-play data.")
    parser.add_argument("-t", "--tag", required=True, help="Data subdirectory tag.")
    parser.add_argument("-n", "--num-games", type=int, default=1000, help="Number of games.")
    parser.add_argument("--threads", type=int, default=4, help="Parallel game threads.")
    parser.add_argument(
        "--games-per-file", type=int, default=100, help="Games per .slog file."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/workspace/mount/data",
        help="Root directory for data output.",
    )
    args = parser.parse_args()

    play_game = find_play_game()
    macondo = find_macondo()
    lexicon = find_lexicon()

    out_dir = Path(args.data_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(play_game),
        "--player", "--type=hasty",
        "--player", "--type=hasty",
        "--lexicon", str(lexicon),
        "--macondo-binary", str(macondo),
        "--binary-log-dir", str(out_dir),
        "--games-per-file", str(args.games_per_file),
        "--games", str(args.num_games),
        "--threads", str(args.threads),
    ]

    print(f"Running: {' '.join(cmd)}")
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

#!/usr/bin/env python3
"""Data generator for the sim-evidence kill-test (docs/sim_residual_feedback.md).

Alternates between two phases until stopped (Ctrl-C), accumulating under
<mount>/kill_test/<tag>/slogs:

  1. HastyBot self-play: one batch of games into a fresh .slog file
     (timestamp-named, so batches from any number of runs coexist).
  2. sim_obs_tool over the directory: every .slog without a .sobs sidecar gets
     one (HastyBot-equity top-K candidates, SimRunner rollouts). Files that
     already have their sidecar are skipped, and sidecars appear atomically,
     so the loop -- and the whole script -- can be stopped and restarted
     arbitrarily; a restart resumes exactly where generation left off.

Run the 4-armed experiment on the accumulated data with
scripts/kill_test.py -t <tag> (which may run while this keeps generating; it
snapshots whatever complete .slog/.sobs pairs exist).

Usage:
    ./py/scripts/generate_kill_test_data.py -t apple
"""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.generate_data import run_games
from util.argparse_ext import ArgumentDefaultsHelpFormatter

SIM_OBS_TOOL = "/workspace/repo/target/engine/sim_obs_tool"
MOUNT_ROOT = Path("/workspace/mount")


def slog_dir(tag: str) -> Path:
    return MOUNT_ROOT / "kill_test" / tag / "slogs"


def run_sim_obs_tool(out_dir: Path, args) -> int:
    cmd = [
        SIM_OBS_TOOL,
        f"--slog-dir={out_dir}",
        f"--rollouts={args.rollouts}",
        f"--top-k={args.top_k}",
        f"--positions-per-game={args.positions_per_game}",
        f"--threads={args.threads}",
    ]
    return subprocess.run(cmd, capture_output=False).returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-t",
        "--tag",
        required=True,
        help="run tag; data accumulates under <mount>/kill_test/<tag>/slogs",
    )
    p.add_argument("--threads", type=int, default=8, help="num c++ threads")
    p.add_argument(
        "--games-per-batch",
        type=int,
        default=200,
        help="self-play games per .slog file / generation cycle",
    )
    p.add_argument("--rollouts", type=int, default=200, help="sim rollouts per candidate")
    p.add_argument("--top-k", type=int, default=10, help="candidates simmed per position")
    p.add_argument("--positions-per-game", type=int, default=1)
    args = p.parse_args()

    out_dir = slog_dir(args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Accumulating kill-test data under {out_dir} (Ctrl-C to stop)")

    batch = 0
    try:
        while True:
            batch += 1
            existing = len(list(out_dir.glob("*.sobs")))
            print(f"\n=== cycle {batch} ({existing} .sobs files so far) ===")
            rc = run_games(
                out_dir,
                num_games=args.games_per_batch,
                games_per_file=args.games_per_batch,
                threads=args.threads,
                player_spec="--type=hastybot",
            )
            if rc != 0:
                print(f"play_game exited with code {rc}; stopping", file=sys.stderr)
                return rc
            rc = run_sim_obs_tool(out_dir, args)
            if rc != 0:
                print(f"sim_obs_tool exited with code {rc}; stopping", file=sys.stderr)
                return rc
    except KeyboardInterrupt:
        # An interrupt loses at most the in-flight cycle's unfinished work:
        # .slog batches and .sobs sidecars both land atomically, and the next
        # run re-derives any missing sidecar from its .slog.
        print("\nstopped; data is consistent and generation resumes on rerun")
    return 0


if __name__ == "__main__":
    sys.exit(main())

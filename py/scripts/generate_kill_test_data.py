#!/usr/bin/env python3
"""Data generator for the sim-evidence kill-test (docs/sim_residual_feedback.md).

Alternates between two phases until stopped (Ctrl-C), accumulating under
<mount>/kill_test/<tag>/slogs:

  1. HastyBot self-play: one batch of games into a fresh .slog file
     (timestamp-named, so batches from any number of runs coexist).
  2. sim_obs_tool over exactly the .slog files still missing a .sobs sidecar
     (normally just the fresh batch; after an interrupted run, also the
     backlog). Sidecars appear atomically, so the loop -- and the whole
     script -- can be stopped and restarted arbitrarily; a restart resumes
     exactly where generation left off.

The per-cycle logic (KillTestParams / run_one_cycle) is also driven by the
cloud worker entrypoint (py/cloud/worker_entrypoint.py), which runs the same
cycle on rented machines and uploads each completed pair to the results
bucket.

Run the 4-armed experiment on the accumulated data with
scripts/kill_test.py -t <tag> (which may run while this keeps generating; it
snapshots whatever complete .slog/.sobs pairs exist).

Usage:
    ./py/scripts/generate_kill_test_data.py -t apple
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scribblez.hardware import default_thread_count
from scripts.generate_data import run_games
from util.argparse_ext import ArgumentDefaultsHelpFormatter

SIM_OBS_TOOL = "/workspace/repo/target/engine/sim_obs_tool"
MOUNT_ROOT = Path("/workspace/mount")


@dataclass(frozen=True)
class KillTestParams:
    """Knobs for one generation cycle (a self-play batch plus its sidecars)."""

    threads: int
    games_per_batch: int = 200
    rollouts: int = 200
    top_k: int = 10
    positions_per_game: int = 1
    open_leaves: bool = False


def slog_dir(tag: str) -> Path:
    return MOUNT_ROOT / "kill_test" / tag / "slogs"


def run_sim_obs_tool(pending: list[Path], params: KillTestParams) -> int:
    cmd = [
        SIM_OBS_TOOL,
        *[f"--slog-file={p}" for p in pending],
        f"--rollouts={params.rollouts}",
        f"--top-k={params.top_k}",
        f"--positions-per-game={params.positions_per_game}",
        f"--threads={params.threads}",
    ]
    if params.open_leaves:
        cmd.append("--open-leaves")
    return subprocess.run(cmd, capture_output=False).returncode


def run_one_cycle(out_dir: Path, params: KillTestParams) -> int:
    """One generation cycle: a self-play batch, then sidecars for every .slog
    in `out_dir` still missing one (the fresh batch, plus any backlog an
    earlier interrupted run left behind). Returns the first nonzero subprocess
    exit code, or 0."""
    rc = run_games(
        out_dir,
        num_games=params.games_per_batch,
        games_per_file=params.games_per_batch,
        threads=params.threads,
        player_spec="--type=hastybot",
    )
    if rc != 0:
        print(f"play_game exited with code {rc}", file=sys.stderr)
        return rc
    pending = sorted(s for s in out_dir.glob("*.slog") if not s.with_suffix(".sobs").exists())
    if not pending:
        return 0
    rc = run_sim_obs_tool(pending, params)
    if rc != 0:
        print(f"sim_obs_tool exited with code {rc}", file=sys.stderr)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-t",
        "--tag",
        required=True,
        help="run tag; data accumulates under <mount>/kill_test/<tag>/slogs",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=default_thread_count(),
        help="num c++ threads (default: all logical processors)",
    )
    p.add_argument(
        "--games-per-batch",
        type=int,
        default=KillTestParams.games_per_batch,
        help="self-play games per .slog file / generation cycle",
    )
    p.add_argument(
        "--rollouts", type=int, default=KillTestParams.rollouts, help="sim rollouts per candidate"
    )
    p.add_argument(
        "--top-k", type=int, default=KillTestParams.top_k, help="candidates simmed per position"
    )
    p.add_argument("--positions-per-game", type=int, default=KillTestParams.positions_per_game)
    p.add_argument(
        "--open-leaves",
        action="store_true",
        help="sim with the opponent's retained leave known, replenishments hidden (the "
        "open-leaves information condition); use a dedicated tag, and pass "
        "--open-leaves to kill_test.py as well",
    )
    args = p.parse_args()
    params = KillTestParams(
        threads=args.threads,
        games_per_batch=args.games_per_batch,
        rollouts=args.rollouts,
        top_k=args.top_k,
        positions_per_game=args.positions_per_game,
        open_leaves=args.open_leaves,
    )

    out_dir = slog_dir(args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Accumulating kill-test data under {out_dir} (Ctrl-C to stop)")

    batch = 0
    try:
        while True:
            batch += 1
            existing = len(list(out_dir.glob("*.sobs")))
            print(f"\n=== cycle {batch} ({existing} .sobs files so far) ===")
            rc = run_one_cycle(out_dir, params)
            if rc != 0:
                return rc
    except KeyboardInterrupt:
        # An interrupt loses at most the in-flight cycle's unfinished work:
        # .slog batches and .sobs sidecars both land atomically, and the next
        # run re-derives any missing sidecar from its .slog.
        print("\nstopped; data is consistent and generation resumes on rerun")
    return 0


if __name__ == "__main__":
    sys.exit(main())

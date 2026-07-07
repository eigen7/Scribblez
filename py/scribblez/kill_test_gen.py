"""Kill-test data generation: the parameter set and one generation cycle.

The library home for the sim-evidence kill-test's generation logic (see
docs/sim_residual_feedback.md), shared by every driver: the local CLI
(scripts/generate_kill_test_data.py), the worker entrypoint that runs on local
and cloud workers (py/cloud/worker_entrypoint.py), and the master dashboard.

One cycle = one HastyBot self-play batch into a fresh timestamp-named .slog
file, then sim_obs_tool over every .slog still missing its .sobs sidecar (the
fresh batch plus any backlog an interrupted run left). Both artifacts land
atomically, so cycles can be interrupted and resumed freely and any number of
workers can generate into the same tag.
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scripts.generate_data import run_games

from scribblez.params import param

SIM_OBS_TOOL = "/workspace/repo/target/engine/sim_obs_tool"
MOUNT_ROOT = Path("/workspace/mount")
DATA_ROOT = MOUNT_ROOT / "kill_test"


@dataclass(frozen=True)
class KillTestParams:
    """A tag's generation parameters. Frozen at task creation: every worker on
    a tag must generate with identical settings for the data to be analyzable
    as one corpus. Worker-level knobs (thread count) deliberately live outside.
    """

    games_per_batch: int = param(200, "self-play games per .slog file / generation cycle")
    rollouts: int = param(200, "sim rollouts per candidate")
    top_k: int = param(10, "candidates simmed per position")
    positions_per_game: int = param(1, "positions sampled per game for sim observation")
    open_leaves: bool = param(
        False,
        "sim with the opponent's retained leave known, replenishments hidden (the "
        "open-leaves information condition); use a dedicated tag, and pass "
        "--open-leaves to kill_test.py as well",
    )


@dataclass(frozen=True)
class CycleResult:
    returncode: int
    new_pairs: list[Path]  # .slog paths that gained a .sobs sidecar this cycle
    gen_seconds: float  # self-play batch wall time
    sim_seconds: float  # sim_obs_tool wall time


def data_dir(tag: str) -> Path:
    return DATA_ROOT / tag


def slog_dir(tag: str) -> Path:
    return data_dir(tag) / "slogs"


def run_sim_obs_tool(pending: list[Path], params: KillTestParams, threads: int) -> int:
    cmd = [
        SIM_OBS_TOOL,
        *[f"--slog-file={p}" for p in pending],
        f"--rollouts={params.rollouts}",
        f"--top-k={params.top_k}",
        f"--positions-per-game={params.positions_per_game}",
        f"--threads={threads}",
    ]
    if params.open_leaves:
        cmd.append("--open-leaves")
    return subprocess.run(cmd, capture_output=False).returncode


def run_one_cycle(out_dir: Path, params: KillTestParams, threads: int) -> CycleResult:
    """One generation cycle into `out_dir`, with per-phase wall times."""
    t0 = time.monotonic()
    rc = run_games(
        out_dir,
        num_games=params.games_per_batch,
        games_per_file=params.games_per_batch,
        threads=threads,
        player_spec="--type=hastybot",
    )
    gen_seconds = time.monotonic() - t0
    if rc != 0:
        print(f"play_game exited with code {rc}", file=sys.stderr)
        return CycleResult(rc, [], gen_seconds, 0.0)

    pending = sorted(s for s in out_dir.glob("*.slog") if not s.with_suffix(".sobs").exists())
    if not pending:
        return CycleResult(0, [], gen_seconds, 0.0)
    t1 = time.monotonic()
    rc = run_sim_obs_tool(pending, params, threads)
    sim_seconds = time.monotonic() - t1
    if rc != 0:
        print(f"sim_obs_tool exited with code {rc}", file=sys.stderr)
    done = [s for s in pending if s.with_suffix(".sobs").exists()]
    return CycleResult(rc, done, gen_seconds, sim_seconds)

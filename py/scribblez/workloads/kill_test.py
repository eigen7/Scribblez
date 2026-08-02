"""The kill-test data-generation workload (docs/sim_residual_feedback.md).

One cycle = one HastyBot self-play batch into a fresh timestamp-named .slog
file in the worker's private work dir, then sim_obs_tool over every .slog
still missing its .sobs sidecar (the fresh batch plus any backlog an
interrupted run left), then delivery of every complete pair to the tag's
slogs/ store (a rename for local workers, an upload for cloud ones). Both
artifacts land atomically, so cycles can be interrupted and resumed freely and
any number of workers can generate into the same tag.

Shared by every driver: the local CLI (scripts/generate_kill_test_data.py) and
the worker entrypoint that the master dashboard and cloud pods run.
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scribblez.params import param
from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.workloads import pair_store
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkerContext, WorkloadSpec

SIM_OBS_TOOL = "/workspace/repo/target/engine/sim_obs_tool"

# The tag's pair store, under the tag's data/ dir (locally and in the bucket).
SLOGS_DIR = "slogs"


@dataclass(frozen=True)
class KillTestParams:
    """A tag's generation parameters. Frozen at task creation: every worker on
    a tag must generate with identical settings for the data to be analyzable
    as one corpus. Worker-level knobs (thread count) deliberately live outside.
    """

    games_per_batch: int = param(200, "self-play games per generation cycle")
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
    gen_seconds: float  # self-play batch wall time
    sim_seconds: float  # sim_obs_tool wall time


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
        threads=threads,
        player_spec=hasty_player_spec(),
    )
    gen_seconds = time.monotonic() - t0
    if rc != 0:
        print(f"play_game exited with code {rc}", file=sys.stderr)
        return CycleResult(rc, gen_seconds, 0.0)

    pending = sorted(s for s in out_dir.glob("*.slog") if not s.with_suffix(".sobs").exists())
    if not pending:
        return CycleResult(0, gen_seconds, 0.0)
    t1 = time.monotonic()
    rc = run_sim_obs_tool(pending, params, threads)
    sim_seconds = time.monotonic() - t1
    if rc != 0:
        print(f"sim_obs_tool exited with code {rc}", file=sys.stderr)
    return CycleResult(rc, gen_seconds, sim_seconds)


def _cycle(work_dir: Path, params: KillTestParams, threads: int) -> tuple[int, dict]:
    """One cycle in the shared generate loop's (returncode, phases) shape."""
    r = run_one_cycle(work_dir, params, threads)
    return r.returncode, {"gen_s": r.gen_seconds, "sim_s": r.sim_seconds}


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner (the shared pair-store loop over run_one_cycle)."""
    return pair_store.run_pair_generate(ctx, _cycle, ".sobs", SLOGS_DIR)


def progress(spec: WorkloadSpec, tag: str) -> list[tuple[str, object]]:
    return [("pairs", pair_store.count_pairs(spec.paths(tag).data_dir / SLOGS_DIR, ".sobs"))]


def slog_dir(tag: str) -> Path:
    """The tag's pair store (complete .slog/.sobs pairs), for analysis tools."""
    return SPEC.paths(tag).data_dir / SLOGS_DIR


SPEC = WorkloadSpec(
    name="kill_test",
    title="Generate kill-test data",
    params_cls=KillTestParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator",
            runner="scribblez.workloads.kill_test:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            interruptible=True,
            stats=StatsSpec(
                unit="pairs",
                phases={"gen_s": "self-play", "sim_s": "sim", "upload_s": "upload"},
            ),
        ),
    ),
    progress="scribblez.workloads.kill_test:progress",
    sync_data_dirs=(SLOGS_DIR,),
)

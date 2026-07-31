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
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkerContext, WorkloadSpec
from scribblez.workloads.worker import WorkerStats, WorkerStopped

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


def deliver_pairs(sink, out_dir: Path, worker_id: str) -> tuple[int, int, float]:
    """Deliver every complete .slog/.sobs pair in `out_dir` (not just this
    cycle's -- a restarted worker flushes leftovers too) to the tag's slogs/
    store, appending a -<worker_id> stem suffix so names stay globally unique
    across workers while preserving the stem-based pair matching downstream
    tools rely on. Returns (pairs, bytes, seconds).

    The .sobs is delivered before its .slog: a .slog missing its sidecar reads
    as pending work downstream, while an orphaned .sobs is inert -- so the
    store only ever presents complete pairs plus inert leftovers.
    """
    moved, nbytes, t0 = 0, 0, time.monotonic()
    for sobs in sorted(out_dir.glob("*.sobs")):
        slog = sobs.with_suffix(".slog")
        for f in (sobs, slog):
            nbytes += sink.deliver(f, f"{SLOGS_DIR}/{f.stem}-{worker_id}{f.suffix}")
        moved += 1
    return moved, nbytes, time.monotonic() - t0


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner: cycle in a private work dir, deliver pairs."""
    work_dir = ctx.tag_paths().work_dir(ctx.worker_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    stats = WorkerStats(ctx)
    print(f"worker {ctx.worker_id} ({ctx.sink.kind}): generating tag '{ctx.tag}' with {ctx.params}")

    cycle = 0
    try:
        # A restarted worker may find completed pairs a previous run generated
        # but never delivered; flush them first.
        deliver_pairs(ctx.sink, work_dir, ctx.worker_id)
        while ctx.max_cycles == 0 or cycle < ctx.max_cycles:
            cycle += 1
            result = run_one_cycle(work_dir, ctx.params, ctx.threads)
            if result.returncode != 0:
                return result.returncode
            moved, nbytes, secs = deliver_pairs(ctx.sink, work_dir, ctx.worker_id)
            stats.cycle_done(
                {"gen_s": result.gen_seconds, "sim_s": result.sim_seconds, "upload_s": secs},
                units=moved,
                nbytes=nbytes,
            )
            print(f"cycle {cycle}: {moved} pair(s) delivered")
    except WorkerStopped:
        moved, _, _ = deliver_pairs(ctx.sink, work_dir, ctx.worker_id)
        print(f"SIGTERM: flushed {moved} completed pair(s); exiting")
    return 0


def progress(spec: WorkloadSpec, tag: str) -> list[tuple[str, object]]:
    slogs = spec.paths(tag).data_dir / SLOGS_DIR
    pairs = sum(1 for _ in slogs.glob("*.sobs")) if slogs.is_dir() else 0
    return [("pairs", pairs)]


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

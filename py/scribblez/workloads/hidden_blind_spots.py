"""The hidden_blind_spots workload: collect positions where a move from outside
HastyBot's static-equity top moves out-sims all of them, under DeepBot's own
evaluation of Alice's candidates (docs/blind_spots.md, "Hidden-leaves blind
spots on the GPU"): hidden leaves, the opponent's leave from the leave
inference model, two HastyBot replies, then the leaf network.

Its one role, a GPU generator, runs on local or ssh workers. A cycle plays a chunk of
hidden-leaves HastyBot self-play, labels `positions_per_game` turns of each
game into pools as policy_set's generators do (policy_set.write_pools: the
.forgone sidecar, then policy_set_pool_tool), and confirms every pool whose
leader sits outside the cut with blind_spot_confirm_tool. The worker delivers
what the step-4 tools read and drops the pools, megabytes per chunk:

    data/found/<stem>.slog              the chunk's games
    data/found/<stem>.forgone           their forgone plays (inferred beliefs)
    data/found/<stem>.blindspots.json   every confirmed position, kept or not
    data/gcg/<stem>-g<G>-turn<N>.gcg    the game up to each kept position

The controller counts the .gcg files and finishes the generators once the tag
holds target_positions, as blind_spots does (its tick and complete).
"""

import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from scribblez import params as params_mod
from scribblez.params import param
from scribblez.paths import ENGINE_DIR, TagPaths
from scribblez.selfplay import run_games
from scribblez.workloads import policy_set
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkerContext, WorkloadSpec
from scribblez.workloads.blind_spots import GCG_DIR, positions_found
from scribblez.workloads.policy_set import LEAF_SYNTHETIC, PoolLabelParams
from scribblez.workloads.selfplay_gen import hasty_spec
from scribblez.workloads.worker import WorkerStats, WorkerStopped

CONFIRM_TOOL = ENGINE_DIR / "blind_spot_confirm_tool"
FOUND_SUFFIX = ".blindspots.json"
# The tag's data/ subdirectory of everything but the .gcg exports.
FOUND_DIR = "found"
# What a chunk delivers into FOUND_DIR, in order: the games, then their
# sidecars. The .forgone exists only under inferred beliefs.
FOUND_SUFFIXES = (".slog", ".forgone", FOUND_SUFFIX)


@dataclass(frozen=True)
class HiddenBlindSpotsParams(PoolLabelParams):
    # Each of a cycle's two GPU tools loads its TensorRT engines, about 2 s,
    # which a 200-game chunk amortizes over about a minute of labeling.
    games_per_chunk: int = param(200, "games per generator cycle, pooled and confirmed together")
    target_positions: int = param(
        1000,
        "stop every generator once the tag holds this many kept positions (-1: never)",
        end=True,
    )
    cut: int = param(10, "HastyBot's top-K moves by static equity that a move must beat")
    picks: int = param(
        5, "moves from outside the cut, best screen q first, that the confirmation re-sims"
    )
    confirm_rollouts: int = param(5000, "confirming rollouts per move, unraced")
    min_gain: float = param(
        5.0,
        "keep a position whose best outside move beats the best cut move by at least this many "
        "points of win percentage",
    )


def confirm_command(
    params: HiddenBlindSpotsParams,
    chunk_dir: Path,
    leaf: Path | str,
    model: Path | None,
    seed: int,
    threads: int,
) -> list[str]:
    """blind_spot_confirm_tool over the pooled chunk in `chunk_dir`, the kept
    positions' games into its gcg/; with `leaf` LEAF_SYNTHETIC, on the CPU
    reference (tests)."""
    cmd = [
        str(CONFIRM_TOOL), "--slog-dir", str(chunk_dir), "--leaf-model", str(leaf),
        "--gcg-dir", str(chunk_dir / GCG_DIR),
        "--cut", str(params.cut), "--picks", str(params.picks),
        "--rollouts", str(params.confirm_rollouts),
        "--min-gain", str(params.min_gain),
        "--seed", str(seed), "--threads", str(threads),
    ]  # fmt: skip
    if leaf == LEAF_SYNTHETIC:
        cmd += ["--backend", "cpu"]
    if model is not None:
        cmd += ["--leave-inference", str(model)]
    return cmd


def play_chunk(chunk_dir: Path, worker_id: str, params: HiddenBlindSpotsParams, threads: int):
    """Play one chunk into `chunk_dir`, renamed to <stem>-<worker_id>.slog so
    every file derived from it is unique across workers."""
    code = run_games(
        chunk_dir,
        num_games=params.games_per_chunk,
        threads=threads,
        player_spec=hasty_spec(params),
        random_opening_mean=0.0,
        face_up_leaves=False,
    )
    if code != 0:
        raise RuntimeError(f"play_game exited {code}")
    (slog,) = chunk_dir.glob("*.slog")
    slog.rename(slog.with_name(f"{slog.stem}-{worker_id}.slog"))


def confirm_chunk(
    params: HiddenBlindSpotsParams, leaf: Path | str, model: Path | None, chunk_dir: Path, threads
):
    """Pool the chunk in `chunk_dir`, then confirm its candidate blind spots.
    Both tools seed from the chunk stem. Raises subprocess.CalledProcessError
    if a tool fails."""
    policy_set.write_pools(params, leaf, model, chunk_dir, threads)
    (slog,) = chunk_dir.glob("*.slog")
    seed = zlib.crc32(slog.stem.encode())
    subprocess.run(confirm_command(params, chunk_dir, leaf, model, seed, threads), check=True)


def deliver_chunk(sink, chunk_dir: Path) -> tuple[int, int]:
    """Deliver the confirmed chunk in `chunk_dir`: its games and sidecars, then
    each kept position's .gcg, which is what the controller counts. Returns
    (positions kept, bytes)."""
    (slog,) = chunk_dir.glob("*.slog")
    nbytes = 0
    for suffix in FOUND_SUFFIXES:
        src = slog.with_name(slog.stem + suffix)
        if src.exists():
            nbytes += sink.deliver(src, f"{FOUND_DIR}/{src.name}")
    kept = sorted((chunk_dir / GCG_DIR).glob("*.gcg"))
    for gcg in kept:
        nbytes += sink.deliver(gcg, f"{GCG_DIR}/{gcg.name}")
    return len(kept), nbytes


class CycleResult(NamedTuple):
    phases: dict[str, float]  # seconds, keyed as the role's StatsSpec phases
    found: int  # positions kept
    nbytes: int  # delivered


def run_cycle(ctx: WorkerContext, leaf: Path, model: Path | None, chunk_dir: Path) -> CycleResult:
    """One cycle: play, pool, confirm and deliver a chunk in `chunk_dir`."""
    p = ctx.params
    t0 = time.monotonic()
    play_chunk(chunk_dir, ctx.worker_id, p, ctx.threads)
    t1 = time.monotonic()
    confirm_chunk(p, leaf, model, chunk_dir, ctx.threads)
    t2 = time.monotonic()
    found, nbytes = deliver_chunk(ctx.sink, chunk_dir)
    phases = {"gen_s": t1 - t0, "sim_s": t2 - t1, "upload_s": time.monotonic() - t2}
    return CycleResult(phases, found, nbytes)


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner: one chunk per cycle until max_cycles or
    SIGTERM. The work dir is wiped on start, since a stopped cycle's chunk is
    redone rather than resumed."""
    try:
        leaf, model = policy_set.resolve_sources(ctx)
    except (FileNotFoundError, params_mod.ParamsError) as e:
        print(f"error: hidden_blind_spots source: {e}", file=sys.stderr)
        return 1
    work_dir = ctx.tag_paths().work_dir(ctx.worker_id)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    stats = WorkerStats(ctx)
    print(f"worker {ctx.worker_id} ({ctx.sink.kind}): collecting tag '{ctx.tag}' with {ctx.params}")
    cycle = 0
    try:
        while ctx.max_cycles == 0 or cycle < ctx.max_cycles:
            cycle += 1
            chunk_dir = work_dir / f"cycle-{cycle}"
            chunk_dir.mkdir()
            r = run_cycle(ctx, leaf, model, chunk_dir)
            shutil.rmtree(chunk_dir)
            stats.cycle_done(r.phases, units=r.found, nbytes=r.nbytes)
            print(f"cycle {cycle}: {r.found} position(s) kept")
    except WorkerStopped:
        print("SIGTERM: the chunk in progress is discarded")
    return 0


def progress(spec: WorkloadSpec, paths: TagPaths, params) -> list[tuple[str, object]]:
    data_dir = paths.data_dir
    return [
        ("positions kept", positions_found(data_dir)),
        ("chunks confirmed", sum(1 for _ in (data_dir / FOUND_DIR).glob(f"*{FOUND_SUFFIX}"))),
    ]


SPEC = WorkloadSpec(
    name="hidden_blind_spots",
    title="Collect hidden-leaves blind spots (GPU)",
    params_cls=HiddenBlindSpotsParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator (GPU)",
            runner="scribblez.workloads.hidden_blind_spots:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            inputs="scribblez.workloads.policy_set:inputs",
            gpu=True,
            stats=StatsSpec(
                unit="positions",
                phases={"gen_s": "self-play", "sim_s": "pool + confirm", "upload_s": "deliver"},
            ),
        ),
    ),
    scheduler="scribblez.workloads.blind_spots:tick",
    progress="scribblez.workloads.hidden_blind_spots:progress",
    complete="scribblez.workloads.blind_spots:complete",
    finalize="scribblez.workloads.policy_set:finalize",
    pace_role="generate",
    primary_params=("target_positions", "leave_inference_tag", "leave_inference_generation"),
    collected_dirs=(FOUND_DIR, GCG_DIR),
)

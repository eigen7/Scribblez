"""SupremeBot M1a's held-out transfer test data (docs/plans/supreme_bot_m1a.md).

Its one mode so far is step 0's measurement: how many rollouts a label needs
before within-position differences between candidates are resolvable, what a
rollout costs, how fast the recorded ply-one options saturate, and how often
each coupling kind occurs. Step 0 sets the corpus size; the corpus itself is
a later mode of the same tool.

A cycle plays a HastyBot self-play batch with face-up leaves into a fresh
.slog, runs transfer_test_generator over every .slog still missing its
.tmeasure, and delivers each complete set (the .slog, the .tmeasure
and the .trollouts of every rollout) to the tag's store. One position is
measured per game. py/scripts/transfer_test_measure_report.py reads the store.

The leaf model is a position_eval export, pinned at task creation as
move_set_eval pins its teacher, and staged for remote slots the same way.
"""

import functools
import math
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from scribblez.params import param
from scribblez.paths import ENGINE_DIR, TagPaths
from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.workloads import pair_store
from scribblez.workloads.base import (
    RoleSpec,
    SlotPlan,
    StatsSpec,
    WorkerContext,
    WorkloadSpec,
    resolve_input,
)
from scribblez.workloads.move_set_eval import teacher_onnx

TRANSFER_TEST_GENERATOR = str(ENGINE_DIR / "transfer_test_generator")
STORE_DIR = "measure"
JSON_EXT = ".tmeasure"
FLOATS_EXT = ".trollouts"
# The tag-relative name a remote slot finds its leaf model under (RoleSpec.inputs).
TEACHER_INPUT = "inputs/teacher.onnx"

# The generator's peak GPU memory in GiB: the leaf service. Not yet measured
# for this tool; this is the bound position_eval uses for the same export
# served by match eval (1.06 GiB at 28 threads, plus the 1 GiB TensorRT build
# scratch and 0.5 headroom). Replace it with the first run's measurement.
GENERATOR_GPU_GB = 1.06 + 1.0 + 0.5


@dataclass(frozen=True)
class TransferTestParams:
    """A tag's measurement parameters, frozen at task creation."""

    teacher_tag: str = param(
        "transformer-clipped",
        "position_eval tag whose export is the leaf model; one of its exports is pinned at "
        "task creation (the latest, unless teacher_generation names one)",
    )
    teacher_generation: int = param(
        -1,
        "which exported generation of teacher_tag to use (generations count from 0); -1 = its "
        "latest export at task creation, resolved then and frozen",
    )
    target_positions: int = param(
        300,
        "stop once the store holds at least this many measured positions (-1 = run until paused)",
        end=True,
    )
    games_per_batch: int = param(20, "self-play games per cycle; one position is measured per game")
    rollouts: int = param(10000, "rollouts per candidate; the analysis reads every smaller count")
    horizon: int = param(3, "plies before the leaf model scores a rollout (at least 3)")
    saturation_probes: int = param(
        256, "rollouts whose opponent racks feed the ply-one option saturation curves"
    )
    option_k: int = param(16, "options recorded per opponent rack: its static-equity top k")
    random_opening_mean: float = param(2.0, "mean random opening moves per self-play game")


@dataclass(frozen=True)
class CycleResult:
    returncode: int
    gen_seconds: float  # self-play batch wall time
    measure_seconds: float  # transfer_test_generator wall time


def run_generator(pending: list[Path], params: TransferTestParams, threads: int, model: str) -> int:
    """Measure each of `pending` under a seed of its own, so files never share
    rollout streams."""
    for slog in pending:
        cmd = [
            TRANSFER_TEST_GENERATOR,
            "--mode=measure",
            f"--slog-file={slog}",
            f"--leaf-model={model}",
            f"--horizon={params.horizon}",
            f"--rollouts={params.rollouts}",
            f"--saturation-probes={params.saturation_probes}",
            f"--option-k={params.option_k}",
            f"--seed={zlib.crc32(slog.stem.encode())}",
            f"--threads={threads}",
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"transfer_test_generator exited with code {rc}", file=sys.stderr)
            return rc
    return 0


def run_one_cycle(
    out_dir: Path, params: TransferTestParams, threads: int, model: str
) -> CycleResult:
    """One cycle into `out_dir`, with per-phase wall times."""
    t0 = time.monotonic()
    rc = run_games(
        out_dir,
        num_games=params.games_per_batch,
        threads=threads,
        player_spec=hasty_player_spec(endgame=True),
        random_opening_mean=params.random_opening_mean,
        face_up_leaves=True,
    )
    gen_seconds = time.monotonic() - t0
    if rc != 0:
        print(f"play_game exited with code {rc}", file=sys.stderr)
        return CycleResult(rc, gen_seconds, 0.0)
    pending = sorted(s for s in out_dir.glob("*.slog") if not s.with_suffix(JSON_EXT).exists())
    t1 = time.monotonic()
    rc = run_generator(pending, params, threads, model)
    return CycleResult(rc, gen_seconds, time.monotonic() - t1)


def _cycle(
    model: str, work_dir: Path, params: TransferTestParams, threads: int
) -> tuple[int, dict]:
    """One cycle in the shared generate loop's (returncode, phases) shape."""
    r = run_one_cycle(work_dir, params, threads, model)
    return r.returncode, {"gen_s": r.gen_seconds, "measure_s": r.measure_seconds}


def target_files(params: TransferTestParams) -> int:
    """The store size, in .slog files, that holds `target_positions`."""
    if params.target_positions < 0:
        return params.target_positions
    return math.ceil(params.target_positions / params.games_per_batch)


def inputs(params: TransferTestParams, mount_root: Path) -> dict[str, Path]:
    """The generate role's one out-of-tag input: the pinned leaf model."""
    return {TEACHER_INPUT: teacher_onnx(params, mount_root)}


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner: the shared pair-store loop over run_one_cycle."""
    try:
        model = str(resolve_input(ctx, TEACHER_INPUT, teacher_onnx(ctx.params, ctx.mount_root)))
    except FileNotFoundError as e:
        print(f"error: leaf model: {e}", file=sys.stderr)
        return 1
    return pair_store.run_pair_generate(
        ctx,
        functools.partial(_cycle, model),
        JSON_EXT,
        STORE_DIR,
        target_pairs=target_files(ctx.params),
        extra_sidecar_exts=(FLOATS_EXT,),
    )


def gpu_need(params, role: str) -> float | None:
    """The WorkloadSpec.gpu_need hook: GiB one slot of `role` needs."""
    return GENERATOR_GPU_GB


def layout(params, vcpus: int, generator_threads: int | None) -> list[SlotPlan]:
    """The WorkloadSpec.layout hook: one generator with every vCPU."""
    return [SlotPlan("generate", generator_threads or vcpus, gpu_need(params, "generate"))]


def progress(spec: WorkloadSpec, paths: TagPaths, params) -> list[tuple[str, object]]:
    files = pair_store.count_pairs(paths.data_dir / STORE_DIR, JSON_EXT)
    return [("files", files), ("positions", files * params.games_per_batch)]


SPEC = WorkloadSpec(
    name="transfer_test",
    title="SupremeBot M1a transfer test",
    params_cls=TransferTestParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator (GPU)",
            runner="scribblez.workloads.transfer_test:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            inputs="scribblez.workloads.transfer_test:inputs",
            gpu=True,
            stats=StatsSpec(
                unit="files",
                phases={"gen_s": "self-play", "measure_s": "measure", "upload_s": "upload"},
            ),
        ),
    ),
    # Pins teacher_generation; it reads only the two teacher fields.
    finalize="scribblez.workloads.move_set_eval:finalize",
    layout="scribblez.workloads.transfer_test:layout",
    gpu_need="scribblez.workloads.transfer_test:gpu_need",
    progress="scribblez.workloads.transfer_test:progress",
    pace_role="generate",
    collected_dirs=(STORE_DIR,),
)

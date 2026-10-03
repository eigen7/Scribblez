"""SupremeBot M1a's held-out transfer test data (docs/plans/supreme_bot_m1a.md).

Two modes, each with a profile:

- corpus: M1a's records. Per .slog, a .sprobe of every candidate's probes,
  turn by turn, and a labels .sobs, the same candidates simmed further on
  rollouts disjoint from the probes. The train-corpus profile labels at
  L = 100, the test-corpus profile at L = 1,000; the test set is its own tag,
  so train and test never share a game.
- measure: step 0, which set the corpus size: every rollout of every candidate
  at a large count, with option saturation and coupling counts.
  py/scripts/transfer_test_measure_report.py reads its store.

A cycle plays a HastyBot self-play batch with face-up leaves into a fresh
.slog, runs transfer_test_generator over every .slog still missing its mode's
sidecars, and delivers each complete set to the tag's store. One position is
taken per game.

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

from scribblez import params as params_mod
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
MODE_CORPUS = "corpus"
MODE_MEASURE = "measure"
STORE_DIR = "measure"
JSON_EXT = ".tmeasure"
FLOATS_EXT = ".trollouts"
CORPUS_DIR = "corpus"
PROBE_EXT = ".sprobe"
LABELS_EXT = ".sobs"


@dataclass(frozen=True)
class Sidecars:
    """Where a mode's files go: its store dir, the sidecar whose presence marks
    a .slog done (written last), and the ones delivered with it."""

    store_dir: str
    done_ext: str
    extra_exts: tuple[str, ...]


SIDECARS = {
    MODE_CORPUS: Sidecars(CORPUS_DIR, PROBE_EXT, (LABELS_EXT,)),
    MODE_MEASURE: Sidecars(STORE_DIR, JSON_EXT, (FLOATS_EXT,)),
}
# The tag-relative name a remote slot finds its leaf model under (RoleSpec.inputs).
TEACHER_INPUT = "inputs/teacher.onnx"

# The generator's peak GPU memory in GiB: the leaf service. Measured 2026-10-03
# on the RTX 5000 Ada with the transformer-clipped epoch-2543 export at 28
# threads, as the rise in device memory over the baseline: 0.77-1.15 GiB, the
# range being a concurrent trainer's own drift. The figure takes the top of it,
# plus the 1 GiB TensorRT build scratch on a machine without a cached plan (a
# bound from NeuralNet's workspace cap) and 0.5 headroom.
GENERATOR_GPU_GB = 1.15 + 1.0 + 0.5


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
    games_per_batch: int = param(20, "self-play games per cycle; one position is taken per game")
    mode: str = param(
        MODE_MEASURE,
        "corpus: M1a's probes and labels; measure: step 0's noise and saturation measurement",
        choices=(MODE_CORPUS, MODE_MEASURE),
    )
    horizon: int = param(3, "plies after the candidate before the leaf model scores (at least 3)")
    probes_per_candidate: int = param(125, "corpus: probes recorded per candidate")
    label_rollouts: int = param(
        100, "corpus: label rollouts per candidate (100 for training, 1000 for the test set)"
    )
    rollouts: int = param(
        10000, "measure: rollouts per candidate; the analysis reads every smaller count"
    )
    saturation_probes: int = param(
        256, "measure: rollouts whose opponent racks feed the ply-one option saturation curves"
    )
    option_k: int = param(
        16, "measure: options recorded per opponent rack, its static-equity top k"
    )
    random_opening_mean: float = param(2.0, "mean random opening moves per self-play game")


@dataclass(frozen=True)
class CycleResult:
    returncode: int
    gen_seconds: float  # self-play batch wall time
    measure_seconds: float  # transfer_test_generator wall time


def mode_flags(params: TransferTestParams) -> list[str]:
    """The generator flags of the tag's mode."""
    if params.mode == MODE_CORPUS:
        return [
            f"--probes={params.probes_per_candidate}",
            f"--label-rollouts={params.label_rollouts}",
        ]
    return [
        f"--rollouts={params.rollouts}",
        f"--saturation-probes={params.saturation_probes}",
        f"--option-k={params.option_k}",
    ]


def run_generator(pending: list[Path], params: TransferTestParams, threads: int, model: str) -> int:
    """Run the generator over each of `pending` under a seed of its own, so
    files never share rollout streams."""
    for slog in pending:
        cmd = [
            TRANSFER_TEST_GENERATOR,
            f"--mode={params.mode}",
            f"--slog-file={slog}",
            f"--leaf-model={model}",
            f"--horizon={params.horizon}",
            *mode_flags(params),
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
    done_ext = SIDECARS[params.mode].done_ext
    pending = sorted(s for s in out_dir.glob("*.slog") if not s.with_suffix(done_ext).exists())
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
    sidecars = SIDECARS[ctx.params.mode]
    return pair_store.run_pair_generate(
        ctx,
        functools.partial(_cycle, model),
        sidecars.done_ext,
        sidecars.store_dir,
        target_pairs=target_files(ctx.params),
        extra_sidecar_exts=sidecars.extra_exts,
    )


def stored_files(paths: TagPaths, params: TransferTestParams) -> int:
    """The tag's finished .slog files, counted by its mode's done sidecar."""
    sidecars = SIDECARS[params.mode]
    return pair_store.count_pairs(paths.data_dir / sidecars.store_dir, sidecars.done_ext)


def tick(spec: WorkloadSpec, task, hooks):
    """The scheduler entry: finish the generators once the store holds
    `target_positions`. A generator on this machine stops itself there, but an
    ssh one delivers into its own container and cannot count the store."""
    params = params_mod.validate(spec.params_cls, task.params)
    if params_mod.reached(stored_files(hooks.paths, params), target_files(params)):
        hooks.finish("generate")


def gpu_need(params, role: str) -> float | None:
    """The WorkloadSpec.gpu_need hook: GiB one slot of `role` needs."""
    return GENERATOR_GPU_GB


def layout(params, vcpus: int, generator_threads: int | None) -> list[SlotPlan]:
    """The WorkloadSpec.layout hook: one generator with every vCPU."""
    return [SlotPlan("generate", generator_threads or vcpus, gpu_need(params, "generate"))]


def progress(spec: WorkloadSpec, paths: TagPaths, params) -> list[tuple[str, object]]:
    files = stored_files(paths, params)
    return [("files", files), ("positions", files * params.games_per_batch)]


# A corpus file carries only ~20 s of sims per 20 positions, against ~1.7 s of
# generator startup (leaf plan, dictionary) per file, so the corpus profiles
# batch 100 games a cycle.
PROFILES = {
    "train-corpus": {
        "mode": MODE_CORPUS,
        "label_rollouts": 100,
        "target_positions": 10000,
        "games_per_batch": 100,
    },
    "test-corpus": {
        "mode": MODE_CORPUS,
        "label_rollouts": 1000,
        "target_positions": 1000,
        "games_per_batch": 100,
    },
    "measure": {"mode": MODE_MEASURE, "target_positions": 300},
}


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
                phases={"gen_s": "self-play", "measure_s": "sims", "upload_s": "upload"},
            ),
        ),
    ),
    # Pins teacher_generation; it reads only the two teacher fields.
    finalize="scribblez.workloads.move_set_eval:finalize",
    layout="scribblez.workloads.transfer_test:layout",
    gpu_need="scribblez.workloads.transfer_test:gpu_need",
    scheduler="scribblez.workloads.transfer_test:tick",
    progress="scribblez.workloads.transfer_test:progress",
    pace_role="generate",
    collected_dirs=(CORPUS_DIR, STORE_DIR),
    profiles=PROFILES,
    default_profile="train-corpus",
    primary_params=(
        "teacher_tag",
        "target_positions",
        "mode",
        "probes_per_candidate",
        "label_rollouts",
    ),
)

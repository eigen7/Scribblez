"""The move-set-evaluation target-generation workload (docs/roadmap.md A2).

One cycle = one HastyBot self-play batch into a fresh .slog in the worker's
private work dir, then move_set_eval_target_generator over every .slog still
missing its .mset sidecar (the fresh batch plus any backlog an interrupted run
left), then delivery of every complete pair to the tag's slogs/ store -- the
same cycle shape as kill_test, with the sim tool swapped for the distillation
target generator.

The teacher is the tag's frozen `teacher_model` param: an ONNX exported by a
position-eval training run, read in place. Every worker on a tag must read the
same model bytes -- the generator stamps the teacher's content hash into each
.mset, and MsetDataset refuses a corpus with mixed hashes -- which the frozen
param provides as long as the file it names is never overwritten (models/
exports are write-once, so pointing at one is safe).

The generate role is GPU and local-only for now: the teacher runs under
TensorRT, which the cloud worker image cannot host yet (the GPU-workloads item
in docs/cloud_compute.md -- a CUDA worker image plus a way to ship the teacher
to pods). The generator binary already rides in the worker bundle so that
enablement is config, not code, on this side.
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

TARGET_GENERATOR = "/workspace/repo/target/engine/move_set_eval_target_generator"

# The tag's pair store, under the tag's data/ dir (locally and in the bucket).
SLOGS_DIR = "slogs"


@dataclass(frozen=True)
class MoveSetEvalParams:
    """A tag's generation parameters, frozen at task creation. The freeze is
    what keeps the corpus coherent: MsetDataset itself enforces only a single
    teacher hash and information condition per corpus, so a consistent sampling
    scheme (the quotas below) is a generation-policy convention the frozen
    params provide, not something a mixed corpus would fail on. Worker-level
    knobs (thread count) live on the slots.
    """

    teacher_model: str = param(
        "",
        "absolute path to the teacher position-eval ONNX (e.g. a position_eval tag's "
        "models/model_epoch_NNNN.onnx); required, and must never be overwritten in place",
    )
    games_per_batch: int = param(200, "self-play games per generation cycle")
    positions_per_game: int = param(0, "eligible turns targeted per game (0 = every eligible turn)")
    # The stratified candidate sample per position (the generator's quotas):
    # dense head of the equity ranking, a slice of the contention zone, a
    # uniform tail, and exchanges.
    quota_top: int = param(4, "candidates from the head of the equity ranking")
    quota_mid: int = param(4, "candidates sampled from the contention zone")
    quota_tail: int = param(4, "candidates sampled uniformly from the remaining ranks")
    quota_exchange: int = param(2, "exchange candidates")
    mid_rank_limit: int = param(32, "exclusive rank bound of the contention zone")
    # Self-play condition (mirrors position_eval's generation params).
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    random_opening_mean: float = param(
        2.0,
        "open each game with K uniformly-random plies (K ~ round(Exp(mean))); positions "
        "before the last random ply are ineligible, so targets stay agent-play only",
    )
    face_up_leaves: bool = param(
        False,
        "play the face-up-leaves variant (docs/roadmap.md) in self-play generation; the "
        "teacher must then be an open-leaves model (the generator refuses the mismatch), "
        "and each .mset records the condition so the student trains under it too",
    )


@dataclass(frozen=True)
class CycleResult:
    returncode: int
    gen_seconds: float  # self-play batch wall time
    mset_seconds: float  # target-generator wall time


def run_target_generator(pending: list[Path], params: MoveSetEvalParams, threads: int) -> int:
    cmd = [
        TARGET_GENERATOR,
        *[f"--slog-file={p}" for p in pending],
        f"--model={params.teacher_model}",
        f"--quota-top={params.quota_top}",
        f"--quota-mid={params.quota_mid}",
        f"--quota-tail={params.quota_tail}",
        f"--quota-exchange={params.quota_exchange}",
        f"--mid-rank-limit={params.mid_rank_limit}",
        f"--positions-per-game={params.positions_per_game}",
        f"--threads={threads}",
    ]
    return subprocess.run(cmd, capture_output=False).returncode


def run_one_cycle(out_dir: Path, params: MoveSetEvalParams, threads: int) -> CycleResult:
    """One generation cycle into `out_dir`, with per-phase wall times."""
    t0 = time.monotonic()
    rc = run_games(
        out_dir,
        num_games=params.games_per_batch,
        threads=threads,
        player_spec=hasty_player_spec(params.hasty_temperature, params.hasty_top_k, endgame=True),
        random_opening_mean=params.random_opening_mean,
        face_up_leaves=params.face_up_leaves,
    )
    gen_seconds = time.monotonic() - t0
    if rc != 0:
        print(f"play_game exited with code {rc}", file=sys.stderr)
        return CycleResult(rc, gen_seconds, 0.0)

    pending = sorted(s for s in out_dir.glob("*.slog") if not s.with_suffix(".mset").exists())
    if not pending:
        return CycleResult(0, gen_seconds, 0.0)
    t1 = time.monotonic()
    rc = run_target_generator(pending, params, threads)
    mset_seconds = time.monotonic() - t1
    if rc != 0:
        print(f"move_set_eval_target_generator exited with code {rc}", file=sys.stderr)
    return CycleResult(rc, gen_seconds, mset_seconds)


def _cycle(work_dir: Path, params: MoveSetEvalParams, threads: int) -> tuple[int, dict]:
    """One cycle in the shared generate loop's (returncode, phases) shape."""
    r = run_one_cycle(work_dir, params, threads)
    return r.returncode, {"gen_s": r.gen_seconds, "mset_s": r.mset_seconds}


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner (the shared pair-store loop over run_one_cycle),
    after failing fast on a teacher the whole run would trip over."""
    p = ctx.params
    if not p.teacher_model or not Path(p.teacher_model).is_file():
        print(f"error: teacher_model {p.teacher_model!r} is not a readable file", file=sys.stderr)
        return 1
    return pair_store.run_pair_generate(ctx, _cycle, ".mset", SLOGS_DIR)


def progress(spec: WorkloadSpec, tag: str) -> list[tuple[str, object]]:
    return [("pairs", pair_store.count_pairs(spec.paths(tag).data_dir / SLOGS_DIR, ".mset"))]


def slog_dir(tag: str) -> Path:
    """The tag's pair store (complete .slog/.mset pairs) -- what MsetDataset
    takes as a data dir."""
    return SPEC.paths(tag).data_dir / SLOGS_DIR


SPEC = WorkloadSpec(
    name="move_set_eval",
    title="Generate move-set-eval targets",
    params_cls=MoveSetEvalParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator (GPU)",
            runner="scribblez.workloads.move_set_eval:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            kinds=("local",),
            gpu=True,
            interruptible=True,
            stats=StatsSpec(
                unit="pairs",
                phases={"gen_s": "self-play", "mset_s": "targets", "upload_s": "deliver"},
            ),
        ),
    ),
    progress="scribblez.workloads.move_set_eval:progress",
    sync_data_dirs=(SLOGS_DIR,),
)

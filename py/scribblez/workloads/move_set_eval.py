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

Every `sweep_every`-th pair is labeled in the generator's full-sweep mode
instead -- every legal candidate of a few positions per game, capped -- and is
the held-out slice the A3 gate metrics are read on, the stratified ~15-candidate
sample being blind to the tail moves the filter exists to catch. Which pairs
those are is a hash of the .slog stem (sweep_pair), so an interrupted cycle
resumes on the same decision, and the .mset header flag carries it downstream.

Evidence trajectories (docs/roadmap.md item 4) are a separate workload,
evidence_trajectories, which reuses this one's labeling step with the simmed
candidates force-included.

The singleton train role (scribblez/move_set_eval/trainer.py) distills the
student over the tag's pair store: repeated passes over a deterministic
file-level split, per-pass recall/rank metrics against the held-out pairs on
the dashboard's Loss tab. It runs alongside the generator rather than after it,
absorbing each pass's new pairs and holding its epoch budget until the store
reaches `target_pairs` -- so a tag with a worker of each type started together
grows its corpus, trains on all of it, and stops, unattended. It is the lean
growing-corpus loop (roadmap A3 slice 1); the generational consume->train
lifecycle is docs/generational_teacher.md.
"""

import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from scribblez.move_set_eval.targets import complete_pairs, partition_full_sweep
from scribblez.params import param
from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.workloads import mset_targets, pair_store
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkerContext, WorkloadSpec

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
    # The full-sweep held-out slice: the A3 gate metrics (top-K recall,
    # teacher-value regret) have to see every candidate, which the stratified
    # sample above structurally cannot show them.
    sweep_every: int = param(
        20,
        "label every Nth pair with a full sweep of each position's legal candidates instead "
        "of the stratified sample; such a pair is held out, never trained on (0 = none, "
        "which leaves the trainer's holdout_every fallback to reserve stratified pairs)",
    )
    sweep_positions_per_game: int = param(
        2,
        "eligible turns swept per game in a full-sweep pair; a swept position costs "
        "~1000x a stratified one to label, so this is small on purpose",
    )
    sweep_candidate_cap: int = param(
        1500,
        "plays labeled per swept position, by static-equity rank (exchanges and the played "
        "move are kept beyond it); bounds the 20k-move two-blank racks, whose surplus is "
        "redundant blank designations, and leaves normal positions complete",
    )
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
    target_pairs: int = param(
        600,
        "stop generating once the store holds this many pairs (0 = generate until paused). "
        "It is also what tells the trainer its corpus is final, so a tag with both workers "
        "started runs to completion unattended",
    )
    # Student training (the train role; scribblez/move_set_eval/trainer.py).
    train_epochs: int = param(
        20,
        "epochs over the finished corpus before the trainer stops (0 = run until paused). "
        "Passes taken while the store is still growing keep up with the generator and do "
        "not spend this budget, so it always buys passes over the whole corpus. With "
        "target_pairs = 0 there is no declared end to read, so 'finished' falls back to a "
        "pass during which nothing new arrived -- which a trainer outrunning a slow "
        "generator can hit early",
    )
    warmup_pairs: int = param(
        100,
        "pairs the store must hold before training starts. Below this a pass is mostly "
        "reuse of a corpus too small to learn from, and the held-out slice is too thin to "
        "read; the trainer waits (it also waits for the first swept pair, so the gate "
        "metrics are read on the full-sweep slice from the first pass). Reaching "
        "target_pairs releases the wait regardless, so a run smaller than this still runs",
    )
    holdout_every: int = param(
        20,
        "fallback holdout for a corpus with no full-sweep pairs: hold out every Nth pair "
        "(file-level, by sorted stem) for the recall/rank metrics; 0 evaluates on the "
        "training pairs (a smoke check, not a real held-out score). Ignored once sweep_every "
        "produces swept pairs, which are the holdout",
    )
    batch_positions: int = param(64, "positions per training batch")
    lr: float = param(1e-3, "peak learning rate of the warmup-stable-decay schedule")
    lr_warmup_rows: int = param(
        15_000_000,
        "linear LR warmup length, in candidate moves trained (this trainer's rows-clock; "
        "~half a pass over the reference corpus of docs/move_set_eval_results.md)",
    )
    lr_cycle_rows: int = param(
        300_000_000,
        "period of the stable->decay->restart LR cycle, in candidate moves trained "
        "(~10 reference passes; the last fifth of each cycle decays)",
    )
    weight_decay: float = param(1e-4, "AdamW weight decay")
    num_blocks: int = param(10, "board-trunk residual blocks")
    trunk_channels: int = param(192, "board-trunk width")
    num_heads: int = param(4, "cross-attention heads")
    contingent_features: bool = param(
        False,
        "encode the student's board input with the contingent-draw potential features; "
        "independent of the teacher's input arm",
    )
    lambda_sd: float = param(0.004, "score-diff loss weight")
    lambda_planes: float = param(1.0, "placement-plane BCE weight (roadmap item 1 readouts)")
    huber_delta_mean: float = param(10.0, "Huber delta, score-diff mean head")
    huber_delta_std: float = param(10.0, "Huber delta, score-diff std head")


@dataclass(frozen=True)
class CycleResult:
    returncode: int
    gen_seconds: float  # self-play batch wall time
    mset_seconds: float  # target-generator wall time


def sweep_pair(stem: str, sweep_every: int) -> bool:
    """Whether the pair with this .slog stem is labeled as a full sweep.

    A hash of the stem rather than a counter: the generator resumes by
    reprocessing every .slog still missing its .mset, so the decision has to be
    recoverable from the file alone, and must not depend on how many files a
    worker happens to see in one cycle.
    """
    if sweep_every <= 0:
        return False
    return zlib.crc32(stem.encode()) % sweep_every == 0


def label_pending(pending: list[Path], params: MoveSetEvalParams, threads: int) -> int:
    """Label `pending` .slog files, one generator run per selection mode over
    the files that mode claims."""
    stratified = [s for s in pending if not sweep_pair(s.stem, params.sweep_every)]
    swept = [s for s in pending if sweep_pair(s.stem, params.sweep_every)]
    rc = 0
    if stratified:
        rc = mset_targets.label_stratified(
            stratified,
            params.teacher_model,
            mset_targets.StratifiedQuotas.from_params(params),
            params.positions_per_game,
            threads,
        )
    if rc == 0 and swept:
        rc = mset_targets.label_full_sweep(
            swept,
            params.teacher_model,
            params.sweep_candidate_cap,
            params.sweep_positions_per_game,
            threads,
        )
    return rc


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
    rc = label_pending(pending, params, threads)
    return CycleResult(rc, gen_seconds, time.monotonic() - t1)


def _cycle(work_dir: Path, params: MoveSetEvalParams, threads: int) -> tuple[int, dict]:
    """One cycle in the shared generate loop's (returncode, phases) shape."""
    r = run_one_cycle(work_dir, params, threads)
    return r.returncode, {"gen_s": r.gen_seconds, "mset_s": r.mset_seconds}


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner (the shared pair-store loop over run_one_cycle)."""
    if not mset_targets.require_model_file(ctx.params.teacher_model, "teacher_model"):
        return 1
    return pair_store.run_pair_generate(
        ctx, _cycle, ".mset", SLOGS_DIR, target_pairs=ctx.params.target_pairs
    )


def progress(spec: WorkloadSpec, tag: str) -> list[tuple[str, object]]:
    return [("pairs", pair_store.count_pairs(spec.paths(tag).data_dir / SLOGS_DIR, ".mset"))]


def slog_dir(tag: str) -> Path:
    """The tag's pair store (complete .slog/.mset pairs) -- what MsetDataset
    takes as a data dir."""
    return SPEC.paths(tag).data_dir / SLOGS_DIR


def split_pair_stems(stems: list[str], holdout_every: int) -> tuple[list[str], list[str]]:
    """(train, holdout) stems: about one in `holdout_every` is held out.

    File-level (whole pairs) because position-level splits leak through shared
    game prefixes, and decided by a hash of the stem rather than by a position
    in the list -- like sweep_pair, and for a sharper reason here. A trainer
    re-takes this split as the store grows, so an assignment that depended on
    where a stem sat in the sorted list would move pairs between the sides
    whenever one arrived out of order (two generate workers interleave their
    deliveries), and a pair that changed sides is a pair trained on and then
    scored as held out.
    """
    ordered = sorted(stems)
    if holdout_every <= 0:
        return ordered, []
    held = [zlib.crc32(s.encode()) % holdout_every == 0 for s in ordered]
    train = [s for s, h in zip(ordered, held, strict=True) if not h]
    holdout = [s for s, h in zip(ordered, held, strict=True) if h]
    return train, holdout


def split_pairs(store: Path, holdout_every: int) -> tuple[list[Path], list[Path]]:
    """(train, holdout) .mset paths of a tag's complete pairs.

    Full-sweep pairs are the holdout whenever the corpus has any: they are
    evaluation-only by construction, and they are the only pairs the A3 gate
    metrics mean anything on. Their games are then trained on by nobody, which
    is the same file-level reservation `holdout_every` makes -- so the two do
    not compound, and holdout_every only reserves stratified pairs when a
    corpus has no swept ones at all (sweep_every=0, or a corpus predating the
    mode).
    """
    stratified, swept = partition_full_sweep(complete_pairs(store))
    if swept:
        return sorted(stratified), sorted(swept)
    train, holdout = split_pair_stems([f.stem for f in stratified], holdout_every)
    return [store / f"{s}.mset" for s in train], [store / f"{s}.mset" for s in holdout]


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
                phases={
                    "gen_s": "self-play",
                    "mset_s": "targets",
                    "upload_s": "deliver",
                },
            ),
        ),
        RoleSpec(
            name="train",
            title="Student trainer (GPU)",
            runner="scribblez.move_set_eval.trainer:run",
            singleton=True,
            kinds=("local",),
            gpu=True,
            stats=StatsSpec(unit="rows", phases={"train_s": "train", "eval_s": "eval"}),
        ),
    ),
    progress="scribblez.workloads.move_set_eval:progress",
    sync_data_dirs=(SLOGS_DIR,),
)

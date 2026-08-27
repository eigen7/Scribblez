"""The evidence-trajectory workload (docs/roadmap.md item 4): the data for the
evidence-conditioned pass and the proves-best head (items 2 and 3).

One cycle = one HastyBot self-play batch into a fresh .slog in the worker's
private work dir, then evidence_trajectory_generator over every .slog still
missing its .sobs sidecar, then move_set_eval_target_generator (--sobs) over
every .slog still missing its .mset, then delivery of every complete
.slog/.sobs/.mset triple to the tag's slogs/ store -- the move_set_eval cycle
shape with the trajectory phase in front of the labeling.

The trajectory generator sims, at each sampled position, the sequential loop
the deployed agent will run: the greedy anchor, then temperature-softmax
proposals from the tag's frozen `proposer_model` (a move-set-eval student
export), then one uniform-random tail draw -- each candidate under common
random numbers -- and records the ordered candidates with their sim outcomes
(docs/sim_residual_feedback.md, "Evidence-trajectory generation"). The .sobs
is both the evidence input (any prefix of the proposer picks) and the training
target (each held-out simmed candidate's sim value and its CRN-paired gain over
the prefix's best-so-far); the trainer reads no teacher labels.

The .mset labeling still runs, with the simmed candidates force-included: it
is cheap next to the sims, and both tools sample positions from the same seed
stream, so its coverage of the simmed positions is a property of this cycle
rather than something a later task could add. It gives the dashboard the
teacher's value per simmed candidate and keeps the corpus usable for joint
(backbone-unfrozen) training later.

The proposer is frozen like the teacher: every .sobs stamps its content hash,
and a corpus of mixed proposers is refused downstream. Point it at a write-once
export (a move_set_eval tag's models/model_epoch_NNNN.onnx).

The generate role is GPU and local-only: proposer and teacher both run under
TensorRT (see move_set_eval).

The singleton train role (scribblez/evidence/trainer.py) trains the fusion
stage and the proves-best head over the tag's pair store, on top of the
student named by `student_checkpoint` -- its backbone frozen by default, or
(`unfreeze_backbone`) trained jointly under the .mset distillation
anchor -- with the mset trainer's growing-corpus pacing: it absorbs each
pass's new pairs and spends its epoch budget only once the store is final.
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scribblez.params import param
from scribblez.paths import ENGINE_DIR
from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.sim_evidence.position_sets import TrajectoryRecipe
from scribblez.workloads import mset_targets, pair_store
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkerContext, WorkloadSpec

TRAJECTORY_GENERATOR = str(ENGINE_DIR / "evidence_trajectory_generator")

# The tag's pair store, under the tag's data/ dir (locally and in the bucket).
SLOGS_DIR = "slogs"


@dataclass(frozen=True)
class EvidenceTrajectoriesParams:
    """A tag's generation parameters, frozen at task creation so every worker
    sims the same recipe with the same proposer and the corpus reads as one.
    Worker-level knobs (thread count) live on the slots.
    """

    proposer_model: str = param(
        "",
        "absolute path to the move-set-eval student ONNX that proposes trajectory "
        "candidates (a move_set_eval tag's models/model_epoch_NNNN.onnx); required, and "
        "must never be overwritten in place",
    )
    teacher_model: str = param(
        "",
        "absolute path to the teacher position-eval ONNX that labels the .mset sidecars; "
        "required, frozen like proposer_model",
    )
    games_per_batch: int = param(200, "self-play games per generation cycle")
    positions_per_game: int = param(
        1,
        "eligible turns per game given a trajectory (and labeled); a trajectory costs "
        "rollouts x candidates rollout-games, so this is small on purpose",
    )
    # The trajectory recipe.
    rollouts: int = param(200, "Monte-Carlo rollouts per trajectory candidate")
    horizon: int = param(
        0,
        "value truncation (roadmap item 2): rollouts stop after this many plies and "
        "leaf_model scores the horizon; 0 rolls out to a natural game end",
    )
    leaf_model: str = param(
        "",
        "absolute path to the position-eval ONNX that scores rollout horizons; required with, "
        "and only with, horizon, and frozen like proposer_model",
    )
    proposals_min: int = param(2, "least model proposals per trajectory")
    proposals_max: int = param(8, "most model proposals per trajectory")
    temperature: float = param(0.05, "proposal softmax temperature, in win-equity units")
    proposal_pool: int = param(64, "proposals are drawn from the model's top-N unsimmed candidates")
    # The .mset labeling's stratified sample around the forced candidates
    # (move_set_eval's quotas; the resulting .mset labels are read by the trainer
    # only in unfrozen mode).
    quota_top: int = param(4, "labeled candidates from the head of the equity ranking")
    quota_mid: int = param(4, "labeled candidates sampled from the contention zone")
    quota_tail: int = param(4, "labeled candidates sampled uniformly from the remaining ranks")
    quota_exchange: int = param(2, "labeled exchange candidates")
    mid_rank_limit: int = param(32, "exclusive rank bound of the contention zone")
    # Self-play condition (mirrors move_set_eval's generation params).
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    random_opening_mean: float = param(
        2.0,
        "open each game with K uniformly-random plies (K ~ round(Exp(mean))); positions "
        "before the last random ply are ineligible, so targets stay agent-play only",
    )
    face_up_leaves: bool = param(
        True,
        "play the face-up-leaves variant (docs/roadmap.md) in self-play generation and sim "
        "with the opponent's leave known; proposer and teacher must then be open-leaves "
        "models (the tools refuse the mismatch)",
    )
    target_pairs: int = param(
        0,
        "stop generating once the store holds this many pairs (0 = generate until paused). "
        "It is also what tells the trainer its corpus is final; with 0 the trainer reads "
        "'finished' off the store going quiet",
    )
    # Fusion + proves-best training (the train role; scribblez/evidence/trainer.py).
    student_checkpoint: str = param(
        "",
        "absolute path to the move-set-eval student's rolling checkpoint (a move_set_eval "
        "tag's checkpoints/model.pt) whose backbone the fusion stage and proves-best head "
        "train on top of; its arch and encoding arm are read from the checkpoint",
    )
    unfreeze_backbone: bool = param(
        False,
        "train the whole model jointly -- the conditioned pass on sim outcomes plus the plain "
        "pass on the tag's .mset teacher labels (the distillation anchor) -- and export the "
        "plain student per pass; off, the student's backbone is held at its checkpoint and "
        "only the fusion stage and proves-best head train",
    )
    backbone_lr_mult: float = param(
        0.1,
        "unfrozen mode: the backbone's learning rate as a fraction of lr (the fusion stage "
        "and head start from zero-init/random and want the full rate; the distilled trunk "
        "should not be shaken at it)",
    )
    lambda_sim: float = param(
        1.0,
        "unfrozen mode: weight of the sim-outcome (conditioned) loss against the "
        "distillation loss in the joint step's total (0 = plain distillation only)",
    )
    lambda_planes: float = param(
        1.0, "unfrozen mode: placement-plane BCE weight in the distillation loss"
    )
    train_epochs: int = param(
        20,
        "epochs over the finished corpus before the trainer stops (0 = run until paused); "
        "passes over a still-growing corpus do not spend the budget",
    )
    warmup_pairs: int = param(50, "pairs the store must hold before training starts")
    holdout_every: int = param(
        10, "hold out every Nth pair (file-level, by stem hash) for the metrics; 0 = on-train"
    )
    batch_positions: int = param(32, "positions per training batch")
    lr: float = param(1e-3, "peak learning rate of the warmup-stable-decay schedule")
    lr_warmup_rows: int = param(
        800_000,
        "linear LR warmup length, in held-out candidate rows trained (this trainer's "
        "rows-clock: ~4.5 per position per pass, so ~half a pass over a 350k-position corpus)",
    )
    lr_cycle_rows: int = param(
        16_000_000,
        "period of the stable->decay->restart LR cycle, in held-out candidate rows trained "
        "(~10 passes over a 350k-position corpus; the last fifth of each cycle decays)",
    )
    weight_decay: float = param(1e-4, "AdamW weight decay")
    lambda_sd: float = param(0.004, "score-diff (sim delta moments) loss weight")
    lambda_gain: float = param(1.0, "proves-best gain loss weight")
    huber_delta_mean: float = param(10.0, "Huber delta, score-diff mean head (points)")
    huber_delta_std: float = param(10.0, "Huber delta, score-diff std head (points)")
    huber_delta_gain: float = param(0.05, "Huber delta, proves-best gain (win-probability units)")
    # Activation-magnitude restoring forces (docs/fp16_safe_serving.md),
    # read through move_set_eval's LossConfig. Both engage only in the joint
    # (unfrozen) mode -- the only one that trains the shared trunk: the
    # z-loss inside the distillation batch's compute_loss, the pooled-FC
    # penalty via the joint step's PoolFcPenalty recorder (train_loop.py).
    lambda_wld_z: float = param(1e-4, "z-loss weight on the WLD logits (squared logsumexp)")
    lambda_pool_act: float = param(
        1e-6, "mean-square penalty weight on the trunk's pooled-FC pre-activations"
    )
    grad_clip: float = param(
        1.0, "max gradient norm over all trainable params per step (0 = no clipping)"
    )


def recipe_of(params: EvidenceTrajectoriesParams) -> TrajectoryRecipe:
    """The tag's trajectory recipe as the position-set sidecar cache keys it
    (the dashboard's trajectory pane and the trainer's position-set metric
    both sim the set under exactly the tag's recipe)."""
    return TrajectoryRecipe(
        rollouts=params.rollouts,
        proposals_min=params.proposals_min,
        proposals_max=params.proposals_max,
        temperature=params.temperature,
        proposal_pool=params.proposal_pool,
        open_leaves=params.face_up_leaves,
    )


def max_evidence(params: EvidenceTrajectoriesParams) -> int:
    """The padded evidence-set width the trainer uses: the longest trajectory
    the recipe can produce (anchor + proposals_max + the uniform tail)."""
    return 1 + params.proposals_max + 1


@dataclass(frozen=True)
class CycleResult:
    returncode: int
    gen_seconds: float  # self-play batch wall time
    traj_seconds: float  # trajectory-generator wall time
    mset_seconds: float  # target-generator wall time


def run_trajectory_generator(
    pending: list[Path], params: EvidenceTrajectoriesParams, threads: int
) -> int:
    """Give `pending` .slog files trajectory .sobs sidecars, proposed by the
    tag's frozen proposer. The tool skips files whose .sobs already exists, so
    a resumed cycle sims nothing twice."""
    cmd = [
        TRAJECTORY_GENERATOR,
        *[f"--slog-file={p}" for p in pending],
        f"--model={params.proposer_model}",
        f"--rollouts={params.rollouts}",
        *(
            [f"--horizon={params.horizon}", f"--leaf-model={params.leaf_model}"]
            if params.horizon
            else []
        ),
        f"--positions-per-game={params.positions_per_game}",
        f"--proposals-min={params.proposals_min}",
        f"--proposals-max={params.proposals_max}",
        f"--temperature={params.temperature}",
        f"--proposal-pool={params.proposal_pool}",
        f"--threads={threads}",
        *(["--open-leaves"] if params.face_up_leaves else []),
    ]
    rc = subprocess.run(cmd, capture_output=False).returncode
    if rc != 0:
        print(f"evidence_trajectory_generator exited with code {rc}", file=sys.stderr)
    return rc


def run_one_cycle(out_dir: Path, params: EvidenceTrajectoriesParams, threads: int) -> CycleResult:
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
        return CycleResult(rc, gen_seconds, 0.0, 0.0)

    # Trajectories first: the labeling force-includes the simmed candidates
    # from the same-stem .sobs, so it needs them on disk.
    pending = sorted(s for s in out_dir.glob("*.slog") if not s.with_suffix(".mset").exists())
    t1 = time.monotonic()
    unsimmed = [s for s in pending if not s.with_suffix(".sobs").exists()]
    if unsimmed and (rc := run_trajectory_generator(unsimmed, params, threads)) != 0:
        return CycleResult(rc, gen_seconds, time.monotonic() - t1, 0.0)
    traj_seconds = time.monotonic() - t1

    t2 = time.monotonic()
    rc = 0
    if pending:
        rc = mset_targets.label_stratified(
            pending,
            params.teacher_model,
            mset_targets.StratifiedQuotas.from_params(params),
            params.positions_per_game,
            threads,
            with_sobs=True,
        )
    return CycleResult(rc, gen_seconds, traj_seconds, time.monotonic() - t2)


def _cycle(work_dir: Path, params: EvidenceTrajectoriesParams, threads: int) -> tuple[int, dict]:
    """One cycle in the shared generate loop's (returncode, phases) shape."""
    r = run_one_cycle(work_dir, params, threads)
    return r.returncode, {
        "gen_s": r.gen_seconds,
        "traj_s": r.traj_seconds,
        "mset_s": r.mset_seconds,
    }


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner (the shared pair-store loop over run_one_cycle).
    A pair is complete at its .mset, the last member produced; the .sobs rides
    along."""
    p = ctx.params
    ok = mset_targets.require_model_file(p.proposer_model, "proposer_model")
    ok = mset_targets.require_model_file(p.teacher_model, "teacher_model") and ok
    if p.horizon:
        ok = mset_targets.require_model_file(p.leaf_model, "leaf_model") and ok
    if not ok:
        return 1
    return pair_store.run_pair_generate(
        ctx,
        _cycle,
        ".mset",
        SLOGS_DIR,
        target_pairs=p.target_pairs,
        extra_sidecar_exts=(".sobs",),
    )


def progress(spec: WorkloadSpec, tag: str) -> list[tuple[str, object]]:
    return [("pairs", pair_store.count_pairs(spec.paths(tag).data_dir / SLOGS_DIR, ".mset"))]


def slog_dir(tag: str) -> Path:
    """The tag's pair store (complete .slog/.sobs/.mset triples)."""
    return SPEC.paths(tag).data_dir / SLOGS_DIR


SPEC = WorkloadSpec(
    name="evidence_trajectories",
    title="Generate evidence trajectories",
    params_cls=EvidenceTrajectoriesParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator (GPU)",
            runner="scribblez.workloads.evidence_trajectories:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            kinds=("local",),
            gpu=True,
            interruptible=True,
            stats=StatsSpec(
                unit="pairs",
                phases={
                    "gen_s": "self-play",
                    "traj_s": "trajectories",
                    "mset_s": "targets",
                    "upload_s": "deliver",
                },
            ),
        ),
        RoleSpec(
            name="train",
            title="Fusion + proves-best trainer (GPU)",
            runner="scribblez.evidence.trainer:run",
            singleton=True,
            kinds=("local",),
            gpu=True,
            stats=StatsSpec(unit="rows", phases={"train_s": "train", "eval_s": "eval"}),
        ),
    ),
    progress="scribblez.workloads.evidence_trajectories:progress",
    sync_data_dirs=(SLOGS_DIR,),
)

"""The evidence-trajectory workload (docs/roadmap.md item 4): the training data
for the move proposal model (item 5), plus that model's trainer and a match
readout of the agent it drives.

A generate cycle has three phases, each run over every .slog in the worker's
work dir still missing its output, so a resumed cycle picks up a backlog:

  1. HastyBot self-play writes a fresh .slog.
  2. evidence_trajectory_generator sims trajectories into a .sobs sidecar.
  3. move_set_eval_target_generator --sobs labels a .mset sidecar.

Every complete .slog/.sobs/.mset triple is then delivered to the tag's slogs/
store. This is the move_set_eval cycle with the trajectory phase in front.

At each sampled position the trajectory generator sims the loop the deployed
agent runs: the greedy anchor, then on-policy picks drawn by temperature
softmax from the frozen `proposer_model` (a move-set-eval student export), then
a few off-policy candidates drawn uniformly from the untaken legal moves. All
candidates share common random numbers (CRN). The .sobs records the ordered
candidates with their sim outcomes and evidence roles
(docs/plans/sim_residual_feedback.md, "Evidence-trajectory generation"). It
is both the model's evidence input and its training target; the trainer reads
no teacher labels.

The .mset labeling runs anyway, with the simmed candidates force-included. It
is cheap next to the sims, and it can only cover the simmed positions here:
both tools pick positions from the same seed stream, so a later task could not
add it. It gives the dashboard the teacher's value for each simmed candidate.

The proposer is frozen like the teacher: every .sobs stamps the proposer's
content hash, and the trainer's dataset refuses a corpus of mixed proposers.
Point `proposer_model` at a move_set_eval tag's models/model_epoch_NNNN.onnx.
This tag copies it into its own pinned/ on first use (mset_targets.pin_model),
because the source tag prunes its exports as it trains.

Roles:

  - generate: GPU, local only; the proposer and teacher both run under TensorRT.
  - train (scribblez/evidence/trainer.py): trains the move proposal model on
    the store's sim outcomes, starting from the student named by
    `student_checkpoint`. It paces itself to the growing store the way the
    move_set_eval trainer does, and exports the cache/step graph pair
    UltimateBot plays every pass.
  - match_eval: plays every Nth exported pair as UltimateBot (item 6) at this
    tag's rollout and truncation settings against a fixed opponent, through
    the same controller-assigned inbox as position_eval's match role
    (scribblez/match_eval/), so the Match tab tracks the agent's strength as
    it trains.
"""

import functools
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez.params import ParamsError, param
from scribblez.paths import ENGINE_DIR, MATCH_RESULTS_DIR
from scribblez.selfplay import hasty_player_spec, run_games
from scribblez.sim_evidence.position_sets import TrajectoryRecipe
from scribblez.workloads import mset_targets, pair_store
from scribblez.workloads.base import RoleSpec, StatsSpec, WorkerContext, WorkloadSpec

TRAJECTORY_GENERATOR = str(ENGINE_DIR / "evidence_trajectory_generator")

# The tag's pair store, under the tag's data/ dir (locally and in the bucket).
SLOGS_DIR = "slogs"


@dataclass(frozen=True)
class EvidenceTrajectoriesParams:
    """A tag's parameters, frozen at task creation so every worker sims the
    same recipe with the same proposer and the corpus reads as one. Worker-level
    knobs (thread count) live on the slots.
    """

    proposer_model: str = param(
        "",
        "absolute path to the move-set-eval student ONNX that proposes trajectory "
        "candidates (a move_set_eval tag's models/model_epoch_NNNN.onnx); required. Copied "
        "into this tag's pinned/ on first use, since the source tag prunes its exports",
    )
    teacher_model: str = param(
        "",
        "absolute path to the teacher position-eval ONNX that labels the .mset sidecars; "
        "required, frozen like proposer_model",
    )
    games_per_batch: int = param(200, "self-play games per generation cycle")
    positions_per_game: int = param(
        1,
        "eligible turns per game that get a trajectory (and a label); a trajectory costs "
        "rollouts x candidates rollout games, so this is small on purpose",
    )
    # The trajectory recipe.
    rollouts: int = param(200, "Monte-Carlo rollouts per trajectory candidate")
    horizon: int = param(
        0,
        "value truncation (roadmap item 2): rollouts stop after this many plies and "
        "leaf_model scores the position reached; 0 rolls out to the end of the game",
    )
    leaf_model: str = param(
        "",
        "absolute path to the position-eval ONNX that scores truncated rollouts; required "
        "when horizon is set, ignored otherwise; frozen like proposer_model",
    )
    on_policy_min: int = param(2, "least on-policy (proposer) picks per trajectory")
    on_policy_max: int = param(8, "most on-policy (proposer) picks per trajectory")
    temperature: float = param(0.05, "proposal softmax temperature, in win-equity units")
    off_policy_count: int = param(
        3, "labels-only off-policy draws, uniform over the untaken legal moves"
    )
    # Stratum quotas for the .mset labeling's stratified sample, drawn around
    # the force-included simmed candidates: a handful per position for dense
    # teacher-value labels. They do not affect the trajectory itself.
    quota_top: int = param(4, "labeled head candidates")
    quota_mid: int = param(4, "candidates sampled from the contention zone")
    quota_tail: int = param(4, "candidates sampled from the remaining ranks")
    quota_exchange: int = param(2, "exchange candidates")
    mid_rank_limit: int = param(32, "exclusive rank bound of the contention zone")
    # Self-play condition (mirrors move_set_eval's generation params).
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    random_opening_mean: float = param(
        2.0,
        "open each game with K uniformly random plies (K ~ round(Exp(mean))); positions "
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
        "Reaching it also tells the trainer its corpus is final; with 0 the trainer "
        "treats the corpus as final once no pair has arrived for 15 minutes",
    )
    # Move proposal training (the train role; scribblez/evidence/trainer.py).
    student_checkpoint: str = param(
        "",
        "absolute path to the move-set-eval student's rolling checkpoint (a move_set_eval "
        "tag's checkpoints/model.pt) that the fusion stage and proves-best head are trained "
        "on top of; its architecture and encoding arm are read from the checkpoint",
    )
    unfreeze_backbone: bool = param(
        False,
        "train the whole model on the sim-outcome loss (trunk, move encoder and heads all "
        "follow the sim signal, with no distillation anchor) and also export the plain "
        "student each pass; off holds the student's backbone at its checkpoint and trains "
        "only the fusion stage and proves-best head",
    )
    backbone_lr_mult: float = param(
        0.1,
        "with unfreeze_backbone: the backbone's learning rate as a fraction of lr. The fusion "
        "stage and head start untrained and want the full rate; the distilled backbone "
        "would be disrupted by it",
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
    # Subset assembly (evidence.dataset.assemble_subset). Both knobs change how
    # many held-out rows a pass yields, which is the clock the LR schedule
    # runs on, so they are frozen per run like the recipe.
    subsets_per_pool: int = param(
        1,
        "evidence subsets drawn per simmed pool per pass; each trains on the pool's candidates "
        "outside the subset",
    )
    empty_fraction: float = param(
        0.0,
        "probability that a drawn evidence subset is empty (the rows that keep the "
        "evidence-free pass calibrated); 0 draws the subset size uniformly over 0..cap "
        "instead, so a subset is empty about 1/(cap+1) of the time",
    )
    lr: float = param(1e-3, "peak learning rate of the warmup-stable-decay schedule")
    lr_warmup_rows: int = param(
        800_000,
        "linear LR warmup length, in held-out candidate rows trained (this trainer's "
        "rows-clock: ~4.5 per position per pass at subsets_per_pool 1, so ~half a pass over "
        "a 350k-position corpus)",
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
    grad_clip: float = param(
        1.0, "max gradient norm over all trainable params per step (0 = no clipping)"
    )
    # Match eval (the match_eval role; docs/evaluation_plan.md).
    match_every_generations: int = param(
        5, "match-eval cadence: play a match for every Nth exported generation; 0 disables"
    )
    match_pairs: int = param(200, "mirrored game pairs per match (games = 2x)")
    match_opponent: str = param(
        "--type=hastybot-endgame", "the fixed opponent's --player spec, e.g. --type=mset-sim ..."
    )
    match_seed: int = param(
        1, "base seed of the match deals; every generation plays the same pairs (nonzero)"
    )
    match_max_sims: int = param(
        10,
        "UltimateBot's sim budget per turn in match play, anchor included; at most 2 + "
        "on_policy_max. The model trained on evidence sets of at most 1 + on_policy_max (its "
        "export records this), and the agent's last pick conditions on one sim fewer than "
        "the budget",
    )

    # The UltimateBot factory enforces this bound when it loads the export.
    # Checking it here, where params are created, fails one tag-creation form
    # instead of every match.
    def __post_init__(self):
        widest = 2 + self.on_policy_max
        if not 1 <= self.match_max_sims <= widest:
            raise ParamsError(
                f"match_max_sims must be in [1, 2 + on_policy_max] = [1, {widest}], got "
                f"{self.match_max_sims}"
            )


def recipe_of(params: EvidenceTrajectoriesParams) -> TrajectoryRecipe:
    """The tag's trajectory recipe, the key of the position-set .sobs cache. The
    dashboard's trajectory pane and the trainer's position-set metric both sim
    the set under exactly this recipe."""
    return TrajectoryRecipe(
        rollouts=params.rollouts,
        on_policy_min=params.on_policy_min,
        on_policy_max=params.on_policy_max,
        temperature=params.temperature,
        off_policy_count=params.off_policy_count,
        open_leaves=params.face_up_leaves,
    )


def max_off_policy(params: EvidenceTrajectoriesParams) -> int:
    """The most off-policy draws a trajectory can carry: the uniform floor."""
    return params.off_policy_count


def max_evidence_width(params: EvidenceTrajectoriesParams) -> int:
    """The padded evidence-set capacity: the anchor plus the most on-policy
    picks. Off-policy draws are training labels only and never enter an
    evidence set."""
    return 1 + params.on_policy_max


def max_pool_width(params: EvidenceTrajectoriesParams) -> int:
    """The most candidates a position's pool can hold: anchor, on-policy picks
    and off-policy draws. The trainer checks the corpus against it."""
    return 1 + params.on_policy_max + max_off_policy(params)


@dataclass(frozen=True)
class CycleResult:
    returncode: int
    gen_seconds: float  # self-play batch wall time
    traj_seconds: float  # trajectory-generator wall time
    mset_seconds: float  # target-generator wall time


def run_trajectory_generator(
    pending: list[Path], proposer: Path, params: EvidenceTrajectoriesParams, threads: int
) -> int:
    """Give `pending` .slog files trajectory .sobs sidecars, proposed by the
    tag's pinned proposer."""
    cmd = [
        TRAJECTORY_GENERATOR,
        *[f"--slog-file={p}" for p in pending],
        f"--model={proposer}",
        f"--rollouts={params.rollouts}",
        *(
            [f"--horizon={params.horizon}", f"--leaf-model={params.leaf_model}"]
            if params.horizon
            else []
        ),
        f"--positions-per-game={params.positions_per_game}",
        f"--on-policy-min={params.on_policy_min}",
        f"--on-policy-max={params.on_policy_max}",
        f"--temperature={params.temperature}",
        f"--off-policy-count={params.off_policy_count}",
        f"--threads={threads}",
        *(["--open-leaves"] if params.face_up_leaves else []),
    ]
    rc = subprocess.run(cmd, capture_output=False).returncode
    if rc != 0:
        print(f"evidence_trajectory_generator exited with code {rc}", file=sys.stderr)
    return rc


def run_one_cycle(
    out_dir: Path, proposer: Path, params: EvidenceTrajectoriesParams, threads: int
) -> CycleResult:
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

    # Trajectories first: the labeling reads the simmed candidates to
    # force-include from the same-stem .sobs.
    pending = sorted(s for s in out_dir.glob("*.slog") if not s.with_suffix(".mset").exists())
    t1 = time.monotonic()
    unsimmed = [s for s in pending if not s.with_suffix(".sobs").exists()]
    if unsimmed and (rc := run_trajectory_generator(unsimmed, proposer, params, threads)) != 0:
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


def _cycle(
    proposer: Path, work_dir: Path, params: EvidenceTrajectoriesParams, threads: int
) -> tuple[int, dict]:
    """One cycle in the shared generate loop's (returncode, phases) shape."""
    r = run_one_cycle(work_dir, proposer, params, threads)
    return r.returncode, {
        "gen_s": r.gen_seconds,
        "traj_s": r.traj_seconds,
        "mset_s": r.mset_seconds,
    }


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner: the shared pair-store loop over run_one_cycle.
    A triple is complete once its .mset, the last output, exists."""
    p = ctx.params
    ok = True
    try:
        proposer = mset_targets.pin_model(p.proposer_model, ctx.tag_paths(), "proposer_model")
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        ok = False
    ok = mset_targets.require_model_file(p.teacher_model, "teacher_model") and ok
    if p.horizon:
        ok = mset_targets.require_model_file(p.leaf_model, "leaf_model") and ok
    if not ok:
        return 1
    return pair_store.run_pair_generate(
        ctx,
        functools.partial(_cycle, proposer),
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
            runtime=RUNTIME_TORCH,
            ingest="scribblez.generational.train_ingest:tick",
            singleton=True,
            kinds=("local",),
            gpu=True,
            stats=StatsSpec(unit="rows", phases={"train_s": "train", "eval_s": "eval"}),
        ),
        # Local only: with truncation on, the match needs --leaf-model, an
        # absolute path into another tag's models/, and the controller's
        # one-file inbox delivery cannot ship it to an ssh container.
        RoleSpec(
            name="match_eval",
            title="Match eval (GPU)",
            runner="scribblez.match_eval.runner:run",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            dispatch="scribblez.match_eval.dispatch:tick",
            singleton=True,
            kinds=("local",),
            gpu=True,
            stats=StatsSpec(unit="games", phases={"match_s": "match play"}),
        ),
    ),
    progress="scribblez.workloads.evidence_trajectories:progress",
    sync_data_dirs=(SLOGS_DIR,),
    local_data_dirs=(MATCH_RESULTS_DIR,),
)

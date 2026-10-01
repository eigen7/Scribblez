"""The position-evaluation training workload: the teacher model's pipeline.

Three roles share a tag:

  - generate: any number of interchangeable workers, each delivering HastyBot
    self-play chunks into the tag's staging area (workloads/selfplay_gen.py).
  - train: a singleton that trains over a sliding window of complete
    generations, one epoch per generation, exporting ONNX and dashboard
    records per generation. It runs on this machine's GPU or on a rented one;
    a rented trainer reads its generations from the bucket and delivers its
    outputs there (docs/plans/cloud_training.md).
  - match_eval: a singleton that plays exported generations against a fixed
    opponent (scribblez/match_eval/runner.py). It may run on another machine
    (kind "ssh") so the matches do not compete with training for this host's
    GPU. The controller assigns its work and ingests its results
    (scribblez/match_eval/dispatch.py), so the slot needs only a GPU and the
    worker image, not the exports or the database.

The generation scheduler (scribblez/generational/scheduler.py) moves staged
chunks into generation directories and paces the generators against the
trainer's published cursor.

The params below are frozen at task creation because they define the corpus
and the model, which every worker on a tag must share. Knobs the operator may
change mid-run (DataLoader workers, torch threads) are dashboard controls
instead, and per-worker resources (threads, vCPUs) belong to the worker slots.
"""

from dataclasses import dataclass

from cloud import worker_deps
from cloud.runtime_abi import RUNTIME_TORCH

from scribblez.generational.optimizer_arms import OPTIMIZER_SCHEDULE_FREE, OPTIMIZERS
from scribblez.generational.scheduler import TICK_FOR_TASK
from scribblez.params import param
from scribblez.paths import MATCH_RESULTS_DIR
from scribblez.trunk_arms import TRUNK_CONV, TRUNK_TRANSFORMER, TRUNKS
from scribblez.workloads.base import RoleSpec, SlotPlan, StatsSpec, WorkerContext, WorkloadSpec
from scribblez.workloads.selfplay_gen import GENERATOR_STATS, STAGING_DIR, generate, hasty_spec

TRAINER_STATS = StatsSpec(
    unit="rows",
    phases={"train_s": "train", "eval_s": "eval", "upload_s": "upload"},
    background=frozenset({"upload_s"}),  # the trainer's OutputDeliverer thread
)

# Parameter profiles (WorkloadSpec.profiles): one recipe per trunk. Each
# overrides only some of the dataclass defaults below, so a knob no profile
# names has the same value under both. A tuning result that an A/B has shown to
# help the transformer is promoted into its profile; gradient clipping, the
# standard transformer safeguard, is the first such setting. The conv profile
# is what conv runs have always trained under.
PROFILES = {
    TRUNK_TRANSFORMER: {
        "trunk": TRUNK_TRANSFORMER,
        "grad_clip": 1.0,
        "activation_checkpointing": False,
    },
    TRUNK_CONV: {"trunk": TRUNK_CONV},
}


@dataclass(frozen=True)
class PositionEvalParams:
    # Generation.
    games_per_generation: int = param(20000, "self-play games per generation")
    open_ahead: int = param(
        4, "generations kept open ahead of the trainer's cursor before generators are parked"
    )
    hasty_temperature: float = param(0.0, "HastyBot softmax temperature (0 = greedy)")
    hasty_top_k: int = param(10, "HastyBot candidate count when the temperature is > 0")
    random_opening_mean: float = param(
        2.0,
        "open each game with K uniformly random plies (K ~ round(Exp(mean))) before the "
        "HastyBots take over, so the corpus reaches off-policy states; 0 disables",
    )
    face_up_leaves: bool = param(
        True,
        "play the face-up-leaves variant (docs/roadmap.md) in both self-play generation and "
        "match eval, so the model trains and is measured under one information condition",
    )
    weirdbot_generation: bool = param(
        False,
        "generate the diagnostic WeirdBot corpus: the leave-forcing WeirdBot plays both "
        "self-play seats instead of HastyBot, so the interaction between the opponent's "
        "leave letters and board cross-checks dominates the training signal",
    )
    # Match eval (the match_eval role; docs/evaluation_plan.md).
    match_every_generations: int = param(
        5, "match-eval cadence: play a match for every Nth exported generation; 0 disables"
    )
    match_opponent: str = param(
        "--type=hastybot-endgame", "the fixed opponent's --player spec, e.g. --type=sim"
    )
    match_pairs: int = param(
        200, "mirrored game pairs per generation's match; every match plays all of them"
    )
    match_seed: int = param(
        1,
        "base game seed for matches, fixed per tag so every generation faces identical deals; "
        "must be nonzero",
    )
    # Training window.
    window: int = param(4, "generations trained over (sliding window); <=0 keeps all")
    turns_per_game: int = param(1, "turns trained per game per generation; 0 = every eligible turn")
    max_rows: int = param(
        -1, "stop the trainer after this many rows (-1 = run until paused)", end=True
    )
    # Optimization.
    batch_size: int = param(256, "minibatch size")
    optimizer: str = param(
        OPTIMIZER_SCHEDULE_FREE,
        "optimizer arm (scribblez/generational/optim.py): 'wsd' is AdamW on a "
        "warmup-stable-decay schedule over rows trained; 'muon' runs that schedule with Muon "
        "on the transformer tower's weight matrices (transformer trunk only); "
        "'schedule_free' is AdamWScheduleFree, "
        "which needs no schedule or horizon and makes every generation's export deployable",
        choices=OPTIMIZERS,
    )
    lr: float = param(
        0.0,
        "learning rate: the schedule's peak under wsd, the constant rate under schedule_free; "
        "0 = the arm's own default",
    )
    lr_warmup_rows: int = param(
        200_000, "linear LR warmup length, in positions trained (~2.5 default generations)"
    )
    lr_cycle_rows: int = param(
        2_000_000,
        "period of the stable->decay->restart LR cycle, in positions trained "
        "(~25 default generations; the last fifth of each cycle decays); "
        "unused by schedule_free",
    )
    weight_decay: float = param(
        1e-4,
        "AdamW weight decay, applied to weight matrices and conv kernels only (not to norm "
        "gains, biases or the transformer's positional parameters)",
    )
    adam_beta2: float = param(
        0.999,
        "Adam's second-moment decay rate (beta2); 0.95 adapts faster to shifts in gradient "
        "scale and is the usual choice for transformers",
    )
    grad_clip: float = param(
        0.0, "clip each step's gradient to this global norm (clip_grad_norm_); 0 = no clipping"
    )
    # Model.
    num_blocks: int = param(10, "residual blocks")
    trunk_channels: int = param(192, "trunk width")
    use_film: bool = param(
        False,
        "FiLM conditioning where the trunk injects scalar and global context: the scalars "
        "emit a per-channel gain alongside the additive bias; off injects additively only",
    )
    trunk: str = param(
        TRUNK_CONV,
        "trunk tower (scribblez/spatial_trunk.py): 'conv' is a residual conv tower; "
        "'transformer' is a KataGo-style nested-bottleneck transformer over the board cells "
        "plus 27 tile-supply register tokens (one per tile type, carrying its rack and "
        "unseen-pool counts, and the opponent's leave under face-up leaves), so the placement "
        "heads can gate a square's cross-checks on whether the tiles that fit it are available",
        choices=TRUNKS,
    )
    transformer_mid_channels: int = param(
        192, "transformer trunk: width inside each nested-bottleneck block"
    )
    transformer_heads: int = param(
        6, "transformer trunk: attention heads per layer (head dim = mid channels / heads)"
    )
    transformer_ffn_channels: int = param(512, "transformer trunk: SwiGLU FFN hidden width")
    activation_checkpointing: bool = param(
        True,
        "transformer trunk: recompute each attention/FFN pair's activations in backward "
        "instead of storing them; position_eval at batch 256 needs ~4 GiB instead of ~10 GiB "
        "but trains ~30% slower. The transformer profile turns it off",
    )
    transformer_qk_norm: bool = param(
        False,
        "transformer trunk: RMS-normalize each attention head's queries and keys, which "
        "bounds the attention logits and guards against loss spikes at high learning rates",
    )
    # Loss.
    lambda_wld: float = param(
        1.0, "win/draw/loss (value) loss weight; lower it to isolate other heads"
    )
    lambda_sd: float = param(0.0002, "score-diff loss weight")
    lambda_next_placement: float = param(
        0.5, "loss weight of the next-move placement heads (opp and self footprints)"
    )
    lambda_win_placement: float = param(
        0.5,
        "loss weight of the win placement heads, which predict Pr[footprint and that player "
        "wins] (opp and self)",
    )
    huber_delta_mean: float = param(10.0, "Huber delta, score-diff mean head")
    huber_delta_std: float = param(10.0, "Huber delta, score-diff std head")


# The trainer's peak GPU memory in GiB, measured 2026-09-28 on the RTX 5000
# Ada: the whole process as nvidia-smi sees it (the CUDA context included),
# over a generation's training on transformer-clipped data (compiled, bf16)
# and the quality eval that follows it. The eval, not the training step, sets
# the peak. The figure in the table is that peak plus TRAINER_GPU_HEADROOM_GB.
#
#   transformer, activation checkpointing off  13.53 GiB  (training alone 10.58)
#   transformer, activation checkpointing on    8.46 GiB  (training alone  4.94)
#   conv                                        2.86 GiB  (training alone  1.46)
#
# Measured at the default architecture and batch size only (TRAINER_GPU_KEY);
# any other configuration has no figure, and the queue refuses to place it.
# The optimizer arm and QK-norm move memory by well under the headroom.
TRAINER_GPU_HEADROOM_GB = 0.5
# Match eval: a 40-pair match of a transformer-clipped export against
# HastyBot-endgame at 28 threads peaked at 1.06 GiB, with the TensorRT plan
# already cached. On a machine without the cached plan the engine first builds
# one, whose scratch is capped at 1 GiB (NeuralNet's workspace_bytes); that part
# is a bound from the cap, not a measurement.
MATCH_EVAL_GPU_GB = 1.06 + 1.0 + TRAINER_GPU_HEADROOM_GB
_MEASURED_GB = {
    (TRUNK_TRANSFORMER, False): 13.53,
    (TRUNK_TRANSFORMER, True): 8.46,
    (TRUNK_CONV, False): 2.86,
    (TRUNK_CONV, True): 2.86,  # checkpointing only affects the transformer tower
}
# The params the measurements held fixed, at their defaults.
TRAINER_GPU_KEY = (
    "batch_size",
    "num_blocks",
    "trunk_channels",
    "transformer_mid_channels",
    "transformer_heads",
    "transformer_ffn_channels",
)


def gpu_need(params, role: str) -> float | None:
    """The WorkloadSpec.gpu_need hook: GiB one slot of `role` needs."""
    if role == "generate":
        return 0.0
    if role == "match_eval":
        return MATCH_EVAL_GPU_GB
    defaults = PositionEvalParams()
    if any(getattr(params, k) != getattr(defaults, k) for k in TRAINER_GPU_KEY):
        return None
    peak = _MEASURED_GB[(params.trunk, params.activation_checkpointing)]
    return peak + TRAINER_GPU_HEADROOM_GB


def layout(params, vcpus: int, generator_threads: int | None) -> list[SlotPlan]:
    """The WorkloadSpec.layout hook: a trainer and a generator per machine,
    plus match eval when its cadence is on. The generator gets every vCPU,
    as tags placed by hand do: the trainer's loader competes for them only
    while it trains, and the scheduler parks the generator once it runs
    open_ahead generations ahead."""
    roles = ["train", "generate"] + (["match_eval"] if params.match_every_generations > 0 else [])
    threads = {"generate": generator_threads or vcpus}
    return [SlotPlan(r, threads.get(r), gpu_need(params, r)) for r in roles]


def fetch_train_deps(params):
    """Runtime data the trainer needs beyond the bundle: the engine's default
    lexicon (the FFI session loads it before the model is built) and the
    position-evaluation eval datasets. Not Macondo's strategy tables: the
    trainer plays no moves."""
    worker_deps.fetch_lexicon(worker_deps.DEFAULT_LEXICON)
    worker_deps.fetch_eval_positions()


def run_generate(ctx: WorkerContext) -> int:
    """The generate-role runner: selfplay_gen's, with WeirdBot in both seats
    instead of HastyBot when weirdbot_generation is set."""
    p = ctx.params
    return generate(ctx, "--type=weirdbot" if p.weirdbot_generation else hasty_spec(p))


SPEC = WorkloadSpec(
    name="position_eval",
    title="Train position evaluation",
    params_cls=PositionEvalParams,
    roles=(
        RoleSpec(
            name="generate",
            title="Generator",
            runner="scribblez.workloads.position_eval:run_generate",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            stats=GENERATOR_STATS,
        ),
        RoleSpec(
            name="train",
            title="Trainer (GPU)",
            runner="scribblez.position_eval.trainer:run",
            runtime=RUNTIME_TORCH,
            deps="scribblez.workloads.position_eval:fetch_train_deps",
            ingest="scribblez.generational.train_ingest:tick",
            singleton=True,
            kinds=("local", "ssh"),
            gpu=True,
            stats=TRAINER_STATS,
        ),
        RoleSpec(
            name="match_eval",
            title="Match eval (GPU)",
            runner="scribblez.match_eval.runner:run",
            deps="scribblez.workloads.selfplay_gen:fetch_deps",
            dispatch="scribblez.match_eval.dispatch:tick",
            singleton=True,
            kinds=("local", "ssh"),
            gpu=True,
            stats=StatsSpec(unit="games", phases={"match_s": "match play"}),
        ),
    ),
    scheduler=TICK_FOR_TASK,
    layout="scribblez.workloads.position_eval:layout",
    gpu_need="scribblez.workloads.position_eval:gpu_need",
    progress="scribblez.generational.scheduler:progress",
    pace_role="train",
    collected_dirs=(STAGING_DIR, MATCH_RESULTS_DIR),
    profiles=PROFILES,
    default_profile=TRUNK_TRANSFORMER,
    primary_params=(
        "face_up_leaves",
        "games_per_generation",
        "random_opening_mean",
        "match_every_generations",
        "optimizer",
        "trunk",
    ),
)

"""SupremeBot M1a's reader (docs/plans/supreme_bot_m1a.md, PR 4): train one
reader on a transfer_test corpus tag.

A tag is one run of the size sweep: a reader shape (the reader-* profiles),
a corpus tag, and optionally a fixed subset of its positions. The corpus tag's
prior caches must exist (py/scripts/transfer_test_prior.py). A fixed share of
the corpus's positions, chosen by a hash of file and position, is held out
for validation, so every run on a corpus validates on the same positions; the
test corpus stays untouched until the evaluation harness (PR 5).

The trainer (scribblez/transfer_test/trainer.py) publishes its step cursor
in train_state.json; the scheduler finishes the train slot once it reaches
train_steps.
"""

from dataclasses import dataclass
from pathlib import Path

from cloud.runtime_abi import RUNTIME_TORCH

from scribblez import params as params_mod
from scribblez.generational import lifecycle
from scribblez.params import param
from scribblez.paths import TagPaths
from scribblez.workloads import pair_store
from scribblez.workloads.base import RoleSpec, SlotPlan, StatsSpec, WorkloadSpec
from scribblez.workloads.transfer_test import CORPUS_DIR, PROBE_EXT
from scribblez.workloads.transfer_test import SPEC as CORPUS_SPEC

PRIOR_EXT = ".sprior"


@dataclass(frozen=True)
class TransferReaderParams:
    """A reader run's parameters, frozen at task creation."""

    corpus_tag: str = param(
        "m1a-train-corpus", "transfer_test tag whose corpus and prior caches the reader trains on"
    )
    train_positions: int = param(
        -1,
        "train on this many of the corpus's training positions, a fixed random subset "
        "(-1 = all of them)",
    )
    val_fraction: float = param(
        0.05,
        "share of the corpus's positions held out for validation, chosen by a hash of file and "
        "position, so every run on the corpus holds out the same ones",
    )
    # The reader's shape.
    width: int = param(256, "model width")
    depth: int = param(6, "transformer blocks")
    heads: int = param(8, "attention heads")
    kv_heads: int = param(2, "key/value heads, shared by groups of attention heads")
    activation_checkpointing: bool = param(
        False, "recompute each block's activations in the backward pass, trading time for memory"
    )
    # The optimization.
    train_steps: int = param(
        20000,
        "optimizer steps to train (-1 = until paused, at the peak learning rate after the warmup)",
        end=True,
    )
    batch_rows: int = param(32, "rows per optimizer step")
    lr: float = param(3e-4, "peak learning rate (AdamW), reached after the warmup")
    warmup_steps: int = param(
        500, "linear warmup steps; a cosine decay to a tenth follows, ending at train_steps"
    )
    weight_decay: float = param(0.1, "AdamW weight decay on weight matrices")
    grad_clip: float = param(1.0, "gradient norm clip")
    eval_every: int = param(500, "steps between validation passes and checkpoints")
    # The rows (scribblez/transfer_test/rows.py).
    max_held_out: int = param(4, "a row holds out a uniform 1 to this many candidates")
    max_probes: int = param(32, "a probed candidate keeps a uniform 0 to this many probes")
    max_tokens: int = param(2048, "context tokens per row, the root's 226 included")
    query_points: int = param(4, "context prefixes queried per row, the whole context included")
    graded_max: int = param(
        0, "graded rows: each held-out candidate keeps 1 to this many probes (0 = none)"
    )
    # The loss (scribblez/transfer_test/loss.py).
    lambda_wld: float = param(1.0, "win/draw/loss cross-entropy weight")
    lambda_score: float = param(1.0, "score-difference Gaussian NLL weight")
    lambda_expected: float = param(1.0, "expected-score Gaussian NLL weight")
    lambda_footprint: float = param(0.5, "next-move footprint cross-entropy weight")
    lambda_rank: float = param(1.0, "gap-weighted ranking weight")
    rank_temperature: float = param(0.01, "ranking loss: expected-score difference per logit")
    loader_workers: int = param(6, "row-assembly worker processes")
    seed: int = param(0, "row sampling seed")


def store_dir(params, mount_root: Path) -> Path:
    return TagPaths(params.corpus_tag, CORPUS_SPEC.name, mount_root).data_dir / CORPUS_DIR


def finalize(spec: WorkloadSpec, paths: TagPaths, params):
    """Refuse a corpus tag without probes or with a missing prior cache, and
    a run with no validation positions, at task creation where the operator
    sees it."""
    if not 0 < params.val_fraction < 1:
        raise params_mod.ParamsError(
            f"val_fraction must be in (0, 1), not {params.val_fraction}: every eval cycle validates"
        )
    corpus = store_dir(params, paths.mount_root)
    files = pair_store.complete_pairs(corpus, PROBE_EXT)
    if not files:
        raise params_mod.ParamsError(f"corpus tag '{params.corpus_tag}' has no probe files")
    missing = [f.name for f in files if not f.with_suffix(PRIOR_EXT).exists()]
    if missing:
        raise params_mod.ParamsError(
            f"corpus tag '{params.corpus_tag}' lacks {len(missing)} prior cache(s), e.g. "
            f"{missing[0]}; run py/scripts/transfer_test_prior.py --tag {params.corpus_tag}"
        )
    return params


def steps_done(paths: TagPaths) -> int:
    return int(lifecycle.read_train_state(paths).get("steps", 0))


def complete(spec: WorkloadSpec, paths: TagPaths, params) -> bool:
    return params_mod.reached(steps_done(paths), params.train_steps)


def tick(spec: WorkloadSpec, task, hooks):
    """Finish the trainer once its cursor reaches train_steps."""
    params = params_mod.validate(spec.params_cls, task.params)
    if complete(spec, hooks.paths, params):
        hooks.finish("train")


def progress(spec: WorkloadSpec, paths: TagPaths, params) -> list[tuple[str, object]]:
    state = lifecycle.read_train_state(paths)
    return [("steps", state.get("steps", 0)), ("rows", state.get("rows_trained", 0))]


# Peak GPU memory in GiB per reader shape: (width, depth, batch_rows,
# max_tokens, activation_checkpointing) -> GiB. Measured 2026-10-05 on the RTX
# 5000 Ada as torch's peak reserved memory over training steps on real rows
# (padded to the fixed context, so every batch has the measured shape), plus
# 0.5 for the CUDA context and 0.5 headroom. A shape not listed is unmeasured.
_CONTEXT_AND_HEADROOM_GB = 1.0
GPU_GB: dict[tuple, float] = {
    (128, 4, 32, 2048, False): 3.25 + _CONTEXT_AND_HEADROOM_GB,
    (256, 6, 32, 2048, False): 4.55 + _CONTEXT_AND_HEADROOM_GB,
    (512, 8, 32, 2048, False): 11.26 + _CONTEXT_AND_HEADROOM_GB,
    # Without checkpointing this shape does not fit a 16 GiB card.
    (768, 14, 32, 2048, True): 8.71 + _CONTEXT_AND_HEADROOM_GB,
}


def gpu_need(params, role: str) -> float | None:
    key = (
        params.width,
        params.depth,
        params.batch_rows,
        params.max_tokens,
        params.activation_checkpointing,
    )
    return GPU_GB.get(key)


def layout(params, vcpus: int, generator_threads: int | None) -> list[SlotPlan]:
    """One trainer; its threads are the row-assembly workers' cores."""
    return [SlotPlan("train", params.loader_workers, gpu_need(params, "train"))]


# The size sweep's reader shapes: 1.3M, 5.0M, 23.6M and 90.9M parameters.
PROFILES = {
    "reader-1m": {"width": 128, "depth": 4, "heads": 4, "kv_heads": 2},
    "reader-5m": {"width": 256, "depth": 6, "heads": 8, "kv_heads": 2},
    "reader-25m": {"width": 512, "depth": 8, "heads": 8, "kv_heads": 2},
    "reader-100m": {
        "width": 768,
        "depth": 14,
        "heads": 12,
        "kv_heads": 4,
        "activation_checkpointing": True,
    },
}


SPEC = WorkloadSpec(
    name="transfer_reader",
    title="SupremeBot M1a reader",
    params_cls=TransferReaderParams,
    roles=(
        RoleSpec(
            name="train",
            title="Reader trainer (GPU)",
            runner="scribblez.transfer_test.trainer:run",
            runtime=RUNTIME_TORCH,
            ingest="scribblez.generational.train_ingest:tick",
            singleton=True,
            # It reads the corpus tag's store and prior caches from this
            # machine's tree.
            kinds=("local",),
            gpu=True,
            stats=StatsSpec(unit="rows", phases={"train_s": "train", "eval_s": "eval"}),
        ),
    ),
    scheduler="scribblez.workloads.transfer_reader:tick",
    progress="scribblez.workloads.transfer_reader:progress",
    complete="scribblez.workloads.transfer_reader:complete",
    finalize="scribblez.workloads.transfer_reader:finalize",
    layout="scribblez.workloads.transfer_reader:layout",
    gpu_need="scribblez.workloads.transfer_reader:gpu_need",
    pace_role="train",
    profiles=PROFILES,
    default_profile="reader-5m",
    primary_params=("corpus_tag", "train_positions", "train_steps", "width", "depth"),
)

"""The transfer_reader workload's train role (docs/plans/supreme_bot_m1a.md,
PR 4): train one reader on a corpus tag for a fixed number of steps.

The corpus is loaded whole (scribblez/transfer_test/corpus.py) and split into
training and validation positions (split_positions). Worker processes
assemble and collate training rows from random training positions, forked
after the load so they share the corpus's memory. The validation rows are
assembled once, under a fixed seed, so every pass scores the same rows.

AdamW over the decay groups the generational trainers use, with a linear
warmup and a cosine decay to a tenth, in bf16 autocast. The tower, the part
of the reader whose shapes are fixed by the padding, is compiled.

Every eval_every steps: a validation pass (the losses and the held-out
readout, loss.readout), the rolling checkpoint and the step cursor
(train_state.json), then the dashboard record, in that order, so a recorded
pass has its resume point. A stopped worker resumes at the last pass.

The training steps keep their losses and gradient norms on the device, read
once per eval cycle: a per-step read would stall the host on the GPU and
leave the next batch's receive from the row workers unoverlapped.
"""

from __future__ import annotations

import math
import os
import time
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from scribblez import params as params_mod
from scribblez.generational import checkpoint, lifecycle
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.optim import decay_groups
from scribblez.generational.records import TrainRecorder
from scribblez.position_eval.train_loop import GradNormTracker
from scribblez.train_common import timed_print
from scribblez.transfer_test.corpus import CorpusFile, load_corpus
from scribblez.transfer_test.loss import (
    LOSS_TERMS,
    LossWeights,
    finish_readout,
    losses,
    readout,
)
from scribblez.transfer_test.reader import Reader, ReaderConfig
from scribblez.transfer_test.rows import RowConfig, assemble_row
from scribblez.transfer_test.tokens import collate
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.transfer_reader import store_dir
from scribblez.workloads.worker import WorkerStats, WorkerStopped

VAL_SEED = 12345  # the validation rows' sampling seed, fixed across runs
SUBSET_SEED = 0  # the train_positions subset's seed, fixed across runs
MIN_LR_FRACTION = 0.1

Position = tuple[int, int]  # (file index, position in the file)


@dataclass
class ReaderTrainState(GenerationalState):
    """generation_index counts validation passes; steps counts optimizer steps."""

    steps: int = 0


def row_config(params) -> RowConfig:
    return RowConfig(
        max_held_out=params.max_held_out,
        max_probes=params.max_probes,
        max_tokens=params.max_tokens,
        query_points=params.query_points,
        graded_max=params.graded_max,
    )


def query_len(params, max_slots: int) -> int:
    return params.query_points * max_slots


def split_positions(files: list[CorpusFile], params) -> tuple[list[Position], list[Position]]:
    """(training, validation) positions. A position is validation by a hash of
    its file's name and its index, so the split depends on neither the run nor
    the file order; train_positions then keeps a fixed random subset of the
    rest."""
    cut = int(params.val_fraction * 2**32)
    train, val = [], []
    for i, f in enumerate(files):
        stem = f.probes.path.stem
        for p in range(f.num_positions):
            is_val = zlib.crc32(f"{stem}:{p}".encode()) < cut
            (val if is_val else train).append((i, p))
    if params.train_positions >= 0:
        keep = np.random.default_rng(SUBSET_SEED).permutation(len(train))[: params.train_positions]
        train = [train[k] for k in sorted(keep)]
    return train, val


class RowStream(IterableDataset):
    """Endless collated batches of rows at uniformly drawn positions, each
    worker on a stream of its own. A batch leaves the worker as numpy arrays
    (to_numpy), which cross to the trainer pickled through a pipe: as torch
    tensors they would go through /dev/shm, which a container holds to 64 MB,
    and a few prefetched batches of prior placements fill it."""

    def __init__(self, files, positions, cfg: RowConfig, batch_rows: int, q_len: int, seed: int):
        self.files, self.positions, self.cfg = files, positions, cfg
        self.batch_rows, self.q_len, self.seed = batch_rows, q_len, seed

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng((self.seed, info.id if info else 0))
        while True:
            picks = rng.integers(len(self.positions), size=self.batch_rows)
            rows = [self._row(self.positions[k], rng) for k in picks]
            yield collate(rows, self.cfg.max_tokens, self.q_len).apply(torch.Tensor.numpy)

    def _row(self, at: Position, rng):
        return assemble_row(self.files[at[0]], at[0], at[1], self.cfg, rng)


def validation_batches(files, positions, cfg: RowConfig, batch_rows: int, q_len: int) -> list:
    rng = np.random.default_rng(VAL_SEED)
    rows = [assemble_row(files[i], i, p, cfg, rng) for i, p in positions]
    return [
        collate(rows[k : k + batch_rows], cfg.max_tokens, q_len)
        for k in range(0, len(rows), batch_rows)
    ]


def lr_at(step: int, params) -> float:
    """Linear warmup to params.lr, then a cosine decay to MIN_LR_FRACTION of it
    at train_steps; an unbounded run holds params.lr."""
    if step < params.warmup_steps:
        return params.lr * (step + 1) / params.warmup_steps
    if params_mod.unbounded(params.train_steps):
        return params.lr
    t = (step - params.warmup_steps) / max(params.train_steps - params.warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return params.lr * (MIN_LR_FRACTION + (1 - MIN_LR_FRACTION) * cosine)


def loss_weights(params) -> LossWeights:
    return LossWeights(
        wld=params.lambda_wld,
        score=params.lambda_score,
        expected=params.lambda_expected,
        footprint=params.lambda_footprint,
        rank=params.lambda_rank,
        rank_temperature=params.rank_temperature,
    )


class Averages:
    """Running means of named scalars."""

    def __init__(self):
        self.sums: dict[str, float] = defaultdict(float)
        self.count = 0

    def add(self, values: dict[str, float]):
        for k, v in values.items():
            self.sums[k] += v
        self.count += 1

    def means(self) -> dict[str, float]:
        return {k: v / max(self.count, 1) for k, v in self.sums.items()}


def run_steps(model, tower, optimizer, batches, device, params, state, n: int) -> dict:
    """`n` optimizer steps; the mean losses and gradient-norm statistics, read
    from the device once at the end (see the module docstring)."""
    model.train()
    weights = loss_weights(params)
    parameters = [p for p in model.parameters() if p.requires_grad]
    norms = GradNormTracker(device, params.grad_clip)
    sums: dict[str, torch.Tensor] = {}
    for _ in range(n):
        b = next(batches).apply(torch.from_numpy).to(device)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(state.steps, params)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            terms = losses(model(b, tower=tower), b, weights)
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        norms.clip_and_record(parameters)
        optimizer.step()
        for k, v in terms.items():
            sums[k] = sums.get(k, 0) + v.detach().float()
        state.steps += 1
        state.rows_trained += params.batch_rows
    return {**{k: v.item() / n for k, v in sums.items()}, **norms.summary()}


@torch.no_grad()
def validate(model, tower, batches: list, device, params) -> dict:
    """Mean validation losses (val_<term>) and the held-out readout."""
    model.eval()
    weights = loss_weights(params)
    avg, sums = Averages(), defaultdict(float)
    for cpu_batch in batches:
        b = cpu_batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b, tower=tower)
            terms = losses(out, b, weights)
        avg.add({f"val_{k}": float(v) for k, v in terms.items()})
        for k, v in readout(out, b).items():
            sums[k] += float(v)
    return {**avg.means(), **finish_readout(sums)}


def record_of(train: dict, val: dict, lr: float, train_s: float) -> dict:
    """The dashboard record: the training losses as the Loss tab's stacked
    terms (loss, loss_<term>), and everything else under its own name."""
    return {
        "loss": train["total"],
        **{f"loss_{k}": train[k] for k in LOSS_TERMS},
        **{k: v for k, v in train.items() if k.startswith(("grad_norm", "clip"))},
        **val,
        "lr": lr,
        "elapsed_s": train_s,
    }


def publish_config(recorder, tag: str, params, n_params: int = 0):
    weights = loss_weights(params)
    recorder.publish_run(
        tag, asdict(params), n_params, {f"loss_{k}": weights.of(k) for k in LOSS_TERMS}, {}
    )


def save(ctx, paths, model, optimizer, state, config: dict):
    """The rolling checkpoint, then the cursor the scheduler reads."""
    checkpoint.save(paths, model, optimizer, state, config)
    ctx.sink.deliver_output(paths.rolling_checkpoint, "checkpoints/model.pt", keep=True)
    lifecycle.write_train_state(paths, asdict(state))
    ctx.sink.deliver_output(paths.train_state_path, "train_state.json", keep=True)


def build_reader(params, files: list[CorpusFile]) -> Reader:
    cfg = ReaderConfig(
        width=params.width,
        depth=params.depth,
        heads=params.heads,
        kv_heads=params.kv_heads,
        teacher_width=files[0].prior.root_board.shape[-1],
        max_slots=max(int(f.probes.positions["num_candidates"].max()) for f in files),
        activation_checkpointing=params.activation_checkpointing,
    )
    return Reader(cfg)


def load(params, mount_root: Path) -> tuple[list[CorpusFile], list[Position], list[Position]]:
    t0 = time.time()
    files = load_corpus(store_dir(params, mount_root))
    train, val = split_positions(files, params)
    timed_print(
        f"Corpus {params.corpus_tag}: {len(files)} files, {len(train)} training and "
        f"{len(val)} validation positions, loaded in {time.time() - t0:.0f}s"
    )
    return files, train, val


def run(ctx: WorkerContext) -> int:
    """The train-role entry point."""
    params = ctx.params
    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    recorder = TrainRecorder(ctx.sink)
    publish_config(recorder, ctx.tag, params)
    if params_mod.reached(checkpoint.peek_state(paths, ReaderTrainState).steps, params.train_steps):
        timed_print("Training complete (the step budget was spent in an earlier session).")
        return 0

    files, train, val = load(params, ctx.mount_root)
    model = build_reader(params, files).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    timed_print(f"Reader: {n_params:,} parameters, {model.cfg}")
    publish_config(recorder, ctx.tag, params, n_params)
    optimizer = torch.optim.AdamW(
        decay_groups(list(model.named_parameters()), params.weight_decay), lr=params.lr
    )
    state = checkpoint.resume(paths, model, optimizer, device, state_cls=ReaderTrainState)
    config = {**asdict(params), "reader": model.cfg.to_dict()}

    cfg, q_len = row_config(params), query_len(params, model.cfg.max_slots)
    val_batches = validation_batches(files, val, cfg, params.batch_rows, q_len)
    stream = RowStream(files, train, cfg, params.batch_rows, q_len, params.seed + state.steps)
    loader = DataLoader(
        stream, batch_size=None, num_workers=params.loader_workers, prefetch_factor=4
    )
    batches = iter(loader)
    torch.set_float32_matmul_precision("high")
    tower = torch.compile(model.tower)
    stats = WorkerStats(ctx)
    try:
        while not params_mod.reached(state.steps, params.train_steps):
            n = params.eval_every
            if not params_mod.unbounded(params.train_steps):
                n = min(n, params.train_steps - state.steps)
            t0 = time.time()
            train_metrics = run_steps(model, tower, optimizer, batches, device, params, state, n)
            train_s = time.time() - t0
            val_metrics = validate(model, tower, val_batches, device, params)
            eval_s = time.time() - t0 - train_s
            lr = lr_at(state.steps - 1, params)
            timed_print(
                f"[step {state.steps}] loss={train_metrics['total']:.4f} "
                f"val={val_metrics['val_total']:.4f} held-out rmse reader="
                f"{val_metrics['heldout_rmse_reader']:.4f} prior="
                f"{val_metrics['heldout_rmse_prior']:.4f} lr={lr:.2e} {train_s:.0f}s"
            )
            generation = state.generation_index
            state.generation_index += 1
            save(ctx, paths, model, optimizer, state, config)
            recorder.commit_generation(
                generation, state.rows_trained, record_of(train_metrics, val_metrics, lr, train_s)
            )
            stats.cycle_done(
                {"train_s": train_s, "eval_s": eval_s}, units=n * params.batch_rows, nbytes=0
            )
        timed_print(f"Training complete: {state.steps} steps, {state.rows_trained} rows.")
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; the last validation pass is checkpointed.")
    return 0

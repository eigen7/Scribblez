"""One training epoch of the slot form, its held-out eval, and the time to
build a batch of sets (docs/plans/deep_bot_v0.md, "Training both forms" and
"Eval metrics"). The model runs under bf16 autocast; the coverage loss runs in
FP32.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from scribblez.policy_set import metrics
from scribblez.policy_set.dataset import PoolBatcher, PoolItem, batch_order
from scribblez.policy_set.model import coverage_loss, slot_picks, to_grid

# The batch eval_set_ms times: this many sets at this N.
SET_TIMING_POSITIONS = 256
SET_TIMING_N = 8


def to_device(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def slot_forward(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """The slot model's pointer logits and padding mask, under autocast."""
    with torch.autocast(device_type=batch["input_spatial"].device.type, dtype=torch.bfloat16):
        return model(batch)


def slot_loss(
    logits: torch.Tensor, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """The batch's coverage loss summed over the set sizes the slots reach,
    and per N."""
    num_positions = batch["input_spatial"].shape[0]
    q_grid, _ = to_grid(batch["q"].float(), batch["move_pos_id"], num_positions)
    by_n = coverage_loss(logits, q_grid)
    return sum(by_n.values()), by_n


@dataclass
class EpochResult:
    loss: float  # mean coverage loss summed over the set sizes, per batch
    loss_by_n: dict[int, float]
    grad_norm: float  # mean pre-clip global norm
    rows_trained: int  # pools


def run_epoch(
    model,
    optimizer,
    batcher: PoolBatcher,
    items: list[PoolItem],
    device,
    *,
    batch_size: int,
    seed: int,
    lr_fn: Callable[[int], float] | None,
    rows_trained: int,
    grad_clip: float,
    on_batch: Callable[[int, int, float, int], None] | None = None,
) -> EpochResult:
    """One pass over `items` in shuffled batches of `batch_size` pools. A
    pool is the rows-clock's unit."""
    model.train()
    loss_sum = norm_sum = 0.0
    by_n_sum: dict[int, float] = {}
    n_batches = n_rows = 0
    t0 = last = time.time()
    for idx in batch_order(len(items), batch_size, seed):
        batch = to_device(batcher.build([items[i] for i in idx]), device)
        rows_trained += len(idx)
        n_rows += len(idx)
        if lr_fn is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr_fn(rows_trained)
        total, by_n = slot_loss(slot_forward(model, batch)[0], batch)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip if grad_clip > 0 else float("inf")
        )
        optimizer.step()
        n_batches += 1
        loss_sum += total.item()
        norm_sum += norm.item()
        for n, v in by_n.items():
            by_n_sum[n] = by_n_sum.get(n, 0.0) + v.item()
        if on_batch is not None and time.time() - last > 1.0:
            on_batch(n_batches, n_rows, time.time() - t0, rows_trained)
            last = time.time()
    nb = max(n_batches, 1)
    return EpochResult(
        loss=loss_sum / nb,
        loss_by_n={n: v / nb for n, v in by_n_sum.items()},
        grad_norm=norm_sum / nb,
        rows_trained=rows_trained,
    )


@torch.no_grad()
def evaluate(
    model, batcher: PoolBatcher, items: list[PoolItem], device, batch_size: int, prefix: str
) -> dict[str, float]:
    """The held-out coverage loss per N and the set metrics of the slots'
    sets (metrics.set_metrics), keyed under `prefix`."""
    model.eval()
    picks: list[list[int]] = []
    by_n_sum: dict[int, float] = {}
    n_batches = 0
    for start in range(0, len(items), batch_size):
        batch = to_device(batcher.build(items[start : start + batch_size]), device)
        logits, pad = slot_forward(model, batch)
        _, by_n = slot_loss(logits, batch)
        picks += slot_picks(logits, pad)
        for n, v in by_n.items():
            by_n_sum[n] = by_n_sum.get(n, 0.0) + v.item()
        n_batches += 1
    order = {id(it.pool): i for i, it in enumerate(items)}

    def choose(pool, n):
        return np.array(picks[order[id(pool)]][:n])

    out = metrics.set_metrics([it.pool for it in items], choose, prefix)
    for n, v in by_n_sum.items():
        out[f"{prefix}eval_loss_{n}"] = v / max(n_batches, 1)
    return out


@torch.no_grad()
def set_build_ms(model, batcher: PoolBatcher, items: list[PoolItem], device) -> float:
    """Milliseconds to build SET_TIMING_POSITIONS sets of SET_TIMING_N from
    inputs already on the device: the forward and the first SET_TIMING_N
    slots' pointers back on the host, the faster of three runs."""
    model.eval()
    batch = to_device(batcher.build(items[:SET_TIMING_POSITIONS]), device)
    best = float("inf")
    for _ in range(3):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        logits, _ = slot_forward(model, batch)
        logits[:, :SET_TIMING_N].float().cpu()
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best

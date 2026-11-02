"""One training epoch of the policy-set model's forms and their held-out
eval (docs/plans/deep_bot_v0.md, "Training the forms together" and "Eval
metrics").

The forms train side by side on the same batches (forms.FORMS): one
optimizer over every form's parameters, the summed loss, each form's
gradient clipped on its own. Each form's forward runs under bf16 autocast;
the losses run in FP32. Record keys carry the form's name: "loss_<form>_<term>" for its
training loss terms, "<form>_eval_..." for its held-out metrics.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from scribblez.generational.finite_guard import FiniteGuard
from scribblez.policy_set import metrics
from scribblez.policy_set.dataset import PoolBatcher, PoolItem, batch_order
from scribblez.policy_set.forms import FORMS

# The share of training positions encoded transposed: the board is symmetric
# under the diagonal transpose, so a transposed position is another, equally
# real one, and half of each batch doubles the positions the model sees. The
# held-out eval reads positions as they were played.
TRANSPOSE_SHARE = 0.5
# Seeds the held-out loss's random parts (the greedy form's chosen sets), so
# generations are compared on the same rows.
EVAL_SEED = 0


def to_device(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def batch_losses(
    models: nn.ModuleDict, batch, params, rng: np.random.Generator
) -> dict[str, dict[str, torch.Tensor]]:
    """Each form's loss terms on `batch`, by form."""
    return {name: FORMS[name].loss(model, batch, params, rng) for name, model in models.items()}


@dataclass
class EpochResult:
    loss: float  # mean summed loss of every form, per batch
    terms: dict[str, float]  # "<form>_<term>" -> its mean per batch
    grad_norms: dict[str, float]  # form -> mean pre-clip norm
    rows_trained: int  # pools


def run_epoch(
    models: nn.ModuleDict,
    optimizer,
    batcher: PoolBatcher,
    items: list[PoolItem],
    device,
    params,
    *,
    seed: int,
    lr_fn: Callable[[int], float] | None,
    rows_trained: int,
    on_batch: Callable[[int, int, float, int], None] | None = None,
) -> EpochResult:
    """One pass over `items` in shuffled batches of `params.batch_size`
    pools. A pool is the rows-clock's unit. Raises WorkerFatal
    (generational.finite_guard) if a step's loss or a form's gradient norm is
    non-finite, by the epoch's end at the latest."""
    models.train()
    rng = np.random.default_rng(seed)
    clip = params.grad_clip if params.grad_clip > 0 else float("inf")
    # Summed on the device and read back once: a per-step .item() would block
    # the host on the step's backward, which the next batch's replay
    # otherwise overlaps.
    sums: dict[str, torch.Tensor] = {}
    norms: dict[str, torch.Tensor] = {}
    n_batches = n_rows = 0
    guard = FiniteGuard()
    t0 = last = time.time()
    for idx in batch_order(len(items), params.batch_size, seed):
        chosen = [items[i] for i in idx]
        transposes = rng.random(len(chosen)) < TRANSPOSE_SHARE
        batch = to_device(batcher.build(chosen, transposes), device)
        rows_trained += len(idx)
        n_rows += len(idx)
        if lr_fn is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr_fn(rows_trained)
        losses = batch_losses(models, batch, params, rng)
        total = sum(v for terms in losses.values() for v in terms.values())
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        step_norms = {}
        for name, model in models.items():
            step_norms[name] = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).detach()
            norms[name] = norms.get(name, 0.0) + step_norms[name]
        guard.observe(
            f"batch {n_batches} ({_batch_names(chosen)})",
            {
                **{f"{name} loss": sum(terms.values()) for name, terms in losses.items()},
                **{f"{name} gradient norm": norm for name, norm in step_norms.items()},
            },
        )
        optimizer.step()
        n_batches += 1
        sums["total"] = sums.get("total", 0.0) + total.detach()
        for name, terms in losses.items():
            for term, v in terms.items():
                key = f"{name}_{term}"
                sums[key] = sums.get(key, 0.0) + v.detach()
        if on_batch is not None and time.time() - last > 1.0:
            guard.check()  # the progress line syncs anyway
            on_batch(n_batches, n_rows, time.time() - t0, rows_trained)
            last = time.time()
    guard.check()
    nb = max(n_batches, 1)
    return EpochResult(
        loss=float(sums.get("total", 0.0)) / nb,
        terms={k: float(v) / nb for k, v in sums.items() if k != "total"},
        grad_norms={k: float(v) / nb for k, v in norms.items()},
        rows_trained=rows_trained,
    )


def _batch_names(items: list[PoolItem]) -> str:
    """A batch's pools as "<.slog stem> game G turn T", for an error."""
    return ", ".join(f"{it.slog.stem} game {it.pool.game} turn {it.pool.turn}" for it in items)


def precomputed_pick(by_pool: dict[int, list[int]], pool, n: int) -> np.ndarray:
    """A metrics.Chooser over sets already built: the first `n` of the pool's
    picks, looked up by the pool's id()."""
    return np.array(by_pool[id(pool)][:n])


@torch.no_grad()
def evaluate(
    models: nn.ModuleDict, batcher: PoolBatcher, items: list[PoolItem], device, params
) -> dict[str, float]:
    """Per form: its held-out loss terms, its diagnostics (forms.Form.diagnose)
    and the set metrics of its sets (metrics.set_metrics), keyed under
    "<form>_eval_"."""
    models.eval()
    rng = np.random.default_rng(EVAL_SEED)
    n_max = max(metrics.SET_SIZES)
    picks: dict[str, list[list[int]]] = {name: [] for name in models}
    sums: dict[str, float] = {}
    n_batches = 0
    for start in range(0, len(items), params.batch_size):
        batch = to_device(batcher.build(items[start : start + params.batch_size]), device)
        for name, terms in batch_losses(models, batch, params, rng).items():
            for term, v in terms.items():
                key = f"{name}_eval_loss_{term}"
                sums[key] = sums.get(key, 0.0) + v.item()
        for name, model in models.items():
            picks[name] += FORMS[name].sets(model, batch, n_max)
            for key, v in FORMS[name].diagnose(model, batch).items():
                sums[f"{name}_eval_{key}"] = sums.get(f"{name}_eval_{key}", 0.0) + v.item()
        n_batches += 1
    out = {k: v / max(n_batches, 1) for k, v in sums.items()}
    pool_list = [it.pool for it in items]
    for name, form_picks in picks.items():
        by_pool = {id(it.pool): p for it, p in zip(items, form_picks, strict=True)}
        chooser = functools.partial(precomputed_pick, by_pool)
        out.update(metrics.set_metrics(pool_list, chooser, f"{name}_"))
    return out

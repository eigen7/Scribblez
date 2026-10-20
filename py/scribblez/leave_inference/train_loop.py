"""One training epoch of the leave inference model, and its held-out eval
(docs/plans/deep_bot_v0.md, "Eval metrics").

Rows without information (turn 0, an empty leave, a pass) carry no loss: the
loader cannot skip them, so each batch masks them out here. The model runs
under bf16 autocast; the tilted-prior NLL (loss.py) runs in full FP32.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from scribblez.ffi import format_layout
from scribblez.leave_inference import loss
from scribblez.leave_inference.autoregressive import sequence_nll
from scribblez.leave_inference.heads import HEAD_AUTOREGRESSIVE

LAYOUT = format_layout()["constants"]["leave_inference"]
CAUSES = LAYOUT["cause"]
# Bag-size bands the eval breaks NLL down by: (name, lowest, highest).
BAG_BANDS = (("1_20", 1, 20), ("21_40", 21, 40), ("41_60", 41, 60), ("61_100", 61, 100))


@dataclass
class Batch:
    """A batch's inputs on the device, and its valid rows' loss inputs."""

    spatial: torch.Tensor
    scalar: torch.Tensor
    valid: torch.Tensor  # (B,) bool
    tiles: torch.Tensor  # (V, 7) sorted, padded with -1; valid rows only
    leave_counts: torch.Tensor  # (V, 27)
    length: torch.Tensor  # (V,)
    pool: torch.Tensor  # (V, 27)


def to_device(batch: dict, device) -> Batch:
    valid = batch["opp_leave_valid"][:, 0] > 0.5
    return Batch(
        spatial=batch["input_spatial"].to(device, non_blocking=True),
        scalar=batch["input_scalar"].to(device, non_blocking=True),
        valid=valid.to(device),
        tiles=batch["opp_leave_tiles"][valid].to(device),
        leave_counts=loss.counts_from_tiles(batch["opp_leave_tiles"][valid]).to(device),
        length=batch["opp_leave_length"][valid, 0].long().to(device),
        pool=batch["leave_pool"][valid].long().to(device),
    )


def batch_nll(model, table: loss.TableTensors, b: Batch) -> loss.LeaveNllResult:
    """The valid rows' NLL under the model's head. The autoregressive head
    has no implied readouts, so its result carries the NLL alone."""
    with torch.autocast(device_type=b.spatial.device.type, dtype=torch.bfloat16):
        out = model(b.spatial, b.scalar)
    v = b.valid
    if model.head_kind == HEAD_AUTOREGRESSIVE:
        nll = sequence_nll(model.head, out["context"][v], b.tiles, b.length, b.pool)
        return loss.LeaveNllResult(nll=nll)
    return loss.tilted_nll(
        table, out["theta"][v], out["a"][v], out["b"][v], b.leave_counts, b.length, b.pool
    )


@dataclass
class EpochResult:
    loss: float  # mean NLL per valid row, nats
    loss_vs_uniform: float
    valid_frac: float
    grad_norm: float  # mean pre-clip global norm
    rows_trained: int


def run_epoch(
    model,
    optimizer,
    table: loss.TableTensors,
    batches: Iterable[dict],
    device,
    *,
    lr_fn: Callable[[int], float] | None,
    rows_trained: int,
    grad_clip: float,
    on_batch: Callable[[int, int, float, int], None] | None = None,
) -> EpochResult:
    """One pass over `batches`. `rows_trained` counts every row, valid or not,
    as the loader plans them; it is the rows-clock the schedule runs on."""
    model.train()
    nll_sum = uniform_sum = norm_sum = 0.0
    n_valid = n_rows = n_batches = 0
    t0 = last = time.time()
    for raw in batches:
        b = to_device(raw, device)
        rows = b.spatial.shape[0]
        n_rows += rows
        rows_trained += rows
        if lr_fn is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr_fn(rows_trained)
        if not b.valid.any():
            continue
        result = batch_nll(model, table, b)
        optimizer.zero_grad(set_to_none=True)
        result.nll.mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip if grad_clip > 0 else float("inf")
        )
        optimizer.step()
        n_batches += 1
        n_valid += len(result.nll)
        nll_sum += result.nll.sum().item()
        uniform_sum += loss.uniform_nll(b.leave_counts, b.length, b.pool).sum().item()
        norm_sum += norm.item()
        if on_batch is not None and time.time() - last > 1.0:
            on_batch(n_batches, n_rows, time.time() - t0, rows_trained)
            last = time.time()
    nv = max(n_valid, 1)
    return EpochResult(
        loss=nll_sum / nv,
        loss_vs_uniform=(nll_sum - uniform_sum) / nv,
        valid_frac=n_valid / max(n_rows, 1),
        grad_norm=norm_sum / max(n_batches, 1),
        rows_trained=rows_trained,
    )


@torch.no_grad()
def evaluate(model, table: loss.TableTensors, batches: Iterable[dict], device) -> dict[str, float]:
    """The held-out metrics over every row of `batches`."""
    model.eval()
    parts: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "nll",
            "uniform",
            "length",
            "bag",
            "presence",
            "true_presence",
            "value_mean",
            "value_sd",
            "true_value",
        )
    }
    cause_counts = {name: 0 for name in CAUSES}
    for raw in batches:
        for name, code in CAUSES.items():
            cause_counts[name] += int((raw["opp_leave_cause"][:, 0] == code).sum())
        b = to_device(raw, device)
        if not b.valid.any():
            continue
        r = batch_nll(model, table, b)
        bag = raw["bag_size"][raw["opp_leave_valid"][:, 0] > 0.5, 0]
        fields = [
            ("nll", r.nll),
            ("uniform", loss.uniform_nll(b.leave_counts, b.length, b.pool)),
            ("length", b.length),
            ("bag", bag),
        ]
        if r.presence is not None:
            fields += [
                ("presence", r.presence),
                ("true_presence", (b.leave_counts > 0).float()),
                ("value_mean", r.value_mean),
                ("value_sd", r.value_var.sqrt()),
                ("true_value", r.true_value),
            ]
        for key, value in fields:
            parts[key].append(value.float().cpu().numpy())
    return eval_metrics({k: np.concatenate(v) for k, v in parts.items() if v}, cause_counts)


def eval_metrics(rows: dict[str, np.ndarray], cause_counts: dict[str, int]) -> dict[str, float]:
    gap = rows["nll"] - rows["uniform"]
    total = max(sum(cause_counts.values()), 1)
    m = {
        "eval_nll": float(rows["nll"].mean()),
        "eval_nll_uniform": float(rows["uniform"].mean()),
        "eval_nll_vs_uniform": float(gap.mean()),
    }
    # The tilted head's implied readouts; the autoregressive head has none.
    if "presence" in rows:
        m["eval_letter_calibration"] = float(
            np.abs(rows["presence"].mean(axis=0) - rows["true_presence"].mean(axis=0)).mean()
        )
        m["eval_value_bias"] = float((rows["value_mean"] - rows["true_value"]).mean())
        m["eval_value_coverage"] = float(
            (np.abs(rows["true_value"] - rows["value_mean"]) <= rows["value_sd"]).mean()
        )
    for k in range(1, LAYOUT["max_leave_length"] + 1):
        sel = rows["length"] == k
        if sel.any():
            m[f"eval_nll_vs_uniform_len_{k}"] = float(gap[sel].mean())
    for name, lo, hi in BAG_BANDS:
        sel = (rows["bag"] >= lo) & (rows["bag"] <= hi)
        if sel.any():
            m[f"eval_nll_vs_uniform_bag_{name}"] = float(gap[sel].mean())
    for name, count in cause_counts.items():
        if name != "valid":
            m[f"eval_invalid_{name}"] = count / total
    return m

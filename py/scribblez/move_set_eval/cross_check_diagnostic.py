"""Measures whether the move set student's distillation error tracks the
cross-check change a move causes. py/scripts/move_set_eval/
crosscheck_delta_diagnostic.py drives it and explains how to read the tables.

Per candidate move, the student-vs-teacher error (move_errors) is tabulated
against two features of the move's cross-check delta (delta_features). Rows
are split by tiles played, because tile count drives both.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import MsetDataset
from .eval import MAX_CANDIDATES_PER_BATCH
from .model import INPUT_KEYS, MOVE_KEYS, win_equity
from .moves import BOARD

TERCILES = ("low", "mid", "high")
ERRORS = ("equity_abs", "wld_kl", "sd_mean_abs", "plane_kl")
FEATURES = ("changed_bits", "hook_letters")
# A tile count with fewer moves than this gets no table row.
MIN_MOVES_PER_ROW = 300


def delta_features(batch: dict) -> dict[str, np.ndarray]:
    """Per-move cross-check delta features from a batch's delta tensors."""
    real = batch["move_cross_mask"].numpy()
    old = batch["move_cross_old_masks"].numpy().astype(np.uint32)
    new = batch["move_cross_new_masks"].numpy().astype(np.uint32)
    # Hooks onto a word's ends are cross-checks for plays along the other
    # axis: a tile hooking a horizontal word lies in a vertical play's cross
    # word, which the engine files under axis 1. A one-tile play forms a word
    # each way, so all of its entries count.
    squares = batch["move_squares"].numpy()
    horizontal = squares[:, 0] // BOARD == squares[:, 1] // BOARD
    one_tile = batch["move_tile_mask"].sum(dim=1).numpy() == 1
    axes = batch["move_cross_axes"].numpy()
    word_end = real & ((axes == horizontal[:, None]) | one_tile[:, None])
    return {
        "changed_bits": (np.bitwise_count(old ^ new) * real).sum(axis=1),
        "hook_letters": (np.bitwise_count(new) * word_end).sum(axis=1),
    }


def move_errors(out: dict, batch: dict, device) -> dict[str, np.ndarray]:
    """Per-move student-vs-teacher error."""
    t_wld = batch["target_wld"].to(device)
    log_pred = F.log_softmax(out["wld"].float(), dim=1)
    errors = {
        "wld_kl": (t_wld * (torch.log(t_wld.clamp_min(1e-9)) - log_pred)).sum(dim=1),
        "equity_abs": (win_equity(log_pred.exp()) - win_equity(t_wld)).abs(),
        "sd_mean_abs": (
            out["score_diff"][:, 0].float() - batch["target_score_diff"][:, 0].to(device)
        ).abs(),
    }
    if "target_planes" in batch:
        t_planes = batch["target_planes"].to(device)
        log_planes = F.log_softmax(out["planes"].float(), dim=-1)
        kl = (t_planes * (torch.log(t_planes.clamp_min(1e-9)) - log_planes)).sum(dim=-1)
        errors["plane_kl"] = kl.mean(dim=1)
    return {k: v.cpu().numpy() for k, v in errors.items()}


def collect(model, dataset: MsetDataset, device, max_positions: int) -> dict[str, np.ndarray]:
    """Per-move errors, delta features and tiles played over the slice."""
    columns: dict[str, list[np.ndarray]] = {}
    positions = 0
    for batch in dataset.iter_batches(64, max_candidates=MAX_CANDIDATES_PER_BATCH):
        with torch.no_grad():
            out = model(*(batch[k].to(device) for k in (*INPUT_KEYS, *MOVE_KEYS)))
        row = {
            **move_errors(out, batch, device),
            **delta_features(batch),
            "tiles": batch["move_tile_mask"].sum(dim=1).numpy(),
            "is_play": batch["move_scalars"][:, 2].numpy() > 0,
        }
        for key, values in row.items():
            columns.setdefault(key, []).append(values)
        positions += batch["input_spatial"].shape[0]
        if positions >= max_positions:
            break
    return {k: np.concatenate(v) for k, v in columns.items()}


def tercile_of(feature: np.ndarray) -> np.ndarray:
    """0/1/2 by rank, so ties at a cut still split into near-equal thirds."""
    ranks = np.argsort(np.argsort(feature, kind="stable"), kind="stable")
    return ranks * 3 // max(len(feature), 1)


def format_table(data: dict[str, np.ndarray], feature: str, error: str) -> str:
    """Mean `error` by tercile of `feature`, within each tile count; each cell
    also shows the tercile's mean feature value."""
    header = " ".join(f"{t:>20}" for t in TERCILES)
    lines = [
        f"{error} by {feature} tercile, within tiles played",
        f"{'tiles':>5} {'moves':>9} {header} {'high/low':>9}",
    ]
    for tiles in range(1, 8):
        rows = data["is_play"] & (data["tiles"] == tiles)
        if rows.sum() < MIN_MOVES_PER_ROW:
            continue
        values, err = data[feature][rows], data[error][rows]
        tercile = tercile_of(values)
        means = [err[tercile == t].mean() for t in range(3)]
        cells = " ".join(
            f"{f'{means[t]:.4f} ({values[tercile == t].mean():6.1f})':>20}" for t in range(3)
        )
        lines.append(f"{tiles:>5} {rows.sum():>9} {cells} {means[2] / means[0]:>9.2f}")
    return "\n".join(lines)

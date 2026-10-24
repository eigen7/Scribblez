#!/usr/bin/env python3
"""Print what the policy-set greedy form's gain targets look like on a
directory of labeled pool chunks, without training anything.

Two tables:
  seed-only rows   each candidate's expected improvement over the seed (the
                   top-equity move), by equity gap: its mean and median, how
                   often it passes 0.5 and 1 point, and the paired difference
                   it comes from with that difference's standard error and
                   rollout count
  training rows    candidates within 1 equity point of the top on rows drawn
                   as subset assembly draws them (greedy.sample_rows), by set
                   size: the mean target and how often such a candidate is
                   already a member
  reliability      split-half reliability of the seed-only targets, by
                   equity gap: the correlation between targets from the even
                   and the odd rollouts, stepped up to the full count
                   (Spearman-Brown) -- the ceiling on how well any model can
                   correlate with the targets

    py/tools/policy_set_gain_targets.py --pools /workspace/mount/scratch-policy-set-pools-v5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scribblez.policy_set import dataset, greedy
from scribblez.policy_set.model import to_grid

BATCH_POOLS = 8
GAP_EDGES = (0, 0.5, 1, 2, 3, 5, 10, float("inf"))
SET_SIZE_BANDS = ((1, 1), (2, 4), (5, 9), (10, 15), (1, 15))
NEAR_TOP_POINTS = 1
TRAINING_ROW_POOLS = 600
TRAINING_ROWS_PER_POOL = 4
RELIABILITY_GAP_BANDS = ((0, 1), (1, 3), (3, 10))
# Pairs with fewer shared rollouts (the race dropped them early) are left out.
RELIABILITY_MIN_ROLLOUTS = 20


def gap_points(batch) -> torch.Tensor:
    """Each candidate's equity points below its pool's best; the equity input
    is stored /10."""
    return -batch["move_equity"].float() * 10


def seed_rows(batch) -> greedy.SetRows:
    pos_id = batch["move_pos_id"]
    num_positions = batch["input_spatial"].shape[0]
    equity, _ = to_grid(batch["move_equity"].float(), pos_id, num_positions, -torch.inf)
    sizes = torch.bincount(pos_id, minlength=num_positions)
    seeds = (torch.cumsum(sizes, 0) - sizes + equity.argmax(dim=1)).tolist()
    return greedy.make_rows(list(range(num_positions)), [[s] for s in seeds])


def seed_only_samples(items) -> dict[str, np.ndarray]:
    batcher = dataset.PoolBatcher()
    out = {k: [] for k in ("gap", "target", "diff", "se", "n")}
    for start in range(0, len(items), BATCH_POOLS):
        batch = batcher.build(items[start : start + BATCH_POOLS])
        rows = seed_rows(batch)
        move_idx, row_id = greedy.row_candidates(batch["move_pos_id"], rows.position)
        seed = rows.chosen[row_id, 0]
        target = greedy.gain_targets(batch, rows, move_idx, row_id)
        diff, se = greedy.paired_difference(
            batch["rollout_values"].float(), batch["rollouts"], move_idx, seed
        )
        keep = move_idx != seed
        out["gap"].append(gap_points(batch)[move_idx][keep])
        out["target"].append(target[keep] * greedy.GAIN_SCALE)
        out["diff"].append(diff[keep] * greedy.GAIN_SCALE)
        out["se"].append(se[keep] * greedy.GAIN_SCALE)
        out["n"].append(torch.minimum(batch["rollouts"][move_idx], batch["rollouts"][seed])[keep])
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def print_seed_only(items):
    d = seed_only_samples(items)
    print(f"seed-only rows: {len(d['target'])} candidates over {len(items)} pools (pp)")
    print(
        f"{'equity gap':>12} {'n':>7} {'mean':>7} {'median':>7} {'>0.5pp':>7} {'>1pp':>6}"
        f" | {'mean diff':>9} {'median se':>9} {'rollouts':>8}"
    )
    for lo, hi in zip(GAP_EDGES[:-1], GAP_EDGES[1:], strict=True):
        k = (d["gap"] >= lo) & (d["gap"] < hi)
        if not k.any():
            continue
        t = d["target"][k]
        print(
            f"{f'[{lo},{hi})':>12} {k.sum():7d} {t.mean():7.3f} {np.median(t):7.3f}"
            f" {(t > 0.5).mean():7.2f} {(t > 1).mean():6.2f}"
            f" | {d['diff'][k].mean():9.2f} {np.median(d['se'][k]):9.2f}"
            f" {np.median(d['n'][k]):8.0f}"
        )


def training_row_samples(items, rng) -> dict[str, np.ndarray]:
    batcher = dataset.PoolBatcher()
    out = {k: [] for k in ("gap", "target", "size", "member")}
    for start in range(0, len(items), BATCH_POOLS):
        batch = batcher.build(items[start : start + BATCH_POOLS])
        position, chosen = greedy.sample_rows(
            batch["win_prob"].numpy(), batch["move_equity"].numpy(),
            batch["move_pos_id"].numpy(), TRAINING_ROWS_PER_POOL, rng,
        )  # fmt: skip
        rows = greedy.make_rows(position, chosen)
        move_idx, row_id = greedy.row_candidates(batch["move_pos_id"], rows.position)
        member = ((rows.chosen[row_id] == move_idx.unsqueeze(1)) & rows.mask[row_id]).any(1)
        out["gap"].append(gap_points(batch)[move_idx])
        out["target"].append(greedy.gain_targets(batch, rows, move_idx, row_id) * greedy.GAIN_SCALE)
        out["size"].append(rows.mask.sum(1)[row_id])
        out["member"].append(member)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def print_training_rows(items):
    d = training_row_samples(items, np.random.default_rng(0))
    # The top-equity move itself (gap 0) is every row's seed.
    near = (d["gap"] > 0) & (d["gap"] < NEAR_TOP_POINTS)
    print(
        f"\ntraining rows over {len(items)} pools: "
        f"candidates within {NEAR_TOP_POINTS} equity point of the top"
    )
    for lo, hi in SET_SIZE_BANDS:
        k = near & (d["size"] >= lo) & (d["size"] <= hi)
        target, member = d["target"][k].mean(), d["member"][k].mean()
        print(
            f"  set size {lo:2d}-{hi:2d}: n {k.sum():5d}  mean target {target:.3f} pp"
            f"   share already in the set {member:.2f}"
        )


def half_targets(batch, move_idx, seed, parity) -> torch.Tensor:
    """Seed-only targets from every other rollout, starting at `parity`."""
    values = batch["rollout_values"].float()[:, parity::2]
    rollouts = (batch["rollouts"] + 1 - parity) // 2
    diff = greedy.paired_difference(values, rollouts, move_idx, seed)
    return greedy.expected_improvement(*diff) * greedy.GAIN_SCALE


def print_reliability(items):
    batcher = dataset.PoolBatcher()
    out = {k: [] for k in ("gap", "even", "odd", "n")}
    for start in range(0, len(items), BATCH_POOLS):
        batch = batcher.build(items[start : start + BATCH_POOLS])
        rows = seed_rows(batch)
        move_idx, row_id = greedy.row_candidates(batch["move_pos_id"], rows.position)
        seed = rows.chosen[row_id, 0]
        keep = move_idx != seed
        out["gap"].append(gap_points(batch)[move_idx][keep])
        out["even"].append(half_targets(batch, move_idx, seed, 0)[keep])
        out["odd"].append(half_targets(batch, move_idx, seed, 1)[keep])
        out["n"].append(torch.minimum(batch["rollouts"][move_idx], batch["rollouts"][seed])[keep])
    d = {k: torch.cat(v).numpy() for k, v in out.items()}
    print(f"\nseed-only target reliability (pairs with {RELIABILITY_MIN_ROLLOUTS}+ rollouts)")
    for lo, hi in RELIABILITY_GAP_BANDS:
        k = (d["gap"] >= lo) & (d["gap"] < hi) & (d["n"] >= RELIABILITY_MIN_ROLLOUTS)
        half = np.corrcoef(d["even"][k], d["odd"][k])[0, 1]
        print(
            f"  gap [{lo},{hi}): n {k.sum():6d}  split-half corr {half:.3f}"
            f"  full-count reliability {2 * half / (1 + half):.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pools", type=Path, required=True, help="a directory of pool chunks")
    args = parser.parse_args()
    chunks = sorted(p for p in args.pools.iterdir() if p.is_dir() and any(p.glob("*.pool")))
    items = dataset.read_items(chunks)
    print_seed_only(items)
    print_training_rows(items[:TRAINING_ROW_POOLS])
    print_reliability(items)


if __name__ == "__main__":
    main()

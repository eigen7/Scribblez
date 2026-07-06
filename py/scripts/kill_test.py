#!/usr/bin/env python3
"""The 4-armed sim-evidence kill-test (docs/sim_residual_feedback.md, step 3).

Tests the load-bearing hypothesis of the sim-evidence loop in isolation: does
conditioning M_post on Monte-Carlo sim evidence improve its outcome
prediction? One invocation runs the whole experiment against the data
accumulated by scripts/generate_kill_test_data.py under the same tag:

  1. Cache build: every complete .slog/.sobs pair under
     <mount>/kill_test/<tag>/slogs is decoded (each evidence position's
     post-move training row, addressed by identity) and its evidence set
     encoded, into one .npz shard per file under <mount>/kill_test/<tag>/cache.
     Shards are cached across invocations; only new files are processed.
     Files alternate into train/holdout round-robin (--holdout-every), so the
     split is by game and cannot leak.

  2. Four training arms, identical seed and architecture (the fusion stage is
     zero-initialized and parameter counts match across arms; only the
     evidence input differs):
       none      evidence zeroed -- the baseline
       shuffled  real evidence permuted across positions -- a falsification
                 control; any gain over `none` is what the model extracts
                 from evidence marginals rather than position-matched
                 evidence, and `full` should be judged against it
       scalar    scalar sim summaries only (the cheap rung of the ladder)
       full      spatial planes + scalar summaries (the real thing)

The decision metric is best held-out WLD cross-entropy; a comparison table
prints at the end and per-arm histories land in
<mount>/kill_test/<tag>/cache/results/. See the doc's kill-test section for
the decision rubric.

Usage:
    ./py/scripts/kill_test.py -t apple
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scribblez.dataset import row_layout
from scribblez.ffi import decode_rows, get_input_shapes
from scribblez.post_move_value.model import MASK_HEAD_NAMES, compute_loss
from scribblez.sim_evidence.model import EvidencePostMoveModel
from scribblez.sim_evidence.sobs import NUM_EVIDENCE_SCALARS, evidence_features, read_sobs

MOUNT_ROOT = Path("/workspace/mount")
TARGET_KEYS = ("wld", "score_diff", *MASK_HEAD_NAMES)
ARMS = ("none", "shuffled", "scalar", "full")


# ---------------------------------------------------------------------------
# cache build
# ---------------------------------------------------------------------------


def build_shard(slog: Path, sobs: Path, max_k: int) -> dict[str, np.ndarray]:
    """Decode one file's evidence positions into a shard's arrays."""
    positions = read_sobs(sobs)
    games = np.array([p.game_index for p in positions], dtype=np.int64)
    turns = np.array([p.turn_index for p in positions], dtype=np.int64)
    rows = decode_rows(slog, games, turns, post_move=True)

    input_shapes, targets = row_layout()
    spatial_shape = input_shapes[0].dims
    scalar_width = input_shapes[1].dims[0]
    spatial_floats = int(np.prod(spatial_shape))

    n = len(positions)
    ev_planes = np.zeros((n, max_k, 5, 15, 15), dtype=np.float16)
    ev_scalars = np.zeros((n, max_k, NUM_EVIDENCE_SCALARS), dtype=np.float32)
    ev_mask = np.zeros((n, max_k), dtype=bool)
    for i, pos in enumerate(positions):
        ev_planes[i], ev_scalars[i], ev_mask[i] = evidence_features(pos, max_k)

    shard = {
        "input_spatial": rows[:, :spatial_floats].reshape(n, *spatial_shape).astype(np.float16),
        "input_scalar": rows[:, spatial_floats : spatial_floats + scalar_width],
        "ev_planes": ev_planes,
        "ev_scalars": ev_scalars,
        "ev_mask": ev_mask,
    }
    for name, start, end, dims in targets:
        shard[name] = rows[:, start:end].reshape(n, *dims)
    return shard


def build_cache(slog_dir: Path, cache: Path, holdout_every: int, max_k: int):
    (cache / "train").mkdir(parents=True, exist_ok=True)
    (cache / "holdout").mkdir(parents=True, exist_ok=True)

    slogs = sorted(slog_dir.glob("*.slog"))
    pairs = [(s, s.with_suffix(".sobs")) for s in slogs if s.with_suffix(".sobs").exists()]
    if not pairs:
        raise SystemExit(
            f"no .slog/.sobs pairs under {slog_dir} (run generate_kill_test_data.py first)"
        )
    print(f"{len(pairs)} .slog/.sobs pairs; holdout = every {holdout_every}th file")

    added = 0
    for i, (slog, sobs) in enumerate(pairs):
        split = "holdout" if i % holdout_every == 0 else "train"
        out = cache / split / f"{slog.stem}.npz"
        if out.exists():
            continue
        shard = build_shard(slog, sobs, max_k)
        np.savez_compressed(out, **shard)
        added += len(shard["ev_mask"])
        print(f"  [{split}] {out.name}: {len(shard['ev_mask'])} positions")
    print(f"cache up to date ({added} positions added)")


# ---------------------------------------------------------------------------
# training arms
# ---------------------------------------------------------------------------


def load_split(cache: Path, split: str) -> dict[str, torch.Tensor]:
    """Concatenate a split's shards into one in-memory tensor dict."""
    shards = sorted((cache / split).glob("*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {cache / split}")
    columns: dict[str, list[np.ndarray]] = {}
    for path in shards:
        with np.load(path) as z:
            for key in z.files:
                columns.setdefault(key, []).append(z[key])
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in columns.items()}


def apply_evidence_mode(data: dict[str, torch.Tensor], mode: str, seed: int):
    """Mutate a split's evidence tensors in place per the arm being trained."""
    if mode == "none":
        data["ev_planes"].zero_()
        data["ev_scalars"].zero_()
        data["ev_mask"].zero_()
    elif mode == "scalar":
        data["ev_planes"].zero_()
    elif mode == "shuffled":
        perm = torch.randperm(len(data["ev_mask"]), generator=torch.Generator().manual_seed(seed))
        for key in ("ev_planes", "ev_scalars", "ev_mask"):
            data[key] = data[key][perm]
    elif mode != "full":
        raise ValueError(f"unknown evidence mode {mode}")


def batch_slices(n: int, batch_size: int, generator: torch.Generator | None):
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for lo in range(0, n, batch_size):
        yield order[lo : lo + batch_size]


def to_device(data: dict[str, torch.Tensor], idx: torch.Tensor, device) -> dict[str, torch.Tensor]:
    out = {}
    for key, tensor in data.items():
        t = tensor[idx].to(device)
        if t.dtype == torch.float16:
            t = t.float()
        out[key] = t
    return out


def forward(model, batch):
    return model(
        batch["input_spatial"],
        batch["input_scalar"],
        batch["ev_planes"],
        batch["ev_scalars"],
        batch["ev_mask"],
    )


@torch.no_grad()
def evaluate(model, data, device, batch_size: int) -> dict[str, float]:
    """Held-out metrics; wld_ce is the kill-test's decision metric."""
    model.eval()
    n = len(data["wld"])
    ce_sum = brier_sum = correct = sd_ae_sum = 0.0
    for idx in batch_slices(n, batch_size, generator=None):
        batch = to_device(data, idx, device)
        out = forward(model, batch)
        target_idx = batch["wld"].argmax(dim=1)
        probs = out["wld"].softmax(dim=1)
        ce_sum += F.cross_entropy(out["wld"], target_idx, reduction="sum").item()
        brier_sum += ((probs - batch["wld"]) ** 2).sum().item()
        correct += (out["wld"].argmax(1) == target_idx).sum().item()
        sd_ae_sum += (out["score_diff"][:, 0] - batch["score_diff"].squeeze(1)).abs().sum().item()
    model.train()
    return {
        "wld_ce": ce_sum / n,
        "brier": brier_sum / n,
        "wld_acc": correct / n,
        "sd_mae": sd_ae_sum / n,
    }


def train_arm(arm: str, cache: Path, args, device) -> dict:
    """Train one arm from scratch and return its result record."""
    torch.manual_seed(args.seed)
    train = load_split(cache, "train")
    holdout = load_split(cache, "holdout")
    apply_evidence_mode(train, arm, seed=args.seed)
    apply_evidence_mode(holdout, arm, seed=args.seed + 1)

    input_shapes = {s.name: s.dims for s in get_input_shapes()}
    model = EvidencePostMoveModel(
        spatial_planes=input_shapes["input_spatial"][0],
        scalar_size=input_shapes["input_scalar"][0],
        trunk_channels=args.trunk_channels,
        num_blocks=args.num_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n=== arm={arm} train={len(train['wld'])} holdout={len(holdout['wld'])} "
        f"params={n_params:,} device={device} ==="
    )

    generator = torch.Generator().manual_seed(args.seed)
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        loss_sum = 0.0
        n_batches = 0
        for idx in batch_slices(len(train["wld"]), args.batch_size, generator):
            batch = to_device(train, idx, device)
            out = forward(model, batch)
            losses = compute_loss(out, {k: batch[k] for k in TARGET_KEYS}, lambda_sd=args.lambda_sd)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            loss_sum += losses["total"].item()
            n_batches += 1

        metrics = evaluate(model, holdout, device, args.batch_size)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss_sum / max(n_batches, 1)
        history.append(metrics)
        print(
            f"epoch {epoch:3d}  train_loss={metrics['train_loss']:.4f}  "
            f"holdout: wld_ce={metrics['wld_ce']:.4f} brier={metrics['brier']:.4f} "
            f"acc={metrics['wld_acc']:.4f} sd_mae={metrics['sd_mae']:.1f}  "
            f"({time.time() - t0:.0f}s)"
        )

    best = min(history, key=lambda m: m["wld_ce"])
    return {"arm": arm, "history": history, "best": best}


def print_summary(results: list[dict]):
    by_arm = {r["arm"]: r["best"] for r in results}
    base_ce = by_arm["none"]["wld_ce"]
    print("\n=== kill-test summary (best held-out epoch per arm) ===")
    print(f"{'arm':10s} {'wld_ce':>8s} {'d_vs_none':>10s} {'brier':>8s} {'acc':>7s} {'epoch':>6s}")
    for arm in ARMS:
        b = by_arm[arm]
        print(
            f"{arm:10s} {b['wld_ce']:8.4f} {b['wld_ce'] - base_ce:+10.4f} "
            f"{b['brier']:8.4f} {b['wld_acc']:7.4f} {b['epoch']:6d}"
        )
    print(
        "\nDecision rubric (docs/sim_residual_feedback.md): the hypothesis survives if\n"
        "`full` beats `none` by a margin that (a) dwarfs seed noise and (b) holds up\n"
        "against `shuffled` (which bounds what evidence marginals alone provide).\n"
        "`scalar` vs `full` locates how much of the win needs the spatial planes."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-t", "--tag", required=True, help="tag used with generate_kill_test_data.py")
    p.add_argument("--holdout-every", type=int, default=10, help="every Nth file is holdout")
    p.add_argument("--max-k", type=int, default=10, help="evidence candidates kept per position")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--lambda-sd",
        type=float,
        default=0.004,
        help="Score-diff loss weight (the production trainer's default), so the "
        "WLD head -- the decision metric -- dominates the objective.",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--trunk-channels", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated subset of arms to run (default: all four)",
    )
    args = p.parse_args()

    root = MOUNT_ROOT / "kill_test" / args.tag
    cache = root / "cache"
    build_cache(root / "slogs", cache, args.holdout_every, args.max_k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = cache / "results"
    results_dir.mkdir(exist_ok=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = []
    for arm in arms:
        record = train_arm(arm, cache, args, device)
        arg_record = {k: v for k, v in vars(args).items() if k != "arms"}
        (results_dir / f"{arm}.json").write_text(
            json.dumps({"args": arg_record, **record}, indent=2) + "\n"
        )
        results.append(record)

    if set(arms) == set(ARMS):
        print_summary(results)
    print(f"per-arm histories: {results_dir}")


if __name__ == "__main__":
    main()

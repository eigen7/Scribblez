#!/usr/bin/env python3
"""Sim-evidence kill-test (docs/sim_residual_feedback.md, roadmap step 3).

Tests the load-bearing hypothesis of the sim-evidence loop in isolation: does
conditioning M_post on Monte-Carlo sim evidence improve its outcome
prediction? Two subcommands:

  build   Pair each .slog file in --slog-dir with its .sobs sidecar (produced
          by target/engine/sim_obs_tool), decode every evidence position's
          post-move training row by identity, encode the evidence set, and
          write one .npz shard per file into --cache-dir. Files are assigned
          round-robin to train/holdout (--holdout-every), so the split is by
          game and cannot leak.

  train   Train one arm on the cached shards and report held-out metrics per
          epoch. --evidence selects the arm:
            full      spatial planes + scalar summaries (the real thing)
            scalar    scalar summaries only (the cheap rung of the ladder)
            none      evidence zeroed -- the baseline; same architecture and
                      parameter count, so the comparison isolates the input
            shuffled  real evidence, permuted across positions -- a
                      falsification control: its gain over `none` measures
                      what the model extracts from evidence *marginals*
                      rather than from position-matched evidence

Results are written to <cache-dir>/results/<tag>.json; the decision metric is
held-out WLD cross-entropy (see the doc's kill-test section).

Usage:
    ./py/scripts/sim_evidence/kill_test.py build --slog-dir D --cache-dir C
    ./py/scripts/sim_evidence/kill_test.py train --cache-dir C --evidence none --tag baseline
    ./py/scripts/sim_evidence/kill_test.py train --cache-dir C --evidence full --tag full
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

TARGET_KEYS = ("wld", "score_diff", *MASK_HEAD_NAMES)


# ---------------------------------------------------------------------------
# build
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


def cmd_build(args):
    cache = Path(args.cache_dir)
    (cache / "train").mkdir(parents=True, exist_ok=True)
    (cache / "holdout").mkdir(parents=True, exist_ok=True)

    slogs = sorted(Path(args.slog_dir).glob("*.slog"))
    pairs = [(s, s.with_suffix(".sobs")) for s in slogs if s.with_suffix(".sobs").exists()]
    if not pairs:
        raise SystemExit(f"no .slog/.sobs pairs in {args.slog_dir} (run sim_obs_tool first)")
    print(f"{len(pairs)} .slog/.sobs pairs; holdout = every {args.holdout_every}th file")

    counts = {"train": 0, "holdout": 0}
    for i, (slog, sobs) in enumerate(pairs):
        split = "holdout" if i % args.holdout_every == 0 else "train"
        out = cache / split / f"{slog.stem}.npz"
        if out.exists():
            continue
        shard = build_shard(slog, sobs, args.max_k)
        np.savez_compressed(out, **shard)
        counts[split] += len(shard["ev_mask"])
        print(f"  [{split}] {out.name}: {len(shard['ev_mask'])} positions")
    print(f"done: {counts['train']} train / {counts['holdout']} holdout positions added")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def load_split(cache: Path, split: str) -> dict[str, torch.Tensor]:
    """Concatenate a split's shards into one in-memory tensor dict."""
    shards = sorted((cache / split).glob("*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {cache / split} (run `build` first)")
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


def cmd_train(args):
    cache = Path(args.cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train = load_split(cache, "train")
    holdout = load_split(cache, "holdout")
    apply_evidence_mode(train, args.evidence, seed=args.seed)
    apply_evidence_mode(holdout, args.evidence, seed=args.seed + 1)

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
        f"arm={args.evidence} train={len(train['wld'])} holdout={len(holdout['wld'])} "
        f"params={n_params:,} device={device}"
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
    results_dir = cache / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{args.tag}.json"
    arg_record = {k: v for k, v in vars(args).items() if k not in ("fn", "command")}
    out_path.write_text(
        json.dumps({"args": arg_record, "history": history, "best": best}, indent=2) + "\n"
    )
    print(f"best holdout wld_ce={best['wld_ce']:.4f} (epoch {best['epoch']}) -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="decode rows + evidence into .npz shards")
    b.add_argument("--slog-dir", required=True, help="directory of .slog + .sobs pairs")
    b.add_argument("--cache-dir", required=True, help="output shard directory")
    b.add_argument("--holdout-every", type=int, default=10, help="every Nth file is holdout")
    b.add_argument("--max-k", type=int, default=10, help="evidence candidates kept per position")
    b.set_defaults(fn=cmd_build)

    t = sub.add_parser("train", help="train one arm and report holdout metrics")
    t.add_argument("--cache-dir", required=True)
    t.add_argument("--evidence", choices=["full", "scalar", "none", "shuffled"], required=True)
    t.add_argument("--tag", required=True, help="results filename stem")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument(
        "--lambda-sd",
        type=float,
        default=0.004,
        help="Score-diff loss weight (the production trainer's default), so the "
        "WLD head -- the decision metric -- dominates the objective.",
    )
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--trunk-channels", type=int, default=96)
    t.add_argument("--num-blocks", type=int, default=6)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

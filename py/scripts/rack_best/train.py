#!/usr/bin/env python3
"""Train the rack-best toy: predict the longest word formable from a 7-tile rack.

This is a step up from word-validity: the rack is an unordered multiset, so the
task needs anagram/subset search, which the ordered DAWG-walk tools cannot do
directly. Held-out accuracy measures whether a tool lets the network search the
lexicon. The `anagram` module is the hypothesis built for exactly this.

Usage:
    python -m scripts.rack_best.train --lexicon-module anagram
    python -m scripts.rack_best.train --lexicon-module none           # baseline
    python -m scripts.rack_best.train --lexicon-module soft_traversal  # walk tool (expected weak)
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scribblez.max_move_per_lane.lexicon_compiler import compile_kwg, default_kwg_path
from scribblez.max_move_per_lane.lexicon_modules import (
    available_modules,
    build_lexicon_module,
    parse_module_opts,
    resolve_lane_ffn_mult,
)
from scribblez.rack_best.data import make_dataset
from scribblez.rack_best.model import RackBestModel, encode_racks, onehot_racks

CACHE_DIR = Path("/workspace/mount/cache/rack_best")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the rack-best toy (longest word in a 7-tile rack).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--real-lexicon", default=default_kwg_path(), help="Real .kwg for labels + tool."
    )
    p.add_argument("--num-racks", type=int, default=300_000, help="Distinct racks to sample.")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="Stop after this many steps (0=off).")
    p.add_argument("--eval-every", type=int, default=0, help="Eval every N steps (0=per epoch).")
    p.add_argument(
        "--lexicon-module",
        default="none",
        choices=available_modules(),
        help="Frozen compiled-lexicon tool (compiled from --real-lexicon).",
    )
    p.add_argument("--lexicon-opt", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--lexicon-mode", default="replace", choices=["add", "replace"])
    p.add_argument("--lexicon-replace-ffn-mult", type=int, default=1)
    p.add_argument("--lexicon-starve-ffn", action="store_true")
    return p


def load_dataset(args):
    """Encoded racks (N, 7) and longest-length labels (N,), cached on disk since
    label generation (anagram search per rack) is the slow part."""
    key = hashlib.sha256(
        f"{compile_kwg(args.real_lexicon).source_hash}:{args.num_racks}:{args.seed}".encode()
    ).hexdigest()[:16]
    cache = CACHE_DIR / f"{key}.npz"
    if cache.exists():
        data = np.load(cache)
        return torch.from_numpy(data["enc"]), torch.from_numpy(data["labels"])
    racks, labels = make_dataset(args.real_lexicon, args.num_racks, args.seed)
    enc = encode_racks(racks)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, enc=enc.numpy(), labels=labels)
    return enc, torch.from_numpy(labels)


def accuracy(model, enc, labels, idx, device, batch_size) -> dict:
    """Exact and within-1 longest-length accuracy over `idx`."""
    model.eval()
    exact = within1 = 0
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            b = idx[start : start + batch_size]
            pred = model(onehot_racks(enc[b].to(device))).argmax(-1).cpu()
            y = labels[b]
            exact += (pred == y).sum().item()
            within1 += (pred - y).abs().le(1).sum().item()
    model.train()
    return {"exact": exact / len(idx), "within1": within1 / len(idx)}


def main() -> int:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    enc, labels = load_dataset(args)
    perm = rng.permutation(len(enc))
    n_hold = int(len(enc) * args.holdout_frac)
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    dist = np.bincount(labels.numpy(), minlength=8)
    print(f"racks: {len(enc):,}  train: {len(train_idx):,}  holdout: {n_hold:,}")
    print(f"label distribution (longest length 0..7): {dist.tolist()}")

    lexicon_module = build_lexicon_module(
        args.lexicon_module,
        channels=args.channels,
        kwg_path=args.real_lexicon,
        **parse_module_opts(args.lexicon_opt),
    )
    lane_ffn_mult = resolve_lane_ffn_mult(
        args.lexicon_mode,
        lexicon_module is not None,
        args.lexicon_starve_ffn,
        args.lexicon_replace_ffn_mult,
    )
    model = RackBestModel(
        channels=args.channels,
        n_layers=args.layers,
        n_heads=args.heads,
        ffn_mult=args.ffn_mult,
        lexicon_module=lexicon_module,
        lane_ffn_mult=lane_ffn_mult,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"module={args.lexicon_module} mode={args.lexicon_mode} lane_ffn_mult={lane_ffn_mult}  "
        f"params={n_params:,}"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    for epoch in range(args.epochs):
        order = train_idx[rng.permutation(len(train_idx))]
        for start in range(0, len(order), args.batch_size):
            b = order[start : start + args.batch_size]
            logits = model(onehot_racks(enc[b].to(device)))
            loss = F.cross_entropy(logits, labels[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            if args.eval_every and step % args.eval_every == 0:
                h = accuracy(model, enc, labels, hold_idx, device, args.batch_size)
                print(
                    f"step {step:>6}  loss {loss.item():.4f}  "
                    f"holdout exact {h['exact']:.3f}  within1 {h['within1']:.3f}"
                )
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

        tr = accuracy(model, enc, labels, train_idx[:20000], device, args.batch_size)
        h = accuracy(model, enc, labels, hold_idx, device, args.batch_size)
        print(
            f"epoch {epoch:>2}  train exact {tr['exact']:.3f}  "
            f"holdout exact {h['exact']:.3f}  within1 {h['within1']:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

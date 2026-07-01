#!/usr/bin/env python3
"""Train the rack-best toy: emit a valid longest word from a 7-tile rack.

A decoder-only transformer generates the word letter by letter, constrained at
each step by the frozen forward DAWG (valid word prefixes) and the rack
(available tiles). With the constraint on, every complete decode is a valid
rack-word, so the network only learns to reach the maximal length; with it off
(--no-dawg), the decoder must have learned the lexicon itself and produces
non-words on held-out racks. Held-out valid-longest accuracy is the measurement.

Usage:
    python -m scripts.rack_best.train                 # forward-DAWG constrained
    python -m scripts.rack_best.train --no-dawg       # baseline (no lexicon tool)
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scribblez.max_move_per_lane.lexicon_compiler import compile_kwg, default_kwg_path
from scribblez.rack_best.data import make_dataset
from scribblez.rack_best.model import PAD, RackWordModel, encode_racks, encode_targets

CACHE_DIR = Path("/workspace/mount/cache/rack_best")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train rack-best ordered longest-word generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real-lexicon", default=default_kwg_path())
    p.add_argument("--num-racks", type=int, default=300_000)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--no-dawg", action="store_true", help="Disable the forward-DAWG constraint.")
    return p


def load_dataset(args):
    """Cached (rack (N,7), gen_in (N,8), target (N,8), max_len (N,))."""
    key = hashlib.sha256(
        f"gen:{compile_kwg(args.real_lexicon).source_hash}:{args.num_racks}:{args.seed}".encode()
    ).hexdigest()[:16]
    cache = CACHE_DIR / f"{key}.npz"
    if cache.exists():
        d = np.load(cache)
        return (
            torch.from_numpy(d["rack"]),
            torch.from_numpy(d["gen_in"]),
            torch.from_numpy(d["target"]),
            torch.from_numpy(d["max_len"]),
        )
    racks, words, max_len = make_dataset(args.real_lexicon, args.num_racks, args.seed)
    rack = encode_racks(racks)
    gen_in, target = encode_targets(words)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache, rack=rack.numpy(), gen_in=gen_in.numpy(), target=target.numpy(), max_len=max_len
    )
    return rack, gen_in, target, torch.from_numpy(max_len)


def evaluate(model, compiled, rack, target, max_len, idx, device, batch_size) -> dict:
    """Greedy-decode rates: `valid` (a real formable word, any length), the tool's
    clean guarantee; `valid_longest` (also of max length); `exact` (== canonical)."""
    valid = valid_longest = exact = 0
    for start in range(0, len(idx), batch_size):
        b = idx[start : start + batch_size]
        decoded = model.greedy(rack[b].to(device))
        for row, tgt_row, ml, rk in zip(decoded, target[b], max_len[b], rack[b], strict=True):
            word = "".join(chr(ord("A") + x) for x in row)
            canonical = "".join(chr(ord("A") + int(s)) for s in tgt_row if int(s) < 26)
            rack_letters = [int(x) for x in rk]
            formable = all(row.count(c) <= rack_letters.count(c) for c in set(row))
            is_valid = bool(word) and compiled.contains(word) and formable
            valid += is_valid
            valid_longest += is_valid and len(word) == int(ml)
            exact += word == canonical
    n = len(idx)
    return {"valid": valid / n, "valid_longest": valid_longest / n, "exact": exact / n}


def main() -> int:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    rack, gen_in, target, max_len = load_dataset(args)
    perm = rng.permutation(len(rack))
    n_hold = int(len(rack) * args.holdout_frac)
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    print(f"racks: {len(rack):,}  train: {len(train_idx):,}  holdout: {n_hold:,}")
    print(
        f"max-length distribution (2..7): {np.bincount(max_len.numpy(), minlength=8)[2:].tolist()}"
    )

    compiled = compile_kwg(args.real_lexicon)
    model = RackWordModel(
        compiled,
        channels=args.channels,
        n_layers=args.layers,
        n_heads=args.heads,
        ffn_mult=args.ffn_mult,
        use_dawg=not args.no_dawg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"use_dawg={not args.no_dawg}  params={n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def report(tag, step):
        m = evaluate(model, compiled, rack, target, max_len, hold_idx, device, args.batch_size)
        print(
            f"{tag}  holdout valid {m['valid']:.3f}  valid-longest {m['valid_longest']:.3f}  "
            f"exact {m['exact']:.3f}"
        )

    step = 0
    for epoch in range(args.epochs):
        order = train_idx[rng.permutation(len(train_idx))]
        for start in range(0, len(order), args.batch_size):
            b = order[start : start + args.batch_size]
            logits = model(rack[b].to(device), gen_in[b].to(device), target[b].to(device))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target[b].reshape(-1).to(device),
                ignore_index=PAD,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if args.eval_every and step % args.eval_every == 0:
                report(f"step {step:>6}  loss {loss.item():.4f}", step)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        report(f"epoch {epoch:>2}", step)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train the word-validity toy and report held-out accuracy.

Shuffles a real lexicon (label 1) with its phony counterpart (label 0), holds
out a random split, and trains a small transformer -- optionally with a frozen
compiled-lexicon tool -- to classify words valid/invalid. Because the phonies
match the real lexicon's letter statistics, held-out accuracy is the tool-use
test: a model with no tool can only memorize and sits near chance on held-out
words, while one that learns to use the tool generalizes.

Usage:
    ./py/scripts/word_validity/train.py              # baseline (no tool)
See --help for the compiled-lexicon-tool flags, and docs/word_validity_experiments.md
for the tool-vs-starve protocol.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scribblez.lexical_tool.compiler import compile_kwg, default_kwg_path
from scribblez.lexical_tool.modules import LexiconArgs
from scribblez.word_validity.model import WordValidityModel, encode_words, onehot_batch
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the word-validity toy (real vs phony lexicon).",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real-lexicon", default=default_kwg_path(), help="Real .kwg (label 1).")
    p.add_argument(
        "--phony-lexicon", default=default_kwg_path("PHONY-NWL23"), help="Phony .kwg (label 0)."
    )
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--max-len", type=int, default=15)
    p.add_argument("--holdout-frac", type=float, default=0.1, help="Random held-out fraction.")
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
    LexiconArgs.add_arguments(p)
    return p


def load_split(args, rng):
    """Encoded words, labels, and (train_idx, holdout_idx) for the shuffled mix."""

    def words_in_range(path):
        return [w for w in compile_kwg(path).words() if args.min_len <= len(w) <= args.max_len]

    real = words_in_range(args.real_lexicon)
    phony = words_in_range(args.phony_lexicon)
    words = real + phony
    labels = torch.tensor([1.0] * len(real) + [0.0] * len(phony))
    enc, lengths = encode_words(words)

    perm = rng.permutation(len(words))
    n_hold = int(len(words) * args.holdout_frac)
    print(f"real: {len(real):,}  phony: {len(phony):,}", end="  ")
    print(f"train: {len(words) - n_hold:,}  holdout: {n_hold:,}")
    return enc, lengths, labels, perm[n_hold:], perm[:n_hold]


def accuracy(model, enc, lengths, labels, idx, device, batch_size) -> dict:
    """Accuracy over `idx`, overall and split by true label (real / phony)."""
    model.eval()
    correct = torch.zeros(2)  # [phony(0), real(1)]
    total = torch.zeros(2)
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            b = idx[start : start + batch_size]
            oh = onehot_batch(enc[b].to(device), lengths[b].to(device))
            pred = (model(oh, lengths[b].to(device)) > 0).float().cpu()
            y = labels[b]
            for cls in (0, 1):
                m = y == cls
                total[cls] += m.sum()
                correct[cls] += ((pred == y) & m).sum()
    model.train()
    overall = (correct.sum() / total.sum().clamp_min(1)).item()
    return {
        "acc": overall,
        "real": (correct[1] / total[1].clamp_min(1)).item(),
        "phony": (correct[0] / total[0].clamp_min(1)).item(),
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    enc, lengths, labels, train_idx, hold_idx = load_split(args, rng)

    lex = LexiconArgs.from_args(args)
    lexicon_module = lex.build(channels=args.channels, kwg_path=args.real_lexicon)
    lane_ffn_mult = lex.lane_ffn_mult(lexicon_module is not None)
    model = WordValidityModel(
        channels=args.channels,
        n_layers=args.layers,
        n_heads=args.heads,
        ffn_mult=args.ffn_mult,
        lexicon_module=lexicon_module,
        lane_ffn_mult=lane_ffn_mult,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"module={args.lexicon_module} mode={args.lexicon_mode} lane_ffn_mult={lane_ffn_mult}"
    print(f"{tag}  params={n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    for epoch in range(args.epochs):
        order = train_idx[rng.permutation(len(train_idx))]
        for start in range(0, len(order), args.batch_size):
            b = order[start : start + args.batch_size]
            oh = onehot_batch(enc[b].to(device), lengths[b].to(device))
            logit = model(oh, lengths[b].to(device))
            loss = F.binary_cross_entropy_with_logits(logit, labels[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            if args.eval_every and step % args.eval_every == 0:
                h = accuracy(model, enc, lengths, labels, hold_idx, device, args.batch_size)
                print(
                    f"step {step:>6}  loss {loss.item():.4f}  "
                    f"holdout {h['acc']:.3f} (real {h['real']:.3f} / phony {h['phony']:.3f})"
                )
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

        tr = accuracy(model, enc, lengths, labels, train_idx[:20000], device, args.batch_size)
        h = accuracy(model, enc, lengths, labels, hold_idx, device, args.batch_size)
        print(
            f"epoch {epoch:>2}  train {tr['acc']:.3f}  "
            f"holdout {h['acc']:.3f} (real {h['real']:.3f} / phony {h['phony']:.3f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

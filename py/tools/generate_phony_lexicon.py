#!/usr/bin/env python3
"""Generate a phony lexicon that mimics a real one's letter statistics.

The phony lexicon is a same-size set of plausible-but-invalid "words" -- the kind
a discriminator must actually look up to reject (``YOP`` looks real, ``QVF`` does
not). It pairs with a real lexicon for the word-validity toy: shuffle the two
together with valid/invalid labels and see whether a model can separate them.
Because the phonies match the real lexicon's character statistics, surface
features cannot separate them -- only true lexical membership can.

Method:
  * Fit a character Markov model (default order 3, with start/end padding so
    short words generate too) on the real words.
  * For each length L, aim for the SAME count as the real lexicon has at L:
      - short L (the whole 26^L space is small): enumerate every length-L string,
        drop the real words, and keep the most word-like by model log-likelihood
        -- this fills the short tail exactly, where blind sampling would starve;
      - long L: sample fixed-length strings from the model, dropping real words
        and duplicates, until the target is met or a give-up bound is hit.
  * Reject any string that is a real word; never repeat a phony.

Writes the phony words to a ``.txt`` (one per line) and a real ``.kwg`` (via the
DAWG writer), and prints a per-length target-vs-produced report.
"""

import argparse
import heapq
import math
import random
import string
from collections import Counter, defaultdict
from itertools import product

from scribblez.lexical_tool.compiler import (
    CompiledLexicon,
    compile_kwg,
    default_kwg_path,
    write_kwg,
)
from util.argparse_ext import ArgumentDefaultsHelpFormatter

LETTERS = string.ascii_uppercase
START, END = "^", "$"


class CharMarkov:
    """An order-k character model with start/end padding (so 2-letter words and
    word boundaries are modeled), used both to score and to sample strings."""

    def __init__(self, order: int, alpha: float = 0.1):
        self.order = order
        self.alpha = alpha
        self.counts: dict = defaultdict(lambda: defaultdict(int))
        self._score: dict = {}  # ctx -> {sym: log P(sym | ctx)} over letters + END
        self._sample: dict = {}  # ctx -> (letters, weights) over letters only

    def fit(self, words):
        pad = (START,) * self.order
        for w in words:
            seq = pad + tuple(w) + (END,)
            for i in range(self.order, len(seq)):
                self.counts[seq[i - self.order : i]][seq[i]] += 1

    def _score_dist(self, ctx):
        d = self._score.get(ctx)
        if d is None:
            c = self.counts.get(ctx, {})
            syms = (*LETTERS, END)
            total = sum(c.get(s, 0) for s in syms) + self.alpha * len(syms)
            d = {s: math.log((c.get(s, 0) + self.alpha) / total) for s in syms}
            self._score[ctx] = d
        return d

    def _sample_dist(self, ctx):
        sd = self._sample.get(ctx)
        if sd is None:
            c = self.counts.get(ctx, {})
            weights = [c.get(s, 0) + self.alpha for s in LETTERS]
            self._sample[ctx] = sd = (LETTERS, weights)
        return sd

    def log_likelihood(self, word: str) -> float:
        """Log P(word) as a complete word -- includes the terminating END, so a
        string that rarely ends at this length scores lower."""
        seq = (START,) * self.order + tuple(word) + (END,)
        lp = 0.0
        for i in range(self.order, len(seq)):
            lp += self._score_dist(seq[i - self.order : i])[seq[i]]
        return lp

    def sample_fixed(self, length: int, rng: random.Random) -> str:
        ctx = (START,) * self.order
        out = []
        for _ in range(length):
            letters, weights = self._sample_dist(ctx)
            ch = rng.choices(letters, weights=weights, k=1)[0]
            out.append(ch)
            ctx = ctx[1:] + (ch,)
        return "".join(out)


def phonies_for_length(
    length, target, real, model, rng, enum_max_combos, sample_attempts, giveup_misses
):
    """Up to `target` distinct length-`length` non-words, matched to the model.

    Enumerate-and-rank when the whole space is small enough (fills the short tail
    exactly); otherwise sample with a give-up bound."""
    if 26**length <= enum_max_combos:
        pool = [w for w in map("".join, product(LETTERS, repeat=length)) if w not in real]
        return heapq.nlargest(target, pool, key=model.log_likelihood)

    chosen, seen, attempts, misses = [], set(), 0, 0
    max_attempts = target * sample_attempts
    while len(chosen) < target and attempts < max_attempts and misses < giveup_misses:
        w = model.sample_fixed(length, rng)
        attempts += 1
        if w in real or w in seen:
            misses += 1
            continue
        seen.add(w)
        chosen.append(w)
        misses = 0
    return chosen


def generate(args) -> list[str]:
    real = set(compile_kwg(args.real_lexicon).words())
    in_range = [w for w in real if args.min_len <= len(w) <= args.max_len]
    targets = Counter(len(w) for w in in_range)

    model = CharMarkov(order=args.order)
    model.fit(in_range)
    rng = random.Random(args.seed)

    phonies, report = [], []
    for length in range(args.min_len, args.max_len + 1):
        target = targets.get(length, 0)
        if target == 0:
            continue
        got = phonies_for_length(
            length,
            target,
            real,
            model,
            rng,
            args.enum_max_combos,
            args.sample_attempts,
            args.giveup_misses,
        )
        phonies.extend(got)
        mode = "enum" if 26**length <= args.enum_max_combos else "sample"
        report.append((length, target, len(got), mode))

    print(f"{'len':>4} {'target':>8} {'produced':>9} {'mode':>7}  {'shortfall':>9}")
    short_total = 0
    for length, target, produced, mode in report:
        short = target - produced
        short_total += short
        flag = "" if short == 0 else f"  (-{short})"
        print(f"{length:>4} {target:>8} {produced:>9} {mode:>7}  {short:>9}{flag}")
    print(f"\nreal (in range): {len(in_range):,}   phony: {len(phonies):,}")
    print(f"total shortfall: {short_total:,}")
    return sorted(phonies, key=lambda w: (len(w), w))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--real-lexicon", default=default_kwg_path(), help="Real .kwg to mimic.")
    p.add_argument("--out-txt", default="NWL23_phony.txt", help="Phony word list output.")
    p.add_argument("--out-kwg", default="NWL23_phony.kwg", help="Phony .kwg output.")
    p.add_argument("--order", type=int, default=3, help="Character-model order (context length).")
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--max-len", type=int, default=15)
    p.add_argument(
        "--enum-max-combos",
        type=int,
        default=2_000_000,
        help="Enumerate-and-rank lengths whose 26^L is at or below this (else sample).",
    )
    p.add_argument("--sample-attempts", type=int, default=200, help="Max samples per wanted word.")
    p.add_argument(
        "--giveup-misses",
        type=int,
        default=20_000,
        help="Give up a length after this many consecutive sampling misses.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    phonies = generate(args)
    with open(args.out_txt, "w") as f:
        f.write("\n".join(phonies) + "\n")
    write_kwg(CompiledLexicon.from_words(phonies), args.out_kwg)
    print(f"\nwrote {args.out_txt} and {args.out_kwg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

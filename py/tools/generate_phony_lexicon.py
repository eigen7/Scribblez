#!/usr/bin/env python3
"""Generate a phony lexicon: non-words with a real lexicon's letter statistics.

Phonies are plausible non-words, the kind you can only reject by knowing the
word list (``YOP`` looks real, ``QVF`` does not). Mixed with the real lexicon,
they give the word-validity toy task a dataset in which surface features carry
no signal, so a model can only separate the two by learning membership.

A character Markov model is fit to the real words. For each length the tool
produces as many phonies as the real lexicon has words of that length:

  * Short lengths, where all 26^L strings fit under --enum-max-combos, are
    enumerated and the most word-like non-words kept. Sampling would starve
    here, because most likely strings at these lengths are real words.
  * Longer lengths are sampled from the model until the count is met or the
    give-up bounds are hit.

Writes the phonies as a ``.txt`` (one per line) and a ``.kwg``, and prints a
per-length report of any shortfall. The committed phonies/ lexicon was made with

    py/tools/generate_phony_lexicon.py --real-lexicon /workspace/mount/lexica/NWL23.kwg \\
        --out-txt phonies/PHONY-NWL23.txt --out-kwg phonies/PHONY-NWL23.kwg
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
    """Order-k character model with add-alpha smoothing. Words are padded with
    START and END symbols so the model also learns where words begin and end."""

    def __init__(self, order: int, alpha: float = 0.1):
        self.order = order
        self.alpha = alpha
        self.counts: dict = defaultdict(lambda: defaultdict(int))
        # Lazily built per-context distributions. Scoring includes END so that a
        # string which rarely ends at its length scores low; sampling excludes
        # END because it draws strings of a fixed length.
        self._score: dict = {}  # ctx -> {sym: log P(sym | ctx)}
        self._sample: dict = {}  # ctx -> (letters, weights)

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
        """Log-probability of `word` as a complete word, END included."""
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
    """Return up to `target` distinct non-words of the given length."""
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
    p.add_argument("--min-len", type=int, default=2, help="Shortest word length to generate.")
    p.add_argument("--max-len", type=int, default=15, help="Longest word length to generate.")
    p.add_argument(
        "--enum-max-combos",
        type=int,
        default=2_000_000,
        help="Enumerate every string of length L when 26^L is at most this; sample otherwise.",
    )
    p.add_argument(
        "--sample-attempts", type=int, default=200, help="Sampling budget per wanted word."
    )
    p.add_argument(
        "--giveup-misses",
        type=int,
        default=20_000,
        help="Give up a length after this many consecutive sampling misses.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    args = p.parse_args()

    phonies = generate(args)
    with open(args.out_txt, "w") as f:
        f.write("\n".join(phonies) + "\n")
    write_kwg(CompiledLexicon.from_words(phonies), args.out_kwg)
    print(f"\nwrote {args.out_txt} and {args.out_kwg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

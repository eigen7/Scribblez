#!/usr/bin/env python3
"""Generate a phony lexicon: non-words with a real lexicon's letter statistics.

Phonies are plausible non-words, the kind you can only reject by knowing the
word list (``YOP`` looks real, ``QVF`` does not). Mixed with the real lexicon,
they give the word-validity toy task a dataset in which surface features carry
no signal, so a model can only separate the two by learning membership.

A character Markov model with an end-of-word symbol is fit to the real words.
For each length the tool produces as many phonies as the real lexicon has words
of that length:

  * Short lengths, where all 26^L strings fit under --enum-max-combos, are
    enumerated and the most word-like non-words kept. Sampling would starve
    here, because most likely strings at these lengths are real words.
  * Longer lengths share one sampling loop. The model draws words of whatever
    length it likes, ending each at a sampled end-of-word symbol so that word
    endings look like real ones; draws are bucketed by length until every
    bucket meets its count or the give-up bounds are hit, and each bucket is
    then randomly trimmed to its count.

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
        self._dists: dict = {}  # ctx -> (symbols, weights, {sym: log P(sym | ctx)})

    def fit(self, words):
        pad = (START,) * self.order
        for w in words:
            seq = pad + tuple(w) + (END,)
            for i in range(self.order, len(seq)):
                self.counts[seq[i - self.order : i]][seq[i]] += 1

    def _dist(self, ctx):
        d = self._dists.get(ctx)
        if d is None:
            c = self.counts.get(ctx, {})
            syms = (*LETTERS, END)
            weights = [c.get(s, 0) + self.alpha for s in syms]
            total = sum(weights)
            logp = {s: math.log(w / total) for s, w in zip(syms, weights, strict=True)}
            self._dists[ctx] = d = (syms, weights, logp)
        return d

    def log_likelihood(self, word: str) -> float:
        """Log-probability of `word` as a complete word, END included."""
        seq = (START,) * self.order + tuple(word) + (END,)
        return sum(
            self._dist(seq[i - self.order : i])[2][seq[i]] for i in range(self.order, len(seq))
        )

    def sample(self, max_len: int, rng: random.Random) -> str | None:
        """Draw a word, ending it where the model draws END. Returns None once the
        word outgrows `max_len`."""
        ctx = (START,) * self.order
        out = []
        while len(out) <= max_len:
            syms, weights, _ = self._dist(ctx)
            ch = rng.choices(syms, weights=weights, k=1)[0]
            if ch == END:
                return "".join(out)
            out.append(ch)
            ctx = ctx[1:] + (ch,)
        return None


def enumerated_phonies(length, target, real, model):
    """Return the `target` most word-like non-words of the given length."""
    pool = [w for w in map("".join, product(LETTERS, repeat=length)) if w not in real]
    return heapq.nlargest(target, pool, key=model.log_likelihood)


def sampled_phonies(targets, real, model, rng, sample_attempts, giveup_misses):
    """Sample free-length words until each length in `targets` has at least its
    count of distinct non-words, then randomly trim each length down to its count.
    Gives up after `sample_attempts` draws per wanted word, or after
    `giveup_misses` consecutive draws that land in no unfilled length."""
    max_len = max(targets)
    buckets = {length: set() for length in sorted(targets)}
    unfilled = set(targets)
    attempts, misses = 0, 0
    max_attempts = sum(targets.values()) * sample_attempts
    while unfilled and attempts < max_attempts and misses < giveup_misses:
        w = model.sample(max_len, rng)
        attempts += 1
        if w is None or len(w) not in buckets or w in real or w in buckets[len(w)]:
            misses += 1
            continue
        buckets[len(w)].add(w)
        if len(w) not in unfilled:
            misses += 1
            continue
        misses = 0
        if len(buckets[len(w)]) == targets[len(w)]:
            unfilled.discard(len(w))
    return {
        length: rng.sample(sorted(words), min(len(words), targets[length]))
        for length, words in buckets.items()
    }


def generate(args) -> list[str]:
    real = set(compile_kwg(args.real_lexicon).words())
    in_range = [w for w in real if args.min_len <= len(w) <= args.max_len]
    targets = Counter(len(w) for w in in_range)

    model = CharMarkov(order=args.order)
    model.fit(in_range)
    rng = random.Random(args.seed)

    enum_lengths = {L for L in targets if 26**L <= args.enum_max_combos}
    produced = {L: enumerated_phonies(L, targets[L], real, model) for L in enum_lengths}
    produced |= sampled_phonies(
        {L: t for L, t in targets.items() if L not in enum_lengths},
        real,
        model,
        rng,
        args.sample_attempts,
        args.giveup_misses,
    )
    phonies = [w for words in produced.values() for w in words]
    report = [
        (L, targets[L], len(produced[L]), "enum" if L in enum_lengths else "sample")
        for L in sorted(targets)
    ]

    print(f"{'len':>4} {'target':>8} {'produced':>9} {'mode':>7}  {'shortfall':>9}")
    short_total = 0
    for length, target, got, mode in report:
        short = target - got
        short_total += short
        flag = "" if short == 0 else f"  (-{short})"
        print(f"{length:>4} {target:>8} {got:>9} {mode:>7}  {short:>9}{flag}")
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
        help="Give up sampling after this many consecutive draws that fill no unfilled length.",
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

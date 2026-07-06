#!/usr/bin/env python3
"""Inspect slices of a lexicon -- real or phony -- by pattern, length, or sample.

Reads a ``.kwg`` (decoded to its word list) or a plain ``.txt`` (one word per
line) and prints the words matching some filters, or a length histogram. Handy
for eyeing how plausible a generated phony lexicon looks next to the real one.

Pattern syntax (anchored, case-insensitive): a letter matches itself, ``.``
matches any one letter, ``*`` matches any run. So ``A..`` is every 3-letter word
starting with A, ``*ING`` every word ending in ING, ``Q.`` the 2-letter Q words.

Examples:
  explore_lexicon.py NWL23.kwg --pattern 'A..'
  explore_lexicon.py NWL23_phony.kwg --pattern 'Q.' --pattern 'J.'
  explore_lexicon.py NWL23_phony.txt --length 2 --by-length
  explore_lexicon.py NWL23.kwg --pattern '*ING' --sample 20
"""

import argparse
import random
import re
from collections import Counter

from scribblez.lexical_tool.compiler import compile_kwg


def load_words(path: str) -> list[str]:
    if path.endswith(".kwg"):
        return compile_kwg(path).words()
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def pattern_to_regex(pat: str) -> re.Pattern:
    parts = ["^"]
    for ch in pat.upper():
        if ch == ".":
            parts.append("[A-Z]")
        elif ch == "*":
            parts.append("[A-Z]*")
        elif "A" <= ch <= "Z":
            parts.append(ch)
        else:
            raise ValueError(f"bad pattern character {ch!r} in {pat!r}")
    parts.append("$")
    return re.compile("".join(parts))


def select(words, args) -> list[str]:
    out = words
    if args.length is not None:
        out = [w for w in out if len(w) == args.length]
    if args.min_len is not None:
        out = [w for w in out if len(w) >= args.min_len]
    if args.max_len is not None:
        out = [w for w in out if len(w) <= args.max_len]
    if args.pattern:
        regexes = [pattern_to_regex(p) for p in args.pattern]
        out = [w for w in out if any(r.match(w) for r in regexes)]
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("lexicon", help="A .kwg or .txt lexicon.")
    p.add_argument("--pattern", action="append", default=[], help="Match pattern (repeatable; OR).")
    p.add_argument("--length", type=int, help="Keep only words of exactly this length.")
    p.add_argument("--min-len", type=int)
    p.add_argument("--max-len", type=int)
    p.add_argument("--by-length", action="store_true", help="Print a length histogram of matches.")
    p.add_argument(
        "--sample", type=int, help="Print this many random matches instead of the first."
    )
    p.add_argument("--limit", type=int, default=50, help="Max words to print (0 = all).")
    p.add_argument("--count", action="store_true", help="Print only the match count.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    matches = select(load_words(args.lexicon), args)
    print(f"matches: {len(matches):,}")
    if args.count:
        return 0

    if args.by_length:
        for length, n in sorted(Counter(len(w) for w in matches).items()):
            print(f"  len {length:>2}: {n:,}")
        return 0

    if args.sample is not None:
        shown = random.Random(args.seed).sample(matches, min(args.sample, len(matches)))
    else:
        shown = matches if args.limit == 0 else matches[: args.limit]
    for w in sorted(shown):
        print(w)
    if args.limit and not args.sample and len(matches) > args.limit:
        print(f"... ({len(matches) - args.limit:,} more; --limit 0 for all)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

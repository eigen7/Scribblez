#!/usr/bin/env python3
"""The noise of a sim's map residuals against the rollout count (DeepBot build
step 8, docs/plans/deep_bot.md): reads DeepBot --subtree-dump lines and, for
each (candidate, rack, contender) start, compares the squared residual of
Alice's sims after the contender against the leaf network's prior there with
the sampling noise those sims carry (sim/map_residuals.h).

For every start the dump records, per map, the squared residual summed over
the start's empty squares (r2) and the noise of the same sum (noise: the sims'
variance over their count). r2 - noise estimates the squared signal: what a
residual says that averaging more rollouts would not wash out. Tabulated by
rollout count and by bag size, with the sims' per-rollout variance against
the prior's own variance (its calibration).

    python3 -m scripts.deep_bot.residual_noise dump.jsonl [--by bag|sims]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

MAPS = ("points", "ownership", "end", "going_out")


def _bag_bucket(bag: int) -> str:
    if bag >= 60:
        return "bag 60+"
    if bag >= 30:
        return "bag 30-59"
    if bag >= 8:
        return "bag 8-29"
    return "bag 1-7"


def _rows(path: Path):
    """Yields (bag, dict of field -> value) per start of every turn in the dump."""
    with open(path) as f:
        for line in f:
            turn = json.loads(line)
            fields = turn.get("residual_fields")
            if not fields:
                continue
            for cand in turn["candidates"]:
                for rack in cand.get("residuals", []):
                    for row in rack:
                        yield turn["bag"], dict(zip(fields, row, strict=True))


def _accumulate(rows, key):
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for bag, r in rows:
        k = key(bag, r)
        counts[k] += 1
        for name, value in r.items():
            sums[k][name] += value
    return sums, counts


def _report(sums, counts, out=sys.stdout):
    header = f"{'group':>12} {'starts':>7} {'squares':>8}"
    for m in MAPS:
        header += f" | {m + ' r2':>12} {'noise':>10} {'signal2':>10}"
    header += f" | {'pts spread':>10} {'leaf var':>9} {'prior var':>9} | {'value r':>8}"
    header += " | ownership shift by class (0-4) | going-out shift (0-2)"
    print(header, file=out)
    for k in sorted(counts, key=_sort_key):
        n = counts[k]
        s = sums[k]
        line = f"{k:>12} {n:>7} {s['squares'] / n:>8.1f}"
        for m in MAPS:
            r2 = s[f"{m}_r2"] / n
            noise = s[f"{m}_noise"] / n
            line += f" | {r2:>12.4g} {noise:>10.4g} {r2 - noise:>10.4g}"
        line += (
            f" | {s['points_var'] / n:>10.4g} {s['points_leaf_var'] / n:>9.4g}"
            f" {s['points_prior_var'] / n:>9.4g}"
        )
        line += f" | {s['value_r'] / n:>8.4f}"
        line += " | " + " ".join(f"{s.get(f'ownership_shift_{c}', 0) / n:>6.2f}" for c in range(5))
        line += " | " + " ".join(f"{s.get(f'going_out_shift_{c}', 0) / n:>6.3f}" for c in range(3))
        if k == "sims 1":
            line += "   (one rollout: no noise estimate)"
        print(line, file=out)


def _sort_key(k):
    if isinstance(k, str) and k.startswith("sims "):
        return (0, int(k[5:]))
    return (1, str(k))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dump", type=Path, help="a DeepBot --subtree-dump file")
    ap.add_argument("--by", choices=["sims", "bag"], default="sims", help="how to group the starts")
    args = ap.parse_args(argv)
    if args.by == "sims":
        key = lambda bag, r: f"sims {int(r['sims'])}"  # noqa: E731
    else:
        key = lambda bag, r: _bag_bucket(bag)  # noqa: E731
    sums, counts = _accumulate(_rows(args.dump), key)
    if not counts:
        print("no residuals in the dump (a line needs residual_fields)", file=sys.stderr)
        return 1
    _report(sums, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""How often the policy head's top move at Bob's position is the reply his
subtree chose (DeepBot build step 10, docs/plans/deep_bot.md): reads DeepBot
--subtree-dump lines written with --policy-model and, over the searched
candidates, averages per rack the agreement with β of the policy's top move
over Bob's whole pool and over the candidates his subtree simmed, beside
HastyBot's (the equity-best move's), the baseline the policy must beat.
Tabulated by bag size.

    python3 -m scripts.deep_bot.policy_agreement dump.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SHARES = ("hasty", "policy_pool", "policy_simmed")


def bag_bucket(bag: int) -> str:
    if bag >= 60:
        return "bag 60+"
    if bag >= 30:
        return "bag 30-59"
    if bag >= 8:
        return "bag 8-29"
    return "bag 1-7"


def rows(path: Path):
    """Yields (bag, racks, shares by SHARES name) per searched candidate of
    every turn whose dump measured a policy."""
    with open(path) as f:
        for line in f:
            turn = json.loads(line)
            for cand in turn["candidates"]:
                if not cand["searched"] or "policy" not in cand:
                    continue
                yield (
                    turn["bag"],
                    turn["bob_racks"],
                    {
                        "hasty": cand["replies"]["hasty_share"],
                        "policy_pool": cand["policy"]["pool_share"],
                        "policy_simmed": cand["policy"]["simmed_share"],
                    },
                )


def tabulate(rows) -> dict[str, dict[str, float]]:
    """Per bag bucket (and "all"), the rack-weighted mean of each share and
    the racks counted."""
    sums = defaultdict(lambda: defaultdict(float))
    for bag, racks, shares in rows:
        for bucket in (bag_bucket(bag), "all"):
            sums[bucket]["racks"] += racks
            for name in SHARES:
                sums[bucket][name] += racks * shares[name]
    return {
        bucket: {"racks": s["racks"], **{name: s[name] / s["racks"] for name in SHARES}}
        for bucket, s in sums.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()
    table = tabulate(rows(args.dump))
    if not table:
        raise SystemExit(f"{args.dump}: no searched candidate carries a policy measurement")
    print(f"{'':12} {'racks':>8} {'hasty':>8} {'pool':>8} {'simmed':>8}")
    for bucket in sorted(table, key=lambda b: (b == "all", b)):
        t = table[bucket]
        print(
            f"{bucket:12} {int(t['racks']):8d} {t['hasty']:8.3f} {t['policy_pool']:8.3f} "
            f"{t['policy_simmed']:8.3f}"
        )


if __name__ == "__main__":
    main()

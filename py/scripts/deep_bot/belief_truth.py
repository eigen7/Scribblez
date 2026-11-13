#!/usr/bin/env python3
"""What Bob's belief makes of the leave Alice in fact keeps (engine
agent/belief_truth.h): reads DeepBot --subtree-dump lines written with a
leave inference model and, over the searched candidates, averages per rack
the log probability his belief gives her kept leave over the log probability
no information gives it, in nats. Above zero, the model knows something about
her leave; below, it is worse than knowing nothing.

Tabulated by the candidate's static-equity rank. Rank 0 is HastyBot's move,
the kind of move the model's training games hold; a candidate further down is
one that policy would not have played.

    python3 -m scripts.deep_bot.belief_truth dump.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def rank_bucket(rank: int) -> str:
    if rank == 0:
        return "rank 0"
    return "rank 1-9" if rank < 10 else "rank 10+"


def rows(path: Path):
    """Yields (equity rank, racks, log_prob - prior_log_prob) per searched
    candidate whose dump read a belief."""
    with open(path) as f:
        for line in f:
            for cand in json.loads(line)["candidates"]:
                if "belief" not in cand:
                    continue
                b = cand["belief"]
                yield cand["equity_rank"], b["racks"], b["log_prob"] - b["prior_log_prob"]


def tabulate(rows) -> dict[str, dict[str, float]]:
    """Per rank bucket (and "all"): the candidates and racks counted, the
    rack-weighted mean gain over the prior, and the share of candidates whose
    belief is worse than the prior."""
    sums = defaultdict(lambda: defaultdict(float))
    for rank, racks, gain in rows:
        for bucket in (rank_bucket(rank), "all"):
            s = sums[bucket]
            s["candidates"] += 1
            s["racks"] += racks
            s["gain"] += racks * gain
            s["worse"] += gain < 0
    return {
        bucket: {
            "candidates": s["candidates"],
            "racks": s["racks"],
            "gain": s["gain"] / s["racks"],
            "worse_share": s["worse"] / s["candidates"],
        }
        for bucket, s in sums.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()
    table = tabulate(rows(args.dump))
    if not table:
        raise SystemExit(f"{args.dump}: no searched candidate carries a belief")
    print(
        f"{'':10} {'candidates':>10} {'racks':>8} {'nats over prior':>16} {'worse than prior':>17}"
    )
    for bucket in sorted(table, key=lambda b: (b == "all", len(b), b)):
        t = table[bucket]
        print(
            f"{bucket:10} {int(t['candidates']):10d} {int(t['racks']):8d} {t['gain']:+16.3f} "
            f"{t['worse_share']:17.3f}"
        )


if __name__ == "__main__":
    main()

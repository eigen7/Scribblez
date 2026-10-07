#!/usr/bin/env python3
"""M1a's exhibits (docs/plans/supreme_bot_m1a.md, PR 5): hand-built positions
where the right move blocks a threat that only other moves' rollouts reveal.
See scribblez/transfer_test/exhibits.py.

    transfer_test_exhibits.py build [--out DIR]
        the exhibits' probes, labels and prior cache
    transfer_test_exhibits.py score --reader-tag TAG [--out DIR] [--checkpoint best|last]
        each exhibit's blocker-minus-open gap: labels, prior and reader
"""

import argparse
import json
from pathlib import Path

import torch
from scribblez.paths import TagPaths, add_mount_root_argument
from scribblez.transfer_test import exhibits
from scribblez.transfer_test.evaluate import load_reader
from scribblez.transfer_test.trainer import best_checkpoint_path
from scribblez.workloads.transfer_reader import SPEC as READER_SPEC

DEFAULT_OUT = Path("/workspace/mount/m1a-exhibits")


def print_scores(scores: dict):
    print(f"{'exhibit':<14}{'labels':>9}{'prior':>9}{'reader':>9}{'shuffled':>10}{'probed':>9}")
    for name, s in scores.items():
        print(
            f"{name:<14}{s['label_gap']:9.3f}{s['prior_gap']:9.3f}{s['reader_gap_held_out']:9.3f}"
            f"{s['reader_gap_shuffled']:10.3f}{s['reader_gap_probed']:9.3f}"
        )
    print(
        "\nThe blocker-minus-open gap in expected score: blockers held out (reader), their "
        "outcomes\nshuffled among the probes (shuffled), or their own probes kept (probed)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    s = sub.add_parser("score")
    s.add_argument("--reader-tag", required=True)
    s.add_argument("--checkpoint", choices=("best", "last"), default="best")
    s.add_argument("--device", default="cuda")
    for p in (b, s):
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        add_mount_root_argument(p)
    args = parser.parse_args()

    if args.command == "build":
        exhibits.build(args.out, args.mount_root)
        print(f"exhibits written to {args.out}")
        return
    paths = TagPaths(args.reader_tag, READER_SPEC.name, args.mount_root)
    path = best_checkpoint_path(paths) if args.checkpoint == "best" else paths.rolling_checkpoint
    device = torch.device(args.device)
    reader, config = load_reader(path, device)
    f, blockers = exhibits.load(args.out)
    scores = exhibits.score(reader, config, f, blockers, device)
    print_scores(scores)
    out = paths.root / "evaluations" / f"exhibits.{args.checkpoint}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=1))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

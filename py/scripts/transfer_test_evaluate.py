#!/usr/bin/env python3
"""Score a transfer_reader tag's reader and the baseline arms on a
transfer_test corpus tag (docs/plans/supreme_bot_m1a.md, PR 5): held-out and
probed within-row error and held-out pair accuracy, overall and by tiles in
the bag, with bootstrap intervals against the prior. See
scribblez/transfer_test/evaluate.py. The report is printed and written as JSON
under the reader tag's evaluations/.

Usage: transfer_test_evaluate.py --reader-tag TAG --test-tag TAG
           [--checkpoint best|last] [--replicates 4] [--device cuda]
"""

import argparse
import json

import torch
from scribblez.paths import TagPaths, add_mount_root_argument
from scribblez.transfer_test.corpus import load_corpus
from scribblez.transfer_test.evaluate import ARMS, evaluate, load_reader
from scribblez.transfer_test.trainer import best_checkpoint_path
from scribblez.workloads.transfer_reader import SPEC as READER_SPEC
from scribblez.workloads.transfer_test import CORPUS_DIR
from scribblez.workloads.transfer_test import SPEC as CORPUS_SPEC

LABELS = {
    "prior": "teacher prior",
    "shrinkage": "shrinkage",
    "common_shift": "common shift",
    "reader": "reader",
    "shuffled": "reader, shuffled outcomes",
}


def print_report(r: dict):
    print(
        f"{r['positions']} scored positions ({r['decided_positions']} decided, not scored), "
        f"{r['heldout_candidates']} held-out candidates, {r['pairs']} resolved held-out pairs; "
        f"shrinkage weight {r['shrinkage_probes']} probes"
    )
    header = f"{'arm':<28}{'held-out err':>13}{'probed err':>12}{'pair acc':>10}"
    print(f"\n{header}   held-out err vs prior")
    for arm in ARMS:
        m, ci = r["overall"][arm], r["intervals_vs_prior"][arm]["heldout_rmse"]
        print(
            f"{LABELS[arm]:<28}{m['heldout_rmse']:13.4f}{m['probed_rmse']:12.4f}"
            f"{m['pair_accuracy']:10.3f}   [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        )
    print("\nby tiles in the bag (held-out error / pair accuracy):")
    print(f"{'bag':<8}{'positions':>10}" + "".join(f"{a:>20}" for a in ARMS))
    for band, m in r["by_bag"].items():
        cells = "".join(
            f"{m[a]['heldout_rmse']:>12.4f} / {m[a]['pair_accuracy']:.2f}" for a in ARMS
        )
        print(f"{band:<8}{m['positions']:>10}{cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reader-tag", required=True)
    parser.add_argument("--test-tag", required=True)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--replicates", type=int, default=4, help="rows per test position")
    parser.add_argument("--device", default="cuda")
    add_mount_root_argument(parser)
    args = parser.parse_args()

    reader_paths = TagPaths(args.reader_tag, READER_SPEC.name, args.mount_root)
    path = (
        best_checkpoint_path(reader_paths)
        if args.checkpoint == "best"
        else reader_paths.rolling_checkpoint
    )
    device = torch.device(args.device)
    reader, config = load_reader(path, device)
    files = load_corpus(
        TagPaths(args.test_tag, CORPUS_SPEC.name, args.mount_root).data_dir / CORPUS_DIR
    )
    result = {
        "reader_tag": args.reader_tag,
        "checkpoint": str(path),
        "test_tag": args.test_tag,
        "replicates": args.replicates,
        **evaluate(reader, config, files, device, args.replicates),
    }
    print_report(result)
    out = reader_paths.root / "evaluations" / f"{args.test_tag}.{args.checkpoint}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

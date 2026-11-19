#!/usr/bin/env python3
"""Copy a finished leaf_eval_truth tag's truth into the repo, beside the known
positions (deep_bot_leaf_eval/truth.py's TRUTH_FILE), where the
deep_bot_leaf_eval Positions tab reads it. A position the tag has no truth
for, or whose GCG has changed since, is left out and named.

    py/scripts/deep_bot_leaf_eval/copy_truth.py --tag v1
"""

import argparse
import json
import sys

from scribblez.deep_bot import known_positions
from scribblez.deep_bot_leaf_eval import truth
from scribblez.paths import add_mount_root_argument
from scribblez.workloads import leaf_eval_truth
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", required=True, help="the leaf_eval_truth tag to copy")
    add_mount_root_argument(p)
    args = p.parse_args()
    data_dir = leaf_eval_truth.SPEC.paths(args.tag, args.mount_root).data_dir
    doc, skipped = truth.collect(data_dir, known_positions.load())
    if not doc["positions"]:
        print(f"tag '{args.tag}' holds no truth for the current positions", file=sys.stderr)
        return 1
    truth.TRUTH_FILE.write_text(json.dumps({"tag": args.tag, **doc}, indent=1) + "\n")
    print(f"wrote {len(doc['positions'])} positions to {truth.TRUTH_FILE}")
    if skipped:
        print(f"left out (no truth for the current GCG): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

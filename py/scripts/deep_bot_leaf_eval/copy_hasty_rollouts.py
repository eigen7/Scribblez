#!/usr/bin/env python3
"""Copy a finished leaf_eval_hasty_rollouts tag's rollouts into the repo,
beside the known positions (deep_bot_leaf_eval/hasty_rollouts.py's
ROLLOUTS_FILE), where the Positions tab reads them. A position the tag has
not played out, or whose GCG has changed since, is left out and named.

    py/scripts/deep_bot_leaf_eval/copy_hasty_rollouts.py --tag v2
"""

import argparse
import json
import sys

from scribblez.deep_bot import known_positions
from scribblez.deep_bot_leaf_eval import hasty_rollouts
from scribblez.paths import add_mount_root_argument
from scribblez.workloads import leaf_eval_hasty_rollouts
from util.argparse_ext import ArgumentDefaultsHelpFormatter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", required=True, help="the leaf_eval_hasty_rollouts tag to copy")
    add_mount_root_argument(p)
    args = p.parse_args()
    data_dir = leaf_eval_hasty_rollouts.SPEC.paths(args.tag, args.mount_root).data_dir
    doc, skipped = hasty_rollouts.collect(data_dir, known_positions.load())
    if not doc["positions"]:
        print(f"tag '{args.tag}' holds no rollouts of the current positions", file=sys.stderr)
        return 1
    hasty_rollouts.ROLLOUTS_FILE.write_text(json.dumps({"tag": args.tag, **doc}, indent=1) + "\n")
    print(f"wrote {len(doc['positions'])} positions to {hasty_rollouts.ROLLOUTS_FILE}")
    if skipped:
        print(f"left out (no rollouts of the current GCG): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

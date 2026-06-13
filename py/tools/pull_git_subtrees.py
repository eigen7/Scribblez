#!/usr/bin/env python3
"""Pull updates for each git subtree listed in subtrees/subtrees.json.

Usage:
    py/tools/pull_git_subtrees.py [-y]

By default, prompts for confirmation before pulling each subtree.
Pass -y / --yes to skip prompts and pull all automatically.

Refuses to run if the working tree has uncommitted changes (staged or
unstaged), since ``git subtree pull`` requires a clean tree.
"""
import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBTREES_JSON = os.path.join(REPO_ROOT, "subtrees", "subtrees.json")


def repo_is_clean() -> bool:
    """Return True if the working tree has no staged or unstaged changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.returncode == 0 and result.stdout.strip() == ""


def pull_subtree(prefix: str, url: str, branch: str) -> bool:
    """Run ``git subtree pull`` for one entry. Return True on success."""
    cmd = [
        "git", "subtree", "pull",
        f"--prefix={prefix}",
        url,
        branch,
        "--squash",
    ]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="skip confirmation prompts and pull all subtrees",
    )
    args = parser.parse_args()

    if not os.path.isfile(SUBTREES_JSON):
        print(f"ERROR: subtree manifest not found at {SUBTREES_JSON}",
              file=sys.stderr)
        sys.exit(1)

    with open(SUBTREES_JSON) as f:
        subtrees = json.load(f)

    if not subtrees:
        print("No subtrees defined in subtrees.json.")
        return

    if not repo_is_clean():
        print("ERROR: working tree has uncommitted changes. "
              "Commit or stash them before pulling subtrees.",
              file=sys.stderr)
        sys.exit(1)

    results = []
    for entry in subtrees:
        prefix = entry["prefix"]
        url = entry["url"]
        branch = entry["branch"]

        if not args.yes:
            answer = input(f"Pull {prefix} from {url} ({branch})? [Y/n] ").strip().lower()
            if answer not in ("", "y", "yes"):
                print(f"  Skipping {prefix}.")
                results.append((prefix, "skipped"))
                continue

        ok = pull_subtree(prefix, url, branch)
        results.append((prefix, "ok" if ok else "FAILED"))

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    for prefix, status in results:
        print(f"  {prefix}: {status}")

    if any(s == "FAILED" for _, s in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

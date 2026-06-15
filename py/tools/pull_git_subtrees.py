#!/usr/bin/env python3
"""Pull each git subtree under subtrees/ to its upstream tip.

Usage:
    py/tools/pull_git_subtrees.py [-y]

By default, prompts for confirmation before pulling each subtree.
Pass -y / --yes to skip prompts and pull all automatically.

Refuses to run if the working tree has uncommitted changes (staged or
unstaged), since ``git subtree pull`` requires a clean tree.
"""
import argparse

from setup_check import import_setup_common


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

    dev_tool = import_setup_common().dev_tool()
    dev_tool.pull_git_subtrees("subtrees", assume_yes=args.yes)


if __name__ == "__main__":
    main()

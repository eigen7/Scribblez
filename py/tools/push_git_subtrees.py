#!/usr/bin/env python3
"""Push each git subtree under subtrees/ to its upstream branch.

Usage:
    py/tools/push_git_subtrees.py [-y]

By default, prompts for confirmation before pushing each subtree.
Pass -y / --yes to skip prompts and push all automatically.

Unlike pulling, this does not require a clean working tree.
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
        help="skip confirmation prompts and push all subtrees",
    )
    args = parser.parse_args()

    dev_tool = import_setup_common().dev_tool()
    dev_tool.push_git_subtrees("subtrees", assume_yes=args.yes)


if __name__ == "__main__":
    main()

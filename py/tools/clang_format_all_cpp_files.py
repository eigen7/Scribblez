#!/usr/bin/env python3
"""Run clang-format over the engine's C++ sources.

Usage:
    py/tools/clang_format_all_cpp_files.py [--check]

With --check, reports files that would change and exits non-zero if any
exist (useful in CI). Without it, reformats files in place.
"""
import argparse

from setup_check import import_setup_common


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check formatting without modifying files (exit 1 if any differ)",
    )
    args = parser.parse_args()

    dev_tool = import_setup_common().dev_tool()
    dev_tool.clang_format_all_cpp_files(cpp_dirs=["engine"], check=args.check)


if __name__ == "__main__":
    main()

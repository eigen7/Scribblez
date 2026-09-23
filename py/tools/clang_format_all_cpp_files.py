#!/usr/bin/env python3
"""Run clang-format over every C++ source under engine/.

py/tools/clang_format_all_cpp_files.py            # reformat in place
py/tools/clang_format_all_cpp_files.py --check    # report only; exit 1 if any differ
"""

import sys
from pathlib import Path

# Put this checkout's py/ first on sys.path. Otherwise `setup_check` resolves
# through the container's .pth entry, which points at the main checkout (and
# does not exist on the host). Run from a git worktree, the script would then
# silently format the main checkout instead of the worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from setup_check import import_setup_common

if __name__ == "__main__":
    import_setup_common().dev_tool().clang_format_cli(["engine"])

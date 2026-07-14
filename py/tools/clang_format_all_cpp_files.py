#!/usr/bin/env python3
"""Run clang-format over all C++ sources."""

import sys
from pathlib import Path

# Put this checkout's py/ first on sys.path: `setup_check` otherwise resolves
# only through the container's main-checkout .pth entry (and not at all on the
# host), so this script run from a git worktree would silently import -- and
# operate on -- the main checkout instead of its own.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from setup_check import import_setup_common

if __name__ == "__main__":
    import_setup_common().dev_tool().clang_format_cli(["engine"])

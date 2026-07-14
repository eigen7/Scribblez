#!/usr/bin/env python3
"""Scribblez entry point for the local Gitea stack used for PR review.

The implementation is shared across consumer projects; see
submodules/devenv_utils/gitea_serve.py.
"""

import sys
from pathlib import Path

# Put this checkout's py/ first on sys.path: `setup_check` otherwise resolves
# only through the container's main-checkout .pth entry (and not at all on the
# host), so this script run from a git worktree would silently import -- and
# operate on -- the main checkout instead of its own.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from setup_check import import_setup_common  # noqa: E402

setup_common = import_setup_common()

from submodules.devenv_utils import gitea_serve  # noqa: E402

if __name__ == "__main__":
    gitea_serve.main(setup_common.make_config())

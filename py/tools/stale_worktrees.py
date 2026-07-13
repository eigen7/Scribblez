#!/usr/bin/env python3
"""Scribblez entry point for the abandoned-worktree report.

The implementation is shared across consumer projects; see
submodules/devenv_utils/stale_worktrees.py.
"""

from setup_check import import_setup_common

setup_common = import_setup_common()

from submodules.devenv_utils import stale_worktrees  # noqa: E402

if __name__ == "__main__":
    stale_worktrees.main(setup_common.make_config())

#!/usr/bin/env python3
"""Scribblez entry point for the worktree-and-PR workflow tool.

The implementation is shared across consumer projects; see
submodules/devenv_utils/pr_flow.py for the subcommands (worktree / create /
merge) and CLAUDE.md for the workflow they drive.
"""

from setup_check import import_setup_common

setup_common = import_setup_common()

from submodules.devenv_utils import pr_flow  # noqa: E402

if __name__ == "__main__":
    pr_flow.main(setup_common.make_config())

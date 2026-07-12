This directory contains git submodules: full checkouts of repos we control,
each pinned to a commit. The workflow -- changing a submodule, pointer-bump
rules, first-clone initialization, worktree interactions -- is documented in
[devenv_utils/SUBMODULES.md](devenv_utils/SUBMODULES.md). Read it before
touching anything under this directory.

A plain `git clone` leaves the submodules empty; the first run of any
host-side script populates them (see the stanza atop setup_common.py).

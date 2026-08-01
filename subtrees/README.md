This directory contains vendored git subtrees: read-only copies of repos we
control, updated by `git subtree pull` and never edited in place (a
pre-commit hook enforces this). The rules -- updating a copy, changing the
source repo, coordinated changes -- are documented in
[devenv_utils/SUBTREES.md](devenv_utils/SUBTREES.md). Read it before touching
anything under this directory.

This directory contains git subtrees.

Each subtree's remote URL and tracked branch are declared in the `SUBTREES`
list in the repo-root `setup_common.py` (git stores neither anywhere
committed). The pull/push tools read that declaration and operate on every
subtree directory found here.

To pull all subtrees to latest:

```
./py/tools/pull_git_subtrees.py
```

To push local changes back upstream:

```
./py/tools/push_git_subtrees.py
```

## `devenv_utils`

Added via:

```
git subtree add --prefix=subtrees/devenv_utils https://github.com/eigen7/devenv_utils.git main --squash
```

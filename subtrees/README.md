This directory contains git subtrees.

Each subtree's remote URL and tracked branch are declared in the `SUBTREES`
list in the repo-root `setup_common.py` (git records neither anywhere
committed). The pull tool reads that declaration.

## Read-only vendored mirrors

Each `subtrees/<dir>/` is a **read-only** vendored copy of an upstream repo. The
files are committed in-tree, so a plain `git clone` gets everything — no
submodule init step. You do not edit them here, and you do not push from this
checkout.

* **Update** a subtree to its upstream tip:

  ```
  ./py/tools/pull_git_subtrees.py
  ```

  This is a `git subtree pull` (a merge), and it's the only thing that changes a
  subtree. Because the prefix is never edited locally and never pushed, the pull
  is always a clean fast-forward — no conflicts.

* **Change** a subtree's contents: edit its own upstream repo, push there, then
  run the pull tool here. There is deliberately no push from this checkout.

## Enforcement

Direct edits to a subtree are blocked at two layers, as early as git allows:

* **Locally**, `subtrees/devenv_utils/hooks/pre-commit` rejects any commit that
  stages changes under `subtrees/<dir>/`. `setup_wizard.py` activates it via
  `DevTool.ensure_git_hooks()` (which sets `core.hooksPath`); because `.git` is
  bind-mounted into the dev container, that one setting covers git on both the
  host and inside the container.

* **Server-side**, the `subtree-readonly` GitHub Actions workflow runs
  `subtrees/devenv_utils/subtree_guard.py` over each push/PR. This is the
  unbypassable backstop (a local hook can be skipped with `--no-verify`).

A `git subtree pull` is exempt from both: it records a merge commit, and git
does not run `pre-commit` on merges; the CI check walks first-parent history,
which skips the merge and its squash commit.

## `devenv_utils`

Added via:

```
git subtree add --prefix=subtrees/devenv_utils https://github.com/eigen7/devenv_utils.git main --squash
```

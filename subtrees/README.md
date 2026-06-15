This directory contains git subtrees.

Each subtree's remote URL and tracked branch are declared in the `SUBTREES`
list in the repo-root `setup_common.py` (git records neither anywhere
committed). The pull/push tools read that declaration.

## Workflow: one unit per commit

A `subtrees/<dir>/` is a real git subtree, and you *may* edit it. The one rule
is that **a single commit must touch exactly one "unit"** — either a single
subtree (`subtrees/<dir>/...`) or the parent repo (everything else). It may not
mix the parent with a subtree, nor two different subtrees. That keeps each
commit pushable to a single destination:

* Changes under `subtrees/<dir>/` are pushed to that subtree's own upstream:

  ```
  ./py/tools/push_git_subtrees.py
  ```

* Changes elsewhere are pushed to this repo with a normal `git push`.

* To sync the local snapshot up to the subtree's latest upstream:

  ```
  ./py/tools/pull_git_subtrees.py
  ```

## Enforcement

Two layers share one implementation, `subtrees/devenv_utils/commit_purity.py`
(vendored, so any repo using devenv_utils gets the same rule):

* **Locally**, the hook `subtrees/devenv_utils/hooks/pre-commit` calls it with
  `--staged`. `setup_wizard.py` activates it by setting `core.hooksPath` to that
  hooks dir — and because `.git` is bind-mounted into the dev container, that one
  setting covers git run both on the host and inside the container. It's a
  guardrail: `git commit --no-verify` skips it, and it only applies once
  `setup_wizard.py` has run.

* **Server-side**, the `commit-purity` GitHub Actions workflow
  (`.github/workflows/commit-purity.yml`) runs the same script over a commit
  range on every push and PR. This is the unbypassable backstop, active for
  every developer regardless of local config. (The workflow YAML must live at
  the repo root — GitHub won't run it from the subtree — but it's a thin stub
  around the shared script.)

A `git subtree pull` is exempt from both: it records a *merge* commit, and both
the hook (git skips `pre-commit` on merges) and the CI check (`--no-merges`)
ignore merges.

## `devenv_utils`

Added via:

```
git subtree add --prefix=subtrees/devenv_utils https://github.com/eigen7/devenv_utils.git main --squash
```

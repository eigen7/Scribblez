This directory contains git subtrees.

| dir   | url  | branch |
| ----- | ---- | ----------- |
| devenv_utils | https://github.com/eigen7/devenv_utils.git | main |

Added via:

```
git subtree add --prefix=subtrees/<dir> <url> <branch> --squash
```

To update:

```
git subtree pull --prefix=subtrees/<dir> <url> <branch> --squash
```

To push a change:

```
git subtree push --prefix=subtrees/<dir> <url> <branch>
```

## Importing

`__init__.py` in this directory makes subtree packages importable as
`subtrees.<dir>` from the repo root (e.g. `from subtrees.devenv_utils import
SetupWizardTool`). It is project-owned and not part of any subtree.

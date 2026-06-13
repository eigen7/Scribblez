This directory contains git subtrees.

To pull all subtrees to latest:

```
./py/tools/pull_git_subtrees.py
```

## `devenv_utils`

Added via:

```
git subtree add --prefix=subtrees/devenv_utils https://github.com/eigen7/devenv_utils.git main --squash
```

To update:

```
git subtree pull --prefix=subtrees/devenv_utils https://github.com/eigen7/devenv_utils.git main --squash
```

To push a change:

```
git subtree push --prefix=subtrees/devenv_utils https://github.com/eigen7/devenv_utils.git main
```

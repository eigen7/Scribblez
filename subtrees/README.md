This directory contains git submodules.

## `devenv_utils`

`subtrees/devenv_utils/` is a git submodule of
https://github.com/eigen7/devenv_utils.git (the URL is recorded in the
repo-root `.gitmodules`). The directory is a full checkout of that repo, so it
cannot silently diverge from upstream: any edit made here is, by construction,
a change to the devenv_utils repo itself.

* **Change** the code: edit in place, commit inside the submodule, and push to
  the devenv_utils repo. Then commit the updated submodule pointer here
  (`git add subtrees/devenv_utils`).

* **Update** to the upstream tip without local changes:

  ```
  git -C subtrees/devenv_utils pull origin main
  git add subtrees/devenv_utils
  ```

## Cloning and initialization

A plain `git clone` leaves the submodule directory empty. Every host-side
entry point (`setup_wizard.py`, `run_docker.py`, ...) imports `setup_common`,
which populates it automatically (`git submodule update --init`) before
importing from it — so no manual step is required. `git clone
--recurse-submodules` also works.

Two git config settings, applied by `setup_wizard.py`, keep the submodule in
sync day to day:

* `submodule.recurse=true` — `git pull` / `git checkout` update the submodule
  working tree to match the commit the superproject records.
* `push.recurseSubmodules=check` — git refuses to push a commit whose
  submodule pointer references a commit absent from the submodule's remote
  (which would break every other clone).

## Worktrees

`git worktree add` does not populate submodules: run `git submodule update
--init` inside a new worktree before using it. Conversely, `git worktree
remove` refuses to remove a worktree containing a populated submodule; pass
`--force`.

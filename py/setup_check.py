"""Setup-version gate for the in-container entrypoints under py/.

The version number and the actual check live in the repo-root `setup_common`
module. The scripts here run with `py/` (not the repo root) on `sys.path`, so
they cannot import `setup_common` directly. This thin wrapper bridges that gap
without a module-level path hack: the one line that puts the repo root on
`sys.path` lives inside the function, scoped to the moment it is needed.
"""


def check_setup_version() -> None:
    """Abort unless .env.json records a current setup (see setup_common)."""
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parent.parent)
    added = repo_root not in sys.path
    if added:
        sys.path.insert(0, repo_root)
    try:
        from setup_common import check_setup_version as _check_setup_version
    finally:
        # Leave sys.path as we found it; the import is cached in sys.modules,
        # so the repo root no longer needs to be on the path after this.
        if added:
            sys.path.remove(repo_root)

    _check_setup_version()

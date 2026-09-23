"""Lets py/ entrypoints import the repo-root `setup_common` module.

Scripts under py/ run with py/, not the repo root, on sys.path, so a plain
`import setup_common` fails. Use `setup_common = import_setup_common()`
instead.
"""

import sys
from pathlib import Path


def import_setup_common():
    """Put the repo root on sys.path and return the `setup_common` module."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import setup_common

    return setup_common

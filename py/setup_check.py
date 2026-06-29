"""Importer for the repo-root `setup_common` module from the py/ entrypoints.

Inside the dev container, the scripts under py/ run with py/ -- not the repo
root -- on PYTHONPATH, so a plain `import setup_common` does not resolve. Use
`import_setup_common()` instead: it puts the repo root on `sys.path` and returns
the imported module. It also works when run from outside the container.
"""

import sys
from pathlib import Path


def import_setup_common():
    """Import and return the repo-root `setup_common` module.

    When running inside the dev container, use this rather than
    `import setup_common`: the py/ entrypoints run with py/ -- not the repo
    root -- on PYTHONPATH, so a direct import would not resolve. For
    convenience it works from outside the container too. Typical use:

        setup_common = import_setup_common()
    """
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import setup_common
    return setup_common

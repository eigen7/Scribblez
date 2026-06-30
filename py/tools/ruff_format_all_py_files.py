#!/usr/bin/env python3
"""Format and lint all of the project's Python sources with Ruff.

The Python counterpart of py/tools/clang_format_all_cpp_files.py. Default mode
reformats in place and applies Ruff's safe lint fixes; ``--check`` reports what
would change and exits non-zero if anything does, without modifying files.

Ruff reads its configuration from the repo-root pyproject.toml ([tool.ruff]).
The target paths below are the project's first-party Python; the read-only
subtrees and the build directory are excluded by that config.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# First-party Python: the package/scripts/tests tree, the host-side setup
# scripts at the repo root, and the position-metadata helper. Paths are
# relative to REPO_ROOT and passed straight to Ruff, which applies the
# pyproject excludes within them.
TARGETS = [
    "py",
    "positions",
    "build_docker_image.py",
    "run_docker.py",
    "setup_common.py",
    "setup_wizard.py",
]


def _abort(message: str):
    """Print an error to stderr and exit non-zero."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _run_ruff(args: list[str]) -> int:
    """Run ``ruff`` with *args* from the repo root; return its exit code."""
    return subprocess.run(["ruff", *args, *TARGETS], cwd=REPO_ROOT).returncode


def main(check: bool):
    """Format and lint the targets, or check them when *check* is true."""
    if shutil.which("ruff") is None:
        _abort("ruff not found on PATH.")
    if check:
        # Linter first, then the formatter, so a single run surfaces both
        # classes of problem. Combine the exit codes so either failing fails
        # the whole check.
        rc = _run_ruff(["check"])
        rc |= _run_ruff(["format", "--check"])
        if rc:
            sys.exit(1)
        print("All Python files are correctly formatted and linted.")
    else:
        # Apply lint fixes before formatting so the formatter has the last word
        # on layout (Ruff's recommended order).
        if _run_ruff(["check", "--fix"]):
            _abort("ruff found lint issues it could not fix automatically.")
        _run_ruff(["format"])
        print("Done.")


def _parse_args() -> bool:
    """Return whether ``--check`` was requested."""
    parser = argparse.ArgumentParser(
        description="Run Ruff over the project's Python sources. With --check, "
        "report files that would change and exit non-zero if any "
        "do; otherwise reformat in place and apply lint fixes.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check formatting/lint without modifying files (exit 1 if any differ)",
    )
    return parser.parse_args().check


if __name__ == "__main__":
    main(_parse_args())

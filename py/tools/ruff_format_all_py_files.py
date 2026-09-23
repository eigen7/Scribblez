#!/usr/bin/env python3
"""Format and lint the project's first-party Python with Ruff.

The Python counterpart of clang_format_all_cpp_files.py:

    py/tools/ruff_format_all_py_files.py            # apply safe lint fixes, then reformat
    py/tools/ruff_format_all_py_files.py --check    # report only; exit 1 if anything differs

Configuration lives in the repo-root pyproject.toml ([tool.ruff]), which also
excludes subtrees/ and target/.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Relative to REPO_ROOT and passed straight to Ruff, which applies the
# pyproject excludes within them.
TARGETS = [
    "py",
    "docker-setup",
    "build_and_push_worker_image.py",
    "build_docker_image.py",
    "run_docker.py",
    "setup_common.py",
    "setup_wizard.py",
]


def _abort(message: str):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _run_ruff(args: list[str]) -> int:
    return subprocess.run(["ruff", *args, *TARGETS], cwd=REPO_ROOT).returncode


def main(check: bool):
    if shutil.which("ruff") is None:
        _abort("ruff not found on PATH.")
    if check:
        # Run both even if the first fails, so one run reports every problem.
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

#!/usr/bin/env python3
"""Run clang-format in-place on every C++ source file under engine/.

Usage:
    py/tools/clang_format_all_cpp_files.py [--check]

With --check, reports files that would change and exits non-zero if any
exist (useful in CI). Without it, reformats files in place.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_DIR = os.path.join(REPO_ROOT, "engine")
CPP_EXTENSIONS = {".cpp", ".h", ".inl", ".hpp", ".cc", ".cxx"}


def find_cpp_files(root: str) -> list[str]:
    """Return sorted list of C++ file paths under *root*."""
    result = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1] in CPP_EXTENSIONS:
                result.append(os.path.join(dirpath, name))
    result.sort()
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check formatting without modifying files (exit 1 if any differ)",
    )
    args = parser.parse_args()

    clang_format = "clang-format"
    if subprocess.run([clang_format, "--version"], capture_output=True).returncode != 0:
        print("ERROR: clang-format not found on PATH.", file=sys.stderr)
        sys.exit(1)

    files = find_cpp_files(ENGINE_DIR)
    if not files:
        print("No C++ files found under engine/.")
        return

    print(f"Found {len(files)} C++ file(s) under engine/.")

    if args.check:
        bad = []
        for path in files:
            result = subprocess.run(
                [clang_format, "--dry-run", "--Werror", path],
                capture_output=True,
            )
            if result.returncode != 0:
                bad.append(os.path.relpath(path, REPO_ROOT))
        if bad:
            print(f"{len(bad)} file(s) need formatting:")
            for f in bad:
                print(f"  {f}")
            sys.exit(1)
        else:
            print("All files are correctly formatted.")
    else:
        for path in files:
            relpath = os.path.relpath(path, REPO_ROOT)
            subprocess.run([clang_format, "-i", path], check=True)
            print(f"  formatted {relpath}")
        print("Done.")


if __name__ == "__main__":
    main()

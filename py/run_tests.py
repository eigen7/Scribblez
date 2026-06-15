#!/usr/bin/env python3
"""Run the full Scribblez test suite: C++ engine tests + Python tests + web UI tests.

Usage:
    py/run_tests.py [--cpp-only] [--python-only] [--web-only] [--verbose]

The C++ tests are run via CTest against the existing target/ directory (use
py/build.py first if you haven't built yet). The Python tests are run via
pytest against py/tests/. The web tests are run via `npm run test:run`
(vitest) in web/.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "target")
WEB_DIR = os.path.join(ROOT, "web")
PYTHON_DIR = os.path.join(ROOT, "py")


def run(cmd, cwd=None):
    """Run a command, streaming output. Return the exit code."""
    print(f"\n$ {cmd}  (cwd={cwd or ROOT})")
    return subprocess.run(cmd, shell=True, cwd=cwd or ROOT).returncode


def run_cpp_tests(verbose: bool) -> int:
    if not os.path.isdir(BUILD_DIR):
        print(f"ERROR: build directory not found at {BUILD_DIR}.\n"
              "Run py/build.py first.", file=sys.stderr)
        return 1
    if shutil.which("ctest") is None:
        print("ERROR: ctest not found on PATH.", file=sys.stderr)
        return 1
    flags = "--output-on-failure"
    if verbose:
        flags += " -V"
    return run(f"ctest {flags}", cwd=BUILD_DIR)


def run_web_tests(verbose: bool) -> int:
    if shutil.which("npm") is None:
        print("ERROR: npm not found on PATH. Install Node.js to run web tests.",
              file=sys.stderr)
        return 1
    if not os.path.isdir(os.path.join(WEB_DIR, "node_modules")):
        print(f"ERROR: web/node_modules not found.\n"
              "Run py/build.py (or `npm install` in web/) first.",
              file=sys.stderr)
        return 1
    cmd = "npm run test:run"
    if verbose:
        cmd += " -- --reporter=verbose"
    return run(cmd, cwd=WEB_DIR)


def run_python_tests(verbose: bool) -> int:
    if shutil.which("python3") is None:
        print("ERROR: python3 not found on PATH.", file=sys.stderr)
        return 1
    test_dir = os.path.join(PYTHON_DIR, "tests")
    if not os.path.isdir(test_dir):
        print(f"ERROR: python test directory not found at {test_dir}.",
              file=sys.stderr)
        return 1
    flags = "-v" if verbose else ""
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHON_DIR
    cmd = f"python3 -m pytest {test_dir} {flags}".strip()
    print(f"\n$ {cmd}  (cwd={ROOT})")
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=env).returncode


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpp-only", action="store_true",
                       help="run only the C++ engine tests")
    group.add_argument("--python-only", action="store_true",
                       help="run only the Python tests")
    group.add_argument("--web-only", action="store_true",
                       help="run only the web UI tests")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable verbose test output")
    args = parser.parse_args()

    run_cpp = not args.web_only and not args.python_only
    run_py = not args.cpp_only and not args.web_only
    run_web = not args.cpp_only and not args.python_only

    results = []
    if run_cpp:
        print("=" * 60)
        print("C++ engine tests")
        print("=" * 60)
        results.append(("C++ engine tests", run_cpp_tests(args.verbose)))
    if run_py:
        print("\n" + "=" * 60)
        print("Python tests")
        print("=" * 60)
        results.append(("Python tests", run_python_tests(args.verbose)))
    if run_web:
        print("\n" + "=" * 60)
        print("Web UI tests")
        print("=" * 60)
        results.append(("Web UI tests", run_web_tests(args.verbose)))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    failed = 0
    for name, rc in results:
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        print(f"  {name}: {status}")
        if rc != 0:
            failed += 1

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

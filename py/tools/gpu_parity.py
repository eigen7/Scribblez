#!/usr/bin/env python3
"""Check that the engine's TensorRT inference matches PyTorch, for every model
family: position evaluation, move set evaluation, move proposal, and leave
inference.

py/tools/gpu_parity.py

These checks run on the GPU, so they are not part of py/run_tests.py (see
CLAUDE.md for when to run them). Each family is one test binary under
target/engine, built by py/build.py, and one Python generator that exports a
tiny random-init model with its PyTorch reference outputs. This script
generates each fixture once into a scratch dir and hands it to the binary,
which builds TensorRT plans and compares; a binary run by hand with no
argument generates its own fixture for every case instead. The binaries run
one after another, so the check holds the GPU for one plan build at a time.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Put this checkout's py/ first on sys.path. Otherwise `scribblez` resolves
# through the container's .pth entry, which points at the main checkout: run
# from a git worktree, the script would test the main checkout's engine.
PY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PY_DIR))

from scribblez.paths import ENGINE_DIR  # noqa: E402

# (test binary, fixture generator module and its flags)
FAMILIES = [
    ("test_nn_inference_parity", ["scripts.position_eval.gen_parity_fixture", "--num-rows", "8"]),
    ("test_mset_inference_parity", ["scripts.move_set_eval.gen_mset_parity_fixture"]),
    ("test_proposal_inference_parity", ["scripts.move_set_eval.gen_proposal_parity_fixture"]),
    ("test_leave_inference_parity", ["scripts.leave_inference.gen_parity_fixture"]),
]


def run(cmd: list[str]) -> bool:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    env = os.environ | {"PYTHONPATH": str(PY_DIR)}
    return subprocess.run(cmd, cwd=PY_DIR, env=env).returncode == 0


def check_family(binary_name: str, generator: list[str], scratch: Path) -> bool:
    binary = ENGINE_DIR / binary_name
    if not binary.is_file():
        print(f"ERROR: {binary} not found. Run py/build.py first.", file=sys.stderr)
        return False
    fixture = scratch / binary_name
    if not run([sys.executable, "-m", *generator, "--out-dir", str(fixture)]):
        return False
    return run([str(binary), str(fixture)])


def main():
    with tempfile.TemporaryDirectory(prefix="scribblez_gpu_parity_") as scratch:
        results = [(name, check_family(name, gen, Path(scratch))) for name, gen in FAMILIES]
    print("\nSummary")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()

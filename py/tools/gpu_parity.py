#!/usr/bin/env python3
"""Check that the engine's TensorRT inference matches PyTorch, for every model
family: position evaluation, move set evaluation, move proposal, leave
inference, and the DeepBot leaf eval; and that the GPU rollout kernel matches
its CPU reference rollout for rollout, with the leaf eval rows it builds on
the device.

py/tools/gpu_parity.py

These checks run on the GPU, so they are not part of py/run_tests.py (see
CLAUDE.md for when to run them). Each family is one test binary under
target/engine, built by py/build.py, and one Python generator that exports a
tiny random-init model with its PyTorch reference outputs. This script
generates each fixture once into a scratch dir and hands it to the binary,
which builds TensorRT plans and compares; a binary run by hand with no
argument generates its own fixture for every case instead. The binaries run
one after another, so the check holds the GPU for one plan build at a time.

The rollout check runs gpu_rollout_demo small, with the leaf eval fixture's
model restamped with the engine's own leave values (real.onnx), so it
compares the device's leaf rows and values with the CPU's too.
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
    ("test_leaf_eval_parity", ["scripts.deep_bot_leaf_eval.gen_parity_fixture"]),
]
# The rollout check: gpu_rollout_demo over a few candidates, with the leaf
# family's fixture model, at an early, a middle and a late position (with
# tiles still in the bag), so the kernel's equity-bound pruning meets varied
# boards.
ROLLOUT_CHECK = "gpu_rollout_demo"
ROLLOUT_FLAGS = ["--candidates", "6", "--rollouts", "256", "--leaf-check", "1792"]
ROLLOUT_PLIES = [2, 10, 16]
ROLLOUT_LEAF_FAMILY = "test_leaf_eval_parity"


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


def check_rollouts(scratch: Path) -> bool:
    binary = ENGINE_DIR / ROLLOUT_CHECK
    if not binary.is_file():
        print(f"ERROR: {binary} not found. Run py/build.py first.", file=sys.stderr)
        return False
    model = scratch / ROLLOUT_LEAF_FAMILY / "real.onnx"
    flags = [*ROLLOUT_FLAGS, "--leaf-model", str(model)]
    return all([run([str(binary), *flags, "--plies", str(plies)]) for plies in ROLLOUT_PLIES])


def main():
    with tempfile.TemporaryDirectory(prefix="scribblez_gpu_parity_") as scratch:
        results = [(name, check_family(name, gen, Path(scratch))) for name, gen in FAMILIES]
        results.append((ROLLOUT_CHECK, check_rollouts(Path(scratch))))
    print("\nSummary")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()

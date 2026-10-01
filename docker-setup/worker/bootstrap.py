#!/usr/bin/env python3
"""Worker bootstrap: unpack the code+binary bundle and hand off to it.

Baked into the worker Docker image as its entrypoint, so it must stay
dependency-free (stdlib + the image's g++) and stable -- all iterable worker
logic lives in the bundle (py/cloud/worker_entrypoint.py).

The dashboard creates the container, copies its payload into PAYLOAD_DIR over
ssh, and only then starts it (WorkerManager._run_ssh_container):

  bundle.tar.gz      the bundle's tarball for this machine's CPU arch
  positions.tar.gz   the eval datasets, for the roles that read them

Steps:
  1. Unpack the payload at /workspace/repo.
  2. Record this machine's CPU microarchitecture as SCZ_HOST_ARCH (the
     dashboard sets SCZ_BUNDLE_ID and SCZ_BUNDLE_ARCH), and exec the bundle's
     worker entrypoint.

A restarted container unpacks the same payload again.
"""

import os
import subprocess
import sys
import tarfile
from pathlib import Path

# The payload contract with the dashboard. A dashboard copying payloads in
# needs images of at least this protocol: the push records it
# (build_and_push_worker_image.py), and the dashboard refuses an older image
# (py/cloud/runtime_abi.py BOOTSTRAP_PROTOCOL). Bump both together.
BOOTSTRAP_PROTOCOL = 2

PAYLOAD_DIR = Path("/opt/scribblez/payload")
REPO_ROOT = Path("/workspace/repo")
WORKER_ENTRYPOINT = REPO_ROOT / "py" / "cloud" / "worker_entrypoint.py"


def detect_host_arch() -> str:
    """This machine's CPU microarchitecture as a GCC -march value. Mirrors
    py/build.py detect_host_arch() (not importable before the bundle lands)."""
    result = subprocess.run(
        ["g++", "-march=native", "-Q", "--help=target"], capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("-march="):
            arch = line.split("=", 1)[1].strip()
            if arch and arch != "native":
                return arch
    sys.exit("bootstrap: could not determine host CPU arch via g++.")


def main():
    bundle = PAYLOAD_DIR / "bundle.tar.gz"
    if not bundle.is_file():
        sys.exit(
            f"bootstrap: no bundle at {bundle}. The dashboard copies one in before it starts "
            "the container; a container started any other way has none."
        )
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("bundle.tar.gz", "positions.tar.gz"):
        if (PAYLOAD_DIR / name).is_file():
            with tarfile.open(PAYLOAD_DIR / name) as tar:
                tar.extractall(REPO_ROOT, filter="data")
    host_arch = detect_host_arch()
    print(
        f"bootstrap: unpacked bundle {os.environ.get('SCZ_BUNDLE_ID')} "
        f"({os.environ.get('SCZ_BUNDLE_ARCH')}; host arch {host_arch})"
    )
    os.environ["SCZ_HOST_ARCH"] = host_arch
    print(f"bootstrap: handing off to {WORKER_ENTRYPOINT}")
    os.execv(sys.executable, [sys.executable, str(WORKER_ENTRYPOINT)])


if __name__ == "__main__":
    main()

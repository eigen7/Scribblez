#!/usr/bin/env python3
"""Worker-pod bootstrap: fetch a code+binary bundle and hand off to it.

Baked into the worker Docker image as its entrypoint, so it must stay
dependency-free (stdlib + the image's rclone/g++) and stable -- all iterable
worker logic lives in the bundle it downloads (py/cloud/worker_entrypoint.py).

Steps:
  1. Resolve the bundle reference in $SCZ_BUNDLE ("latest" or a concrete
     bundle_id) against the R2 bucket.
  2. Pick the bundle tarball matching this machine's CPU microarchitecture,
     falling back to the generic x86-64 tarball when the exact arch has no
     build (the entrypoint records the fallback so it reaches the operator).
  3. Download + unpack it at /workspace/repo, then exec the bundle's worker
     entrypoint, passing the outcome via SCZ_BUNDLE_ID / SCZ_HOST_ARCH /
     SCZ_BUNDLE_ARCH.

Required env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
R2_BUCKET, plus whatever the worker entrypoint itself needs (SCZ_TAG etc.).
"""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path("/workspace/repo")
WORKER_ENTRYPOINT = REPO_ROOT / "py" / "cloud" / "worker_entrypoint.py"

# Mirrors py/cloud/bundles.py (GENERIC_ARCH, bucket layout) and py/cloud/r2.py
# (rclone env config). Duplicated because this script runs before any bundle
# -- and therefore any py/ module -- exists on the machine.
GENERIC_ARCH = "x86-64"
BUNDLES_PREFIX = "bundles"


def rclone_env() -> dict[str, str]:
    endpoint = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return os.environ | {
        "RCLONE_CONFIG_R2_TYPE": "s3",
        "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
        "RCLONE_CONFIG_R2_ENDPOINT": endpoint,
        "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true",
    }


def rclone(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], env=rclone_env(), capture_output=capture, text=True)


def bundles_path(*parts: str) -> str:
    return "/".join([f"r2:{os.environ['R2_BUCKET']}", BUNDLES_PREFIX, *parts])


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


def resolve_bundle_id(ref: str) -> str:
    if ref != "latest":
        return ref
    res = rclone("cat", bundles_path("LATEST"), capture=True)
    if res.returncode != 0:
        sys.exit(f"bootstrap: could not read bundles/LATEST: {res.stderr.strip()}")
    return res.stdout.strip()


def pick_arch(bundle_id: str, host_arch: str) -> str:
    res = rclone("cat", bundles_path(bundle_id, "manifest.json"), capture=True)
    if res.returncode != 0:
        sys.exit(f"bootstrap: bundle '{bundle_id}' has no manifest: {res.stderr.strip()}")
    archs = json.loads(res.stdout)["archs"]
    if host_arch in archs:
        return host_arch
    if GENERIC_ARCH in archs:
        print(
            f"bootstrap: no '{host_arch}' build in bundle {bundle_id}; "
            f"falling back to '{GENERIC_ARCH}'. Add '{host_arch}' to "
            "SUPPORTED_ARCHS in py/build.py for full speed."
        )
        return GENERIC_ARCH
    sys.exit(f"bootstrap: bundle {bundle_id} has neither '{host_arch}' nor '{GENERIC_ARCH}'.")


def main():
    bundle_id = resolve_bundle_id(os.environ.get("SCZ_BUNDLE", "latest"))
    host_arch = detect_host_arch()
    bundle_arch = pick_arch(bundle_id, host_arch)

    tarball = f"bundle-{bundle_arch}.tar.gz"
    print(f"bootstrap: fetching {bundle_id}/{tarball} (host arch: {host_arch})")
    res = rclone("copyto", bundles_path(bundle_id, tarball), f"/tmp/{tarball}")
    if res.returncode != 0:
        sys.exit(f"bootstrap: bundle download failed with code {res.returncode}")

    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(f"/tmp/{tarball}") as tar:
        tar.extractall(REPO_ROOT, filter="data")
    os.remove(f"/tmp/{tarball}")

    os.environ.update(SCZ_BUNDLE_ID=bundle_id, SCZ_HOST_ARCH=host_arch, SCZ_BUNDLE_ARCH=bundle_arch)
    print(f"bootstrap: handing off to {WORKER_ENTRYPOINT}")
    os.execv(sys.executable, [sys.executable, str(WORKER_ENTRYPOINT)])


if __name__ == "__main__":
    main()

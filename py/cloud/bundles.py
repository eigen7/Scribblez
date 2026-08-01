"""Code+binary bundles: how compiled engine artifacts reach cloud workers.

A bundle is one tarball per supported CPU microarchitecture (the engine is
compiled per-arch under target/archs/<arch>/; see py/build.py), each holding
that arch's binaries plus the arch-independent py/ tree, uploaded to the
results bucket under bundles/<bundle_id>/. A worker pod downloads and unpacks
the tarball matching its CPU at startup (docker-setup/worker/bootstrap.py),
falling back to the generic-x86-64 tarball, so code iteration never requires
rebuilding or re-pushing the worker Docker image.

Bucket layout:

    bundles/LATEST                        text file holding the newest bundle_id
    bundles/<bundle_id>/manifest.json     git provenance (sha, dirty flag) + arch list
    bundles/<bundle_id>/bundle-<arch>.tar.gz   one per arch in SUPPORTED_ARCHS

The bundle_id is "<git-sha-12>[-dirty]-<content-hash-8>"; the content hash
makes successive pushes from the same (possibly dirty) tree distinct.
"""

import hashlib
import json
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from build import SUPPORTED_ARCHS, arch_build_dir

from cloud.credentials import R2Credentials
from cloud.r2 import bucket_path, rclone

REPO_ROOT = Path("/workspace/repo")

# Engine artifacts shipped to workers. Sourced from each arch's build dir and
# placed at target/engine/<name> inside the tarball -- the fixed path all
# python and C++ tooling references. The py/ tree rides along in full (minus
# caches).
BUNDLE_BINARY_NAMES = [
    "play_game",
    "sim_obs_tool",
    "move_set_eval_target_generator",
    "libscribblez_ffi.so",
]

# Arch whose tarball any worker can run (baseline x86-64 codegen); workers
# whose exact arch has no tarball fall back to this one.
GENERIC_ARCH = "x86-64"

BUNDLES_PREFIX = "bundles"
LATEST_NAME = "LATEST"

_TAR_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


@dataclass(frozen=True)
class BundleManifest:
    bundle_id: str
    git_sha: str
    git_dirty: bool
    archs: list[str]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(p in _TAR_EXCLUDE_DIRS for p in parts):
        return None
    return info


def arch_tarball_name(arch: str) -> str:
    return f"bundle-{arch}.tar.gz"


def _create_arch_tarball(arch: str, out_dir: Path) -> Path:
    engine_dir = Path(arch_build_dir(arch)) / "engine"
    for name in BUNDLE_BINARY_NAMES:
        assert (engine_dir / name).is_file(), (
            f"{engine_dir / name} not built; run py/build.py --build-for-all-archs first"
        )
    tar_path = out_dir / arch_tarball_name(arch)
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in BUNDLE_BINARY_NAMES:
            tar.add(engine_dir / name, arcname=f"target/engine/{name}")
        tar.add(REPO_ROOT / "py", arcname="py", filter=_tar_filter)
    return tar_path


def create_bundle(out_dir: Path) -> tuple[list[Path], BundleManifest]:
    """Build one tarball per supported arch plus manifest.json under `out_dir`
    from the current tree, returning (tarball paths, manifest)."""
    tarballs = [_create_arch_tarball(arch, out_dir) for arch in SUPPORTED_ARCHS]
    digest = hashlib.sha256()
    for tar_path in tarballs:
        digest.update(tar_path.read_bytes())
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    bundle_id = f"{sha[:12]}{'-dirty' if dirty else ''}-{digest.hexdigest()[:8]}"
    manifest = BundleManifest(
        bundle_id=bundle_id, git_sha=sha, git_dirty=dirty, archs=list(SUPPORTED_ARCHS)
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return tarballs, manifest


def push_bundle(r2: R2Credentials) -> BundleManifest:
    """Create a bundle from the current tree, upload it, and point LATEST at it."""
    with tempfile.TemporaryDirectory(prefix="scribblez-bundle-") as tmp:
        tmp_dir = Path(tmp)
        tarballs, manifest = create_bundle(tmp_dir)
        dest = bucket_path(r2, BUNDLES_PREFIX, manifest.bundle_id)
        for path in [*tarballs, tmp_dir / "manifest.json"]:
            res = rclone(r2, "copyto", str(path), f"{dest}/{path.name}")
            assert res.returncode == 0, f"upload of {path.name} failed"
        res = rclone(
            r2,
            "rcat",
            bucket_path(r2, BUNDLES_PREFIX, LATEST_NAME),
            capture=True,
            input_text=manifest.bundle_id + "\n",
        )
        assert res.returncode == 0, f"updating {LATEST_NAME} failed: {res.stderr}"
    return manifest


def resolve_bundle_id(r2: R2Credentials, ref: str) -> str:
    """Resolve a bundle reference ("latest" or a concrete bundle_id) to a
    concrete bundle_id, verifying that its manifest exists in the bucket."""
    if ref == "latest":
        res = rclone(r2, "cat", bucket_path(r2, BUNDLES_PREFIX, LATEST_NAME), capture=True)
        assert res.returncode == 0, "no bundles pushed yet (bundles/LATEST missing)"
        ref = res.stdout.strip()
    res = rclone(r2, "cat", bucket_path(r2, BUNDLES_PREFIX, ref, "manifest.json"), capture=True)
    assert res.returncode == 0, f"bundle '{ref}' not found in bucket"
    return ref

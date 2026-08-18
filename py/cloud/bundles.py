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

Deployment is not a step an operator has to remember: `deploy_current_tree`
builds every supported arch and pushes only when the bucket's LATEST does not
already carry this tree, and the dashboard calls it before launching a worker
that runs from a bundle. Its "is this tree already deployed?" test is the
manifest's `source_hash` -- a digest of the exact files a bundle ships -- not
the bundle_id, which is deliberately fresh on every push.
"""

import hashlib
import json
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from dataclasses import fields as fields_of
from pathlib import Path

from build import SUPPORTED_ARCHS, arch_build_dir, build_all_archs, detect_host_arch
from scribblez.hardware import default_thread_count
from scribblez.paths import REPO_ROOT

from cloud.credentials import R2Credentials
from cloud.r2 import bucket_path, rclone

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
    # Digest of the files this bundle shipped (see source_hash). Empty for
    # bundles pushed before the field existed, which therefore never match a
    # local tree -- the first deployment against one pushes.
    source_hash: str = ""


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


def _shipped_files() -> list[tuple[str, Path]]:
    """Every file a bundle ships, as (identity, path): each supported arch's
    binaries plus the shared py/ tree, named as they appear inside a tarball."""
    files = [
        (f"{arch}/{name}", Path(arch_build_dir(arch)) / "engine" / name)
        for arch in SUPPORTED_ARCHS
        for name in BUNDLE_BINARY_NAMES
    ]
    files += [
        (str(path.relative_to(REPO_ROOT)), path)
        for path in (REPO_ROOT / "py").rglob("*")
        if path.is_file() and not set(path.parts) & _TAR_EXCLUDE_DIRS
    ]
    return sorted(files)


def source_hash(cache: dict | None = None) -> str | None:
    """A digest of the tree a bundle would ship right now, or None when some
    arch is unbuilt (nothing to compare until a build produces it).

    This is the deployment test, and it covers compiled binaries rather than
    git state: two pushes of one tree get different bundle_ids by design, and
    a `-dirty` sha says a tree changed without saying into what. Hashing 20 MB
    costs ~80 ms, which is too much for a status poll, so `cache` (owned by
    the caller, keyed by path) holds each file's digest against its size and
    mtime and reduces a repeat call to a stat walk.
    """
    digest = hashlib.sha256()
    for identity, path in _shipped_files():
        try:
            stamp = path.stat()
        except FileNotFoundError:
            return None
        key = (stamp.st_size, stamp.st_mtime_ns)
        cached = cache.get(path) if cache is not None else None
        if cached is None or cached[0] != key:
            cached = (key, hashlib.sha256(path.read_bytes()).hexdigest())
            if cache is not None:
                cache[path] = cached
        digest.update(f"{identity}:{cached[1]}\n".encode())
    return digest.hexdigest()


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
        bundle_id=bundle_id,
        git_sha=sha,
        git_dirty=dirty,
        archs=list(SUPPORTED_ARCHS),
        source_hash=source_hash(),
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


def read_manifest(r2: R2Credentials, bundle_id: str) -> BundleManifest | None:
    """Bundle `bundle_id`'s manifest, or None if the bucket has no such bundle."""
    path = bucket_path(r2, BUNDLES_PREFIX, bundle_id, "manifest.json")
    res = rclone(r2, "cat", path, capture=True)
    if res.returncode != 0:
        return None
    fields = json.loads(res.stdout)
    known = {f.name for f in fields_of(BundleManifest)}
    return BundleManifest(**{k: v for k, v in fields.items() if k in known})


def latest_manifest(r2: R2Credentials) -> BundleManifest | None:
    """The manifest of the bundle at LATEST, or None if nothing is pushed."""
    res = rclone(r2, "cat", bucket_path(r2, BUNDLES_PREFIX, LATEST_NAME), capture=True)
    return read_manifest(r2, res.stdout.strip()) if res.returncode == 0 else None


def build_all_supported_archs(jobs: int | None = None):
    """Rebuild every arch a bundle ships, incrementally (a no-op costs
    seconds). Release, matching py/build.py's default."""
    failed = build_all_archs(
        SUPPORTED_ARCHS, "Release", jobs or default_thread_count(), detect_host_arch()
    )
    assert not failed, f"build failed for arch(s): {', '.join(sorted(failed))}"


def deploy_current_tree(
    r2: R2Credentials, *, jobs: int | None = None, cache=None
) -> BundleManifest:
    """Make LATEST be this tree, and return the manifest it now points at.

    Building first is not optional: the fingerprint covers compiled binaries,
    so pushing without it would ship an arch nobody rebuilt under a fresh,
    current-looking bundle id -- the exact deception this whole mechanism
    exists to prevent. The upload is skipped when LATEST already carries this
    tree, so a redeploy of unchanged code leaves running tasks pinned where
    they are.
    """
    build_all_supported_archs(jobs)
    current = source_hash(cache)
    assert current is not None, "a build just ran; every arch's binaries must exist"
    latest = latest_manifest(r2)
    if latest is not None and latest.source_hash == current:
        return latest
    return push_bundle(r2)


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

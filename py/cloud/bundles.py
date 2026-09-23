"""Bundles: how compiled engine binaries and the py/ tree reach remote workers.

A bundle holds one tarball per CPU microarchitecture its machines need (the
engine is compiled per arch under target/archs/<arch>/; see py/build.py).
Each tarball has that arch's binaries plus the arch-independent py/ tree. At
startup a worker container downloads the tarball matching its CPU, or the
generic x86-64 one (docker-setup/worker/bootstrap.py). Code changes therefore
never require rebuilding the worker Docker image.

Bucket layout:

    bundles/LATEST                             the newest bundle_id
    bundles/<bundle_id>/manifest.json          BundleManifest
    bundles/<bundle_id>/bundle-<arch>.tar.gz   one per arch in the manifest
    deps/positions-<digest>.tar.gz             the eval datasets, by content

The eval datasets (EVAL_POSITIONS_DIRS in scribblez/paths.py, ~40 MB) stay
out of the tarballs, where every deploy would upload a copy per arch. They are
uploaded once per content version under deps/, before the manifest that names
that version, so a manifest never points at a missing object. Train roles
fetch them in cloud/worker_deps.py.

A bundle_id is "<git-sha-12>[-dirty]-<content-hash-8>"; the content hash keeps
successive pushes from the same dirty tree distinct.

The dashboard calls `deploy_current_tree` before it launches a worker that
runs from a bundle, so deploying is never a manual step. It pushes only when
LATEST does not already hold this tree, judged by the manifest's
`source_hash` (a digest of exactly the files a bundle ships) rather than by
the bundle_id, which is new on every push.
"""

import hashlib
import json
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from dataclasses import fields as fields_of
from pathlib import Path

from build import arch_build_dir, build_all_archs, detect_host_arch
from scribblez.hardware import default_thread_count
from scribblez.paths import EVAL_POSITIONS_DIRS, REPO_ROOT

from cloud.credentials import R2Credentials
from cloud.r2 import bucket_path, rclone

# Engine artifacts shipped to workers, placed at target/engine/<name> in the
# tarball: the path all Python and C++ tooling expects them at.
BUNDLE_BINARY_NAMES = [
    "play_game",
    "sim_obs_tool",
    "sim_candidate_survey_tool",
    "move_set_eval_target_generator",
    "libscribblez_ffi.so",
]

# The baseline arch any x86-64 CPU can run: the fallback for a worker whose
# own arch has no tarball.
GENERIC_ARCH = "x86-64"

BUNDLES_PREFIX = "bundles"
LATEST_NAME = "LATEST"
DEPS_PREFIX = "deps"

_TAR_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


@dataclass(frozen=True)
class BundleManifest:
    bundle_id: str
    git_sha: str
    git_dirty: bool
    archs: list[str]
    # Digest of the files this bundle shipped (see source_hash). A manifest
    # without it never matches a local tree, so deploying over it pushes.
    source_hash: str = ""
    # Digest of the eval datasets this bundle was deployed with; it names their
    # deps/ object (eval_positions_object). A train role refuses a bundle
    # without it.
    eval_positions: str = ""


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
            f"{engine_dir / name} not built; run py/build.py --archs {arch} first"
        )
    tar_path = out_dir / arch_tarball_name(arch)
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in BUNDLE_BINARY_NAMES:
            tar.add(engine_dir / name, arcname=f"target/engine/{name}")
        tar.add(REPO_ROOT / "py", arcname="py", filter=_tar_filter)
    return tar_path


def _shipped_files(archs: list[str]) -> list[tuple[str, Path]]:
    """Every file a deploy of `archs` ships, as (identity, path): each arch's
    binaries, the py/ tree, and the eval datasets. The identity is
    "<arch>/<name>" for a binary and the repo-relative path otherwise."""
    files = [
        (f"{arch}/{name}", Path(arch_build_dir(arch)) / "engine" / name)
        for arch in archs
        for name in BUNDLE_BINARY_NAMES
    ]
    files += [
        (str(path.relative_to(REPO_ROOT)), path)
        for path in (REPO_ROOT / "py").rglob("*")
        if path.is_file() and not set(path.parts) & _TAR_EXCLUDE_DIRS
    ]
    return sorted(files + eval_positions_files())


def eval_positions_files() -> list[tuple[str, Path]]:
    """The eval datasets' files as (path under the repo root, path)."""
    return sorted(
        (str(path.relative_to(REPO_ROOT)), path)
        for root in EVAL_POSITIONS_DIRS
        for path in root.rglob("*")
        if path.is_file()
    )


def eval_positions_digest(files=None) -> str:
    """A digest of the eval datasets' file names and contents. It names their
    deps/ object, and a worker checks its copy against it."""
    digest = hashlib.sha256()
    for identity, path in eval_positions_files() if files is None else files:
        digest.update(f"{identity}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
    return digest.hexdigest()[:16]


def eval_positions_object(digest: str) -> str:
    """The deps/ object holding the eval datasets at `digest`."""
    return f"{DEPS_PREFIX}/positions-{digest}.tar.gz"


def create_eval_positions_tarball(out_dir: Path) -> Path:
    """The eval datasets as a tarball that unpacks under a repo root."""
    tar_path = out_dir / "positions.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for identity, path in eval_positions_files():
            tar.add(path, arcname=identity)
    return tar_path


def push_eval_positions(r2: R2Credentials, digest: str):
    """Upload the eval datasets under their digest, unless the bucket already
    has an object by that name (content-addressed, so the same bytes)."""
    dest = bucket_path(r2, eval_positions_object(digest))
    if rclone(r2, "lsf", dest, capture=True).stdout.strip():
        return
    with tempfile.TemporaryDirectory(prefix="scribblez-positions-") as tmp:
        tar_path = create_eval_positions_tarball(Path(tmp))
        res = rclone(r2, "copyto", str(tar_path), dest)
        assert res.returncode == 0, "upload of the eval datasets failed"


def source_hash(archs: list[str], cache: dict | None = None) -> str | None:
    """A digest of the files a bundle of `archs` would ship right now, or None
    when a file is missing (an arch is unbuilt).

    This is the "already deployed?" test. It hashes the shipped files rather
    than using git state, because a `-dirty` sha says the tree changed without
    saying into what. Hashing ~20 MB takes ~80 ms, too slow for a status poll,
    so the caller may pass a `cache` (path -> ((size, mtime), digest)) that
    reduces a repeat call to a stat walk.
    """
    digest = hashlib.sha256()
    for identity, path in _shipped_files(archs):
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


def create_bundle(out_dir: Path, archs: list[str]) -> tuple[list[Path], BundleManifest]:
    """Write one tarball per arch plus manifest.json into `out_dir`, from the
    current tree."""
    archs = sorted(set(archs))
    tarballs = [_create_arch_tarball(arch, out_dir) for arch in archs]
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
        archs=archs,
        source_hash=source_hash(archs),
        eval_positions=eval_positions_digest(),
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return tarballs, manifest


def push_bundle(r2: R2Credentials, archs: list[str]) -> BundleManifest:
    """Create a bundle of `archs` from the current tree, upload it, and point
    LATEST at it."""
    with tempfile.TemporaryDirectory(prefix="scribblez-bundle-") as tmp:
        tmp_dir = Path(tmp)
        tarballs, manifest = create_bundle(tmp_dir, archs)
        push_eval_positions(r2, manifest.eval_positions)
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


def build_archs(archs: list[str], jobs: int | None = None):
    """Build `archs` in Release, as py/build.py does by default. Incremental,
    so an up-to-date tree costs seconds."""
    failed = build_all_archs(
        sorted(set(archs)), "Release", jobs or default_thread_count(), detect_host_arch()
    )
    assert not failed, f"build failed for arch(s): {', '.join(sorted(failed))}"


def deploy_current_tree(
    r2: R2Credentials, archs: list[str], *, jobs: int | None = None, cache=None
) -> BundleManifest:
    """Point LATEST at the current tree built for `archs` (the archs of the
    machines that will run it) and return its manifest.

    It always builds first. Otherwise a stale binary for some arch would ship
    under a fresh bundle id, which looks current but is not. The upload is
    skipped when LATEST already covers every arch in `archs` and its
    source_hash matches this tree (hashed over LATEST's own arch list), so
    redeploying unchanged code leaves running tasks on the bundle they have.
    """
    archs = sorted(set(archs))
    assert archs, "a bundle needs at least one arch: the machines that will run it"
    build_archs(archs, jobs)
    latest = latest_manifest(r2)
    if latest is not None and set(archs) <= set(latest.archs):
        if source_hash(latest.archs, cache) == latest.source_hash:
            return latest
    return push_bundle(r2, archs)


def resolve_bundle_id(r2: R2Credentials, ref: str) -> str:
    """Resolve "latest" or a bundle_id to a bundle_id whose manifest exists
    in the bucket."""
    if ref == "latest":
        res = rclone(r2, "cat", bucket_path(r2, BUNDLES_PREFIX, LATEST_NAME), capture=True)
        assert res.returncode == 0, "no bundles pushed yet (bundles/LATEST missing)"
        ref = res.stdout.strip()
    res = rclone(r2, "cat", bucket_path(r2, BUNDLES_PREFIX, ref, "manifest.json"), capture=True)
    assert res.returncode == 0, f"bundle '{ref}' not found in bucket"
    return ref

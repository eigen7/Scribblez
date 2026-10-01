"""Bundles: how compiled engine binaries and the py/ tree reach remote workers.

A bundle holds one tarball per CPU microarchitecture its machines need (the
engine is compiled per arch under target/archs/<arch>/; see py/build.py).
Each tarball has that arch's binaries plus the arch-independent py/ tree. The
dashboard copies the tarball for a container's machine into the container
before starting it, and the image's bootstrap unpacks it
(docker-setup/worker/bootstrap.py). Code changes therefore never require
rebuilding the worker Docker image.

Bundles live in a store on the controller (STORE_REL under the mount):

    LATEST                             the newest bundle_id
    <bundle_id>/manifest.json          BundleManifest
    <bundle_id>/bundle-<arch>.tar.gz   one per arch in the manifest
    deps/positions-<digest>.tar.gz     the eval datasets, by content

The eval datasets (EVAL_POSITIONS_DIRS in scribblez/paths.py, ~40 MB) stay
out of the tarballs, where every bundle would hold a copy per arch. They are
written once per content version under deps/, and copied only into the
containers of the roles that read them (cloud/worker_deps.py).

A bundle_id is "<git-sha-12>[-dirty]-<content-hash-8>"; the content hash keeps
successive bundles of the same dirty tree distinct.

The dashboard calls `deploy_current_tree` before it launches a worker that
runs from a bundle, so deploying is never a manual step. It writes a bundle
only when LATEST does not already hold this tree, judged by the manifest's
`source_hash` (a digest of exactly the files a bundle ships) rather than by
the bundle_id, which is new on every bundle.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import uuid
from dataclasses import asdict, dataclass
from dataclasses import fields as fields_of
from pathlib import Path

from build import arch_build_dir, build_all_archs, detect_host_arch
from scribblez.hardware import default_thread_count
from scribblez.paths import EVAL_POSITIONS_DIRS, REPO_ROOT

# Engine artifacts shipped to workers, placed at target/engine/<name> in the
# tarball: the path all Python and C++ tooling expects them at.
BUNDLE_BINARY_NAMES = [
    "play_game",
    "sim_obs_tool",
    "sim_candidate_survey_tool",
    "move_set_eval_target_generator",
    "libscribblez_ffi.so",
]

# The bundle store, relative to the mount root.
STORE_REL = "cloud/bundles"
LATEST_NAME = "LATEST"
DEPS_DIR = "deps"

# Bundles kept in the store beyond those a task is pinned to: the newest few,
# so a redeploy back to recent code finds its bundle without a rebuild.
KEEP_NEWEST = 5

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
    # tarball under deps/ (eval_positions_path).
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


def eval_positions_path(store: Path, digest: str) -> Path:
    """The tarball of the eval datasets at `digest`, in the store."""
    return store / DEPS_DIR / f"positions-{digest}.tar.gz"


def create_eval_positions_tarball(out_dir: Path) -> Path:
    """The eval datasets as a tarball that unpacks under a repo root."""
    tar_path = out_dir / "positions.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for identity, path in eval_positions_files():
            tar.add(path, arcname=identity)
    return tar_path


def write_eval_positions(store: Path, digest: str):
    """Write the eval datasets' tarball under its digest, unless the store
    already has it (content-addressed, so the same bytes)."""
    dest = eval_positions_path(store, digest)
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    work = dest.parent / f".tmp-{uuid.uuid4().hex[:12]}"
    work.mkdir()
    try:
        create_eval_positions_tarball(work).replace(dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)


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


def write_bundle(store: Path, archs: list[str]) -> BundleManifest:
    """Create a bundle of `archs` from the current tree in the store, and
    point LATEST at it. The bundle's directory appears whole, by one rename."""
    store.mkdir(parents=True, exist_ok=True)
    work = store / f".tmp-{uuid.uuid4().hex[:12]}"
    work.mkdir()
    try:
        _, manifest = create_bundle(work, archs)
        write_eval_positions(store, manifest.eval_positions)
        dest = store / manifest.bundle_id
        if not dest.exists():
            work.rename(dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _write_atomic(store / LATEST_NAME, manifest.bundle_id + "\n")
    return manifest


def _write_atomic(path: Path, text: str):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_manifest(store: Path, bundle_id: str) -> BundleManifest | None:
    """Bundle `bundle_id`'s manifest, or None if the store has no such bundle."""
    try:
        fields = json.loads((store / bundle_id / "manifest.json").read_text())
    except FileNotFoundError:
        return None
    known = {f.name for f in fields_of(BundleManifest)}
    return BundleManifest(**{k: v for k, v in fields.items() if k in known})


def latest_manifest(store: Path) -> BundleManifest | None:
    """The manifest of the bundle at LATEST, or None if none was written."""
    try:
        bundle_id = (store / LATEST_NAME).read_text().strip()
    except FileNotFoundError:
        return None
    return read_manifest(store, bundle_id)


def arch_tarball(store: Path, bundle_id: str, arch: str) -> Path:
    """Bundle `bundle_id`'s tarball for `arch`, which must exist."""
    path = store / bundle_id / arch_tarball_name(arch)
    assert path.is_file(), f"bundle {bundle_id} has no {arch} build; redeploy"
    return path


def prune(store: Path, keep: set[str]):
    """Delete the store's bundles other than `keep` and the KEEP_NEWEST
    newest, and the eval-dataset tarballs no remaining bundle names."""
    if not store.is_dir():
        return
    bundle_dirs = sorted(
        (d for d in store.iterdir() if d.is_dir() and (d / "manifest.json").is_file()),
        key=lambda d: (d / "manifest.json").stat().st_mtime,
    )
    keep = set(keep) | {d.name for d in bundle_dirs[-KEEP_NEWEST:]}
    for d in bundle_dirs:
        if d.name not in keep:
            shutil.rmtree(d, ignore_errors=True)
    named = {m.eval_positions for b in keep if (m := read_manifest(store, b)) is not None}
    for tarball in (store / DEPS_DIR).glob("positions-*.tar.gz"):
        if tarball.name.removeprefix("positions-").removesuffix(".tar.gz") not in named:
            tarball.unlink(missing_ok=True)


def build_archs(archs: list[str], jobs: int | None = None):
    """Build `archs` in Release, as py/build.py does by default. Incremental,
    so an up-to-date tree costs seconds."""
    failed = build_all_archs(
        sorted(set(archs)), "Release", jobs or default_thread_count(), detect_host_arch()
    )
    assert not failed, f"build failed for arch(s): {', '.join(sorted(failed))}"


def deploy_current_tree(
    store: Path, archs: list[str], *, jobs: int | None = None, cache=None
) -> BundleManifest:
    """Point LATEST at the current tree built for `archs` (the archs of the
    machines that will run it) and return its manifest.

    It always builds first. Otherwise a stale binary for some arch would ship
    under a fresh bundle id, which looks current but is not. Writing a bundle
    is skipped when LATEST already covers every arch in `archs` and its
    source_hash matches this tree (hashed over LATEST's own arch list), so
    redeploying unchanged code leaves running tasks on the bundle they have.
    """
    archs = sorted(set(archs))
    assert archs, "a bundle needs at least one arch: the machines that will run it"
    build_archs(archs, jobs)
    latest = latest_manifest(store)
    if latest is not None and set(archs) <= set(latest.archs):
        if source_hash(latest.archs, cache) == latest.source_hash:
            return latest
    return write_bundle(store, archs)

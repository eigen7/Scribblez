"""A release (docs/plans/deep_bot_generations.md, "Terms"): the three models'
exports published together, under the tag at
releases/release_<k>/{leave,leaf,policy}.onnx with a manifest written last.
A release exists once its manifest does, so a reader never takes two models
of different releases. Release 0 is the seeds'; release k has trained
through generation k - 1.

A release's files never change once its manifest exists, so a copy is the
release wherever it is. A trainer on an ssh machine hands each to the
controller as a delivered set (`deliver`, `install`; workloads/base.py,
DeliveredSet), and the controller pushes the newest to the generators that do
not share the trainer's tree (`newest_files`).
"""

import json
import os
import shutil
from pathlib import Path

from scribblez.deep_bot import KINDS
from scribblez.paths import TagPaths

MANIFEST = "manifest.json"

# Where a remote trainer links each release for the controller to take.
OUTBOX = "release_outbox"


def releases_dir(paths: TagPaths) -> Path:
    return paths.root / "releases"


def release_dir(paths: TagPaths, release: int) -> Path:
    return releases_dir(paths) / f"release_{release:06d}"


def export_path(paths: TagPaths, release: int, kind: str) -> Path:
    return release_dir(paths, release) / f"{kind}.onnx"


def export_rel(release: int, kind: str) -> str:
    """Release `release`'s export of `kind`, relative to the tag root or to
    wherever else its files (below) are laid out."""
    return f"releases/release_{release:06d}/{kind}.onnx"


def write_manifest(paths: TagPaths, release: int, tag: str, rows: dict[str, int]) -> Path:
    """Complete release `release`: its exports must be in place. `rows` is
    each model's rows-clock at the release."""
    manifest = {
        "tag": tag,
        "release": release,
        "models": {kind: f"{kind}.onnx" for kind in KINDS},
        "rows_trained": rows,
    }
    path = release_dir(paths, release) / MANIFEST
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)
    return path


def read_manifest(paths: TagPaths, release: int) -> dict:
    return json.loads((release_dir(paths, release) / MANIFEST).read_text())


def files(release: int) -> list[str]:
    """Release `release`'s files, tag-relative, the manifest last."""
    exports = [export_rel(release, kind) for kind in KINDS]
    return [*exports, f"releases/release_{release:06d}/{MANIFEST}"]


def newest_files(paths: TagPaths, params) -> list[str]:
    """The RoleSpec.tag_files of a role that plays the newest release: its
    files, or none before release 0."""
    number = newest(paths)
    return [] if number is None else files(number)


def deliver(paths: TagPaths, release: int):
    """Link release `release` into the outbox as a set for the controller,
    the manifest, its marker, last. The release stays where it is, for the
    trainer and the generators beside it."""
    for rel in files(release):
        dest = paths.root / OUTBOX / Path(rel).relative_to("releases")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)
        os.link(paths.root / rel, dest)


def install(set_dir: Path, paths: TagPaths):
    """Put a delivered release where the tag keeps its releases. A set
    without its manifest arrived torn and is discarded, and so is a release
    the tag already holds, which a trainer restored onto a new machine
    publishes again: copies of the one it replaces may already be playing."""
    dest = releases_dir(paths) / set_dir.name
    if (set_dir / MANIFEST).is_file() and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(set_dir, dest)
    shutil.rmtree(set_dir, ignore_errors=True)


def exists(paths: TagPaths, release: int) -> bool:
    return (release_dir(paths, release) / MANIFEST).exists()


def numbers(paths: TagPaths) -> list[int]:
    """The complete releases' numbers, ascending."""
    return sorted(
        int(d.name.removeprefix("release_"))
        for d in releases_dir(paths).glob("release_*")
        if (d / MANIFEST).exists()
    )


def newest(paths: TagPaths) -> int | None:
    """The newest complete release's number, or None before release 0."""
    return max(numbers(paths), default=None)

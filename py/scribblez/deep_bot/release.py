"""A release (docs/plans/subtree_bot_generations.md, "Terms"): a tag's models'
exports published together, under the tag at releases/release_<k>/<kind>.onnx,
beside any files of the workload's own (a Layout names both), with a manifest
written last. deep_bot_generation's holds the leave inference model and the
leaf network.
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
from dataclasses import dataclass
from pathlib import Path

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
    wherever else its files (Layout.files) are laid out."""
    return f"{release_rel(release)}/{kind}.onnx"


def release_rel(release: int) -> str:
    """Release `release`'s directory, relative to the tag root."""
    return f"releases/release_{release:06d}"


@dataclass(frozen=True)
class Layout:
    """What a workload's release holds: an export per model kind, and the
    files of the workload's own written beside them before the manifest."""

    kinds: tuple[str, ...]
    extras: tuple[str, ...] = ()

    def write_manifest(self, paths: TagPaths, release: int, tag: str, rows: dict[str, int]):
        """Complete release `release`: its exports and extras must be in
        place. `rows` is each model's rows-clock at the release."""
        manifest = {
            "tag": tag,
            "release": release,
            "models": {kind: f"{kind}.onnx" for kind in self.kinds},
            "rows_trained": rows,
        }
        path = release_dir(paths, release) / MANIFEST
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(path)

    def files(self, release: int) -> list[str]:
        """Release `release`'s files, tag-relative, the manifest last."""
        rel = release_rel(release)
        exports = [export_rel(release, kind) for kind in self.kinds]
        return [*exports, *(f"{rel}/{name}" for name in self.extras), f"{rel}/{MANIFEST}"]

    def newest_files(self, paths: TagPaths) -> list[str]:
        """The files of the newest release, for a role that plays it
        (RoleSpec.tag_files), or none before release 0."""
        number = newest(paths)
        return [] if number is None else self.files(number)

    def deliver(self, paths: TagPaths, release: int):
        """Link release `release` into the outbox as a set for the
        controller, the manifest, its marker, last. The release stays where
        it is, for the trainer and the generators beside it."""
        for rel in self.files(release):
            dest = paths.root / OUTBOX / Path(rel).relative_to("releases")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.unlink(missing_ok=True)
            os.link(paths.root / rel, dest)


def read_manifest(paths: TagPaths, release: int) -> dict:
    return json.loads((release_dir(paths, release) / MANIFEST).read_text())


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

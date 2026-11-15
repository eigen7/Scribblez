"""A release (docs/plans/deep_bot_generations.md, "Terms"): the three models'
exports published together, under the tag at
releases/release_<k>/{leave,leaf,policy}.onnx with a manifest written last.
A release exists once its manifest does, so a reader never takes two models
of different releases. Release 0 is the seeds'; release k has trained
through generation k - 1.
"""

import json
from pathlib import Path

from scribblez.deep_bot import KINDS
from scribblez.paths import TagPaths

MANIFEST = "manifest.json"


def releases_dir(paths: TagPaths) -> Path:
    return paths.root / "releases"


def release_dir(paths: TagPaths, release: int) -> Path:
    return releases_dir(paths) / f"release_{release:06d}"


def export_path(paths: TagPaths, release: int, kind: str) -> Path:
    return release_dir(paths, release) / f"{kind}.onnx"


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


def newest(paths: TagPaths) -> int | None:
    """The newest complete release's number, or None before release 0."""
    done = [
        int(d.name.removeprefix("release_"))
        for d in releases_dir(paths).glob("release_*")
        if (d / MANIFEST).exists()
    ]
    return max(done, default=None)

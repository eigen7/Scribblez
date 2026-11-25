"""A multi-model tag's seeds (docs/plans/subtree_bot_generations.md, "Seeds and
warm start"): one rolling checkpoint per model, copied under the new tag at
seeds/<kind>.pt when it is created, so that the tag does not depend on its
sources afterwards. A trainer on an ssh machine gets them from the controller
(`files`, behind a role's RoleSpec.tag_files).

A seed is a standalone workload's tag (`source`), or a multi-model trainer's
own checkpoint, whose models units.seed_part tells apart.

Importable without torch: the dashboard installs the seeds at task creation.
"""

import shutil
from pathlib import Path

from scribblez.deep_bot import LEAF, LEAVE
from scribblez.params import ParamsError
from scribblez.paths import TagPaths

# The standalone workload each model's seed is a tag of.
WORKLOADS = {LEAVE: "leave_inference", LEAF: "deep_bot_leaf_eval"}


def seed_path(paths: TagPaths, kind: str) -> Path:
    return paths.root / "seeds" / f"{kind}.pt"


def files(paths: TagPaths, kinds: tuple[str, ...]) -> list[str]:
    """The seeds of `kinds`, tag-relative."""
    return [str(seed_path(paths, kind).relative_to(paths.root)) for kind in kinds]


def checkpoint_of(tag: str, workload: str, mount_root: Path) -> Path:
    """The rolling checkpoint of `workload`'s `tag`. Raises ParamsError if it
    has none."""
    path = TagPaths(tag, workload, mount_root).rolling_checkpoint
    if not path.is_file():
        raise ParamsError(f"{workload} tag '{tag}' has no checkpoint at {path}")
    return path


def source(params, kind: str, mount_root: Path) -> Path:
    """The rolling checkpoint of `kind`'s standalone seed tag, the param
    `seed_<kind>_tag`. Raises ParamsError if the tag is unset or has no
    checkpoint."""
    tag = getattr(params, f"seed_{kind}_tag")
    if not tag:
        raise ParamsError(f"seed_{kind}_tag is required")
    return checkpoint_of(tag, WORKLOADS[kind], mount_root)


def install(paths: TagPaths, sources: dict[str, Path]):
    """Copy each kind's seed from `sources` under the tag at `paths`."""
    for kind, src in sources.items():
        dest = seed_path(paths, kind)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dest)

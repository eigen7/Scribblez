"""A deep_bot tag's seeds (docs/plans/deep_bot_generations.md, "Seeds and
warm start"): the rolling checkpoints of three tags of the standalone
workloads, copied under the new tag at seeds/<kind>.pt when it is created, so
that the tag does not depend on its sources afterwards. A trainer on an ssh
machine gets them from the controller (`tag_files`).

Importable without torch: the dashboard installs the seeds at task creation.
"""

import shutil
from pathlib import Path

from scribblez.deep_bot import KINDS, LEAF, LEAVE, POLICY
from scribblez.params import ParamsError
from scribblez.paths import TagPaths

# The standalone workload each model's seed is a tag of.
WORKLOADS = {LEAVE: "leave_inference", LEAF: "deep_bot_leaf_eval", POLICY: "policy_set"}


def seed_path(paths: TagPaths, kind: str) -> Path:
    return paths.root / "seeds" / f"{kind}.pt"


def tag_files(paths: TagPaths, params) -> list[str]:
    """The train role's RoleSpec.tag_files: the three seeds."""
    return [str(seed_path(paths, kind).relative_to(paths.root)) for kind in KINDS]


def source(params, kind: str, mount_root: Path) -> Path:
    """The rolling checkpoint `kind`'s seed tag holds. Raises ParamsError if
    the tag is unset or has no checkpoint."""
    tag = getattr(params, f"seed_{kind}_tag")
    if not tag:
        raise ParamsError(f"seed_{kind}_tag is required")
    path = TagPaths(tag, WORKLOADS[kind], mount_root).rolling_checkpoint
    if not path.is_file():
        raise ParamsError(f"{WORKLOADS[kind]} tag '{tag}' has no checkpoint at {path}")
    return path


def install(paths: TagPaths, params):
    """Copy the three seeds under the tag at `paths`. Raises ParamsError,
    before copying anything, if a seed is missing."""
    sources = {kind: source(params, kind, paths.mount_root) for kind in KINDS}
    for kind, src in sources.items():
        dest = seed_path(paths, kind)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dest)

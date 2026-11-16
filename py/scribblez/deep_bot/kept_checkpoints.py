"""The checkpoints a deep_bot tag keeps: every `keep_checkpoint_every`th
release's, as a seed for a later tag (docs/plans/deep_bot_generations.md,
"The trainer"). A trainer on an ssh machine hands each to the controller as a
delivered set of one file (workloads/base.py, DeliveredSet), so the kept
checkpoints outlive its machine.

Importable without torch: the controller installs what a trainer delivers.
"""

import os
import shutil
from pathlib import Path

from scribblez.paths import TagPaths

# Where a remote trainer links each kept checkpoint for the controller to take.
OUTBOX = "checkpoint_outbox"
# The set's one file. A link appears whole, so it is its own marker.
NAME = "model.pt"


def keep(paths: TagPaths, generation: int, remote: bool):
    """Keep a copy of the rolling checkpoint as release `generation`'s, and
    from a `remote` trainer link it into the outbox."""
    kept = paths.checkpoint_path(generation)
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.rolling_checkpoint, kept)
    if remote:
        dest = paths.root / OUTBOX / f"gen_{generation:06d}" / NAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)
        os.link(kept, dest)


def install(set_dir: Path, paths: TagPaths):
    """Put a delivered checkpoint where the tag keeps them."""
    if (set_dir / NAME).is_file():
        dest = paths.checkpoint_path(int(set_dir.name.removeprefix("gen_")))
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(set_dir / NAME, dest)
    shutil.rmtree(set_dir, ignore_errors=True)

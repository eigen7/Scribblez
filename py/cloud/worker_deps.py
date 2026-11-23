"""Data a worker reads that neither the worker image nor a bundle contains,
and how a worker on another machine gets it.

- The mount files (MOUNT_FILES): the engine's default lexicon and Macondo's
  strategy data (leave values, pre-endgame table, win table), which the
  engine and the trainers read at fixed paths under /workspace/mount. The
  .kwg encodes a copyrighted wordlist, so no image, bundle or store holds it.
  The dashboard copies its own machine's files into each ssh container with
  the payload (`payload_files`, WorkerManager._copy_payload), and the worker
  entrypoint puts them in place before the role starts (`install`). A local
  worker reads the controller's own, which setup_wizard.py installed.
- the eval datasets a train role scores checkpoints against: git-tracked, so
  a checkout has them; a bundle-run worker's container gets the bundle's copy
  from the dashboard, and its bootstrap unpacks it (cloud/bundles.py).
"""

import shutil
from pathlib import Path

from scribblez.paths import EVAL_POSITIONS_DIRS

# The engine's default lexicon (scribblez::Lexicon::Params::name). Workers run
# the engine with its defaults, so this is the one lexicon they need.
DEFAULT_LEXICON = "NWL23"

# The mount files, relative to the mount root: the lexicon, and the Macondo
# files the engine (lexicon/hasty_equity.cpp, sim/win_pct_table.cpp) and the
# leaf eval's Macondo baseline (leave_inference/win_table.py) read.
MOUNT_FILES = (
    f"lexica/{DEFAULT_LEXICON}.kwg",
    f"macondo/data/strategy/{DEFAULT_LEXICON}/leaves.klv2",
    "macondo/data/strategy/default/preendgame.json",
    "macondo/data/strategy/default/winpct.csv",
)

# Where the mount files sit in a container's payload (runtime_abi.PAYLOAD_DIR).
PAYLOAD_MOUNT = "mount"


def payload_files(mount_root: Path) -> dict[str, Path]:
    """The mount files under `mount_root`, keyed by their path in the
    payload."""
    return {f"{PAYLOAD_MOUNT}/{rel}": mount_root / rel for rel in MOUNT_FILES}


def install(payload_dir: Path, mount_root: Path):
    """Copy the payload's mount files under `mount_root`. Copied rather than
    moved, so a restarted container, which starts from the same payload,
    installs them again."""
    for rel in MOUNT_FILES:
        dest = mount_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(payload_dir / PAYLOAD_MOUNT / rel, dest)


def fetch_eval_positions():
    """Check the eval datasets (EVAL_POSITIONS_DIRS) are under the repo root:
    a checkout's own copy for a local slot or a CLI, and for a bundle-run
    worker the bundle's copy, which the dashboard copies into the container of
    each role that reads them (WorkerManager._run_ssh_container)."""
    missing = [d for d in EVAL_POSITIONS_DIRS if not d.is_dir()]
    assert not missing, f"eval datasets missing: {missing}"

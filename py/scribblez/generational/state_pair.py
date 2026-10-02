"""The trainer's state as a named pair, and the cursor rule for every copy of it.

A trainer's resumable state is its rolling checkpoint (checkpoints/model.pt)
plus its cursor (train_state.json, {rows_trained, generation_index}). Two
separately copied files can tear: a copy may carry generation N+1's weights
beside generation N's cursor. So a copy of the state always travels as one
**pair**, named by its generation (`state/gen_NNNNNN/{model.pt,
train_state.json}`), with the cursor written last as the pair's commit
marker.

The **cursor rule** decides between copies: a pair replaces the installed
state only if its `rows_trained` is at least the installed one's. Rows trained
only ever grow, so this is a logical clock. No machine's wall clock is
involved, and the controller never has to load a torch file to compare. A
stale copy, such as a volume left over from an earlier assignment, therefore
can never overwrite fresher state.

`deliver` sends a trainer's pair through its sink. `install` applies the rule.
`take_seed` installs the pair the controller pushes into a new trainer
container (SEED_DIR), and `install_seed` is every trainer's call to it.
"""

import json
import os
import shutil
import time
from pathlib import Path

from scribblez.generational import lifecycle
from scribblez.paths import TagPaths
from scribblez.train_common import timed_print

STATE_DIR = "state"
MODEL_NAME = "model.pt"
CURSOR_NAME = "train_state.json"

# Where the controller pushes its copy of the state into a new trainer
# container, under the tag root (WorkerManager._seed_state).
SEED_DIR = ".seed"

# How long a trainer told a seed is coming waits for it. The controller pushes
# it right after creating the container, so a wait this long means the push
# failed.
SEED_WAIT_SECONDS = 600
SEED_POLL_SECONDS = 2


def pair_rel(generation: int) -> str:
    """The pair's path under the tag root."""
    return f"{STATE_DIR}/gen_{generation:06d}"


def rows_trained(cursor: Path) -> int | None:
    """The rows a cursor file records, or None when it is absent or unreadable."""
    try:
        return int(json.loads(cursor.read_text())["rows_trained"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return None


def installed_rows(paths: TagPaths) -> int:
    """The installed state's rows trained; -1 when there is no checkpoint, so
    that any pair beats it."""
    if not paths.rolling_checkpoint.exists():
        return -1
    return int(lifecycle.read_train_state(paths).get("rows_trained", -1))


def install(pair_dir: Path, paths: TagPaths) -> bool:
    """Install the pair in `pair_dir` as the tag's checkpoint and cursor if it
    has at least as many rows trained as the installed state. Each file moves
    by rename, the cursor last, so a reader never sees a cursor ahead of its
    weights. The pair directory is consumed either way. Returns whether the
    pair was installed."""
    rows = rows_trained(pair_dir / CURSOR_NAME)
    newer = rows is not None and (pair_dir / MODEL_NAME).is_file()
    newer = newer and rows >= installed_rows(paths)
    if newer:
        paths.rolling_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pair_dir / MODEL_NAME, paths.rolling_checkpoint)
        os.replace(pair_dir / CURSOR_NAME, paths.train_state_path)
    shutil.rmtree(pair_dir, ignore_errors=True)
    return newer


def install_seed(paths: TagPaths):
    """A trainer's first step: install the checkpoint and cursor the controller
    pushed into a new container (SCZ_STATE_SEED) when they beat the ones on
    this machine (the cursor rule). A fresh machine takes them, and a machine
    holding fresher state keeps its own."""
    if take_seed(paths, expected=os.environ.get("SCZ_STATE_SEED") == "1"):
        timed_print("installed the controller's checkpoint and cursor")


def take_seed(paths: TagPaths, expected: bool) -> bool:
    """Install the pair the controller pushed into this trainer's container,
    under the cursor rule. When `expected` (SCZ_STATE_SEED) and this machine
    holds no checkpoint of its own, wait for the seed's cursor, its commit
    marker. A restarted container, or a home volume that already holds state,
    starts at once. Returns whether the seed was installed."""
    seed = paths.root / SEED_DIR
    if _awaiting_seed(paths, expected):
        timed_print(
            f"waiting up to {SEED_WAIT_SECONDS} s for the controller's checkpoint and cursor"
        )
    deadline = time.monotonic() + SEED_WAIT_SECONDS
    while _awaiting_seed(paths, expected):
        assert time.monotonic() < deadline, (
            f"no state seed arrived in {SEED_WAIT_SECONDS} s: the controller's push into this "
            "container failed (the slot's row shows why); remove and re-add the slot to retry"
        )
        time.sleep(SEED_POLL_SECONDS)
    return (seed / CURSOR_NAME).exists() and install(seed, paths)


def _awaiting_seed(paths: TagPaths, expected: bool) -> bool:
    """Whether a trainer told a seed is coming still lacks both its own
    checkpoint and the seed's cursor."""
    return (
        expected
        and not paths.rolling_checkpoint.exists()
        and not (paths.root / SEED_DIR / CURSOR_NAME).exists()
    )


def snapshot(path: Path, generation: int) -> Path:
    """A hard link to `path`'s current version, for a delivery that may run
    after the trainer has rewritten `path`. Rewrites replace the file rather
    than modifying it, so the link keeps this version."""
    snap = path.with_name(f"{path.name}.gen{generation}")
    snap.unlink(missing_ok=True)
    os.link(path, snap)
    return snap


def deliver(sink, model_snapshot: Path, cursor_snapshot: Path, generation: int):
    """Send a generation's state pair through `sink`: the model, then the cursor
    (the pair's commit marker), each moved or uploaded from its snapshot. Then
    remove the older pairs, which the cursor rule would never pick over this
    one."""
    rel = pair_rel(generation)
    sink.push_file(model_snapshot, f"{rel}/{MODEL_NAME}")
    sink.push_file(cursor_snapshot, f"{rel}/{CURSOR_NAME}")
    for name in sink.list_dirs(STATE_DIR):
        if f"{STATE_DIR}/{name}" < rel:
            sink.remove_tree(f"{STATE_DIR}/{name}")

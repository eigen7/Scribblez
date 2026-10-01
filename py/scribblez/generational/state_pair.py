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
restore takes the copy with the most rows, wherever it is. A stale copy, such
as a bucket that fell behind the machine it came from or a volume left over
from an earlier assignment, therefore can never overwrite fresher state.

`deliver` sends a trainer's pair through its records sink. `install` applies
the rule. `restore` fetches through a sink the newest pair that beats the
installed state, then installs it.
"""

import json
import os
import shutil
from pathlib import Path

from scribblez.paths import TagPaths

STATE_DIR = "state"
MODEL_NAME = "model.pt"
CURSOR_NAME = "train_state.json"


def pair_rel(generation: int) -> str:
    """The pair's path under the tag root (or the bucket's tag prefix)."""
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
    rows = rows_trained(paths.train_state_path)
    return -1 if rows is None else rows


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


# The pre-pair layout: the checkpoint and cursor as two separate objects.
# Trainers on bundles from before the pairs still write it, so a restore
# considers it as one more candidate. It goes when the bucket does.
LEGACY_PAIR = ("checkpoints/model.pt", "train_state.json")


def restore(paths: TagPaths, sink) -> bool:
    """Install the newest pair the sink holds if it beats the installed state
    (the cursor rule), reading only the candidates' small cursors until one
    wins. Returns whether one was installed. A local sink holds the installed
    state itself, so it has nothing newer to offer."""
    if sink.kind == "local":
        return False
    incoming = paths.root / ".incoming" / STATE_DIR
    shutil.rmtree(incoming, ignore_errors=True)
    candidates = [
        (f"{STATE_DIR}/{name}/{MODEL_NAME}", f"{STATE_DIR}/{name}/{CURSOR_NAME}")
        for name in sink.list_dirs(STATE_DIR)
    ] + [LEGACY_PAIR]
    best, best_rows = None, installed_rows(paths)
    for i, (model_rel, cursor_rel) in enumerate(candidates):
        cursor = incoming / f"{i}.json"
        if sink.fetch_file(cursor_rel, cursor):
            rows = rows_trained(cursor)
            if rows is not None and rows > best_rows:
                best, best_rows = (model_rel, cursor), rows
    if best is None:
        return False
    model_rel, cursor = best
    pair = incoming / "pair"
    pair.mkdir(parents=True, exist_ok=True)
    if not sink.fetch_file(model_rel, pair / MODEL_NAME):
        return False  # removed since it was listed; the next restore looks again
    os.replace(cursor, pair / CURSOR_NAME)
    installed = install(pair, paths)
    shutil.rmtree(incoming, ignore_errors=True)
    return installed

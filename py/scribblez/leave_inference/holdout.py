"""The leave inference trainer's held-out games (docs/plans/deep_bot_v0.md,
"Held-out data: regenerated").

The trainer plays them itself, HastyBot against HastyBot at a fixed seed, so
any machine it runs on regenerates the same set and nothing durable has to
follow it there. play_game plays game g at seed base + g, and a single thread
writes the games in that order, so the files are reproducible too, but for
each game's metadata `sampled_turn`, which the writer picks at random and
the content hash leaves out.

Generators run at seed 0, which draws a fresh random base per chunk, so the
two never share games.

Each .slog gets its .forgone sidecar, as the generators' chunks do. The
sidecars are reproducible as they stand, and the content hash covers them.
"""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scribblez.ffi import struct_dtype
from scribblez.hardware import default_thread_count
from scribblez.leave_inference import forgone
from scribblez.selfplay import run_games
from scribblez.train_common import timed_print

# The held-out games' base seed. Fixed, so every machine plays the same games;
# nonzero, since seed 0 draws a random base.
SEED = 1
# Written last, so a directory without it is a partial set to regenerate.
DONE_MARKER = "complete"


@dataclass(frozen=True)
class HoldoutSet:
    dir: Path
    slogs: tuple[Path, ...]
    content_hash: str  # over the files and their sidecars in name order, sampled_turn zeroed


def _reproducible_bytes(path: Path) -> bytes:
    """The file's bytes with every game's sampled_turn zeroed."""
    header, meta = struct_dtype("SlogFileHeader"), struct_dtype("SlogGameMetadata")
    buf = bytearray(path.read_bytes())
    num_games = int(np.frombuffer(buf, header, count=1)[0]["num_games"])
    metas = np.frombuffer(buf, meta, count=num_games, offset=header.itemsize)
    metas["sampled_turn"] = 0
    return bytes(buf)


def _content_hash(slogs: list[Path]) -> str:
    h = hashlib.sha256()
    for path in slogs:
        h.update(_reproducible_bytes(path))
        h.update(path.with_suffix(forgone.SUFFIX).read_bytes())
    return h.hexdigest()[:16]


def ensure(root: Path, games: int, seed: int, player_spec: str) -> HoldoutSet:
    """The held-out set of `games` games at `seed`, under `root`, playing it
    first if it is missing or incomplete."""
    if seed == 0:
        raise ValueError("the held-out seed must be nonzero; seed 0 draws a random one")
    out = root / f"seed{seed}-games{games}"
    if not (out / DONE_MARKER).exists():
        shutil.rmtree(out, ignore_errors=True)
        timed_print(f"playing {games} held-out games at seed {seed} into {out}")
        run_games(out, games, threads=1, player_spec=player_spec, seed=seed)
        forgone.write_sidecars(out, default_thread_count())
        (out / DONE_MARKER).touch()
    slogs = sorted(out.glob("*.slog"))
    return HoldoutSet(out, tuple(slogs), _content_hash(slogs))

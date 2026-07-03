"""Filling and validating one generation of self-play games.

`ensure_generation` is the single entry point the orchestrator calls to obtain a
complete generation directory: it reuses an already-complete generation as-is, or
(re)generates a missing/partial one from a clean slate. The actual game
generation and game-counting are injected as callables so this decision logic is
testable without the C++ `play_game` binary; the production wiring in the
orchestrator supplies `run_games` and a .slog-header counter.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from ..paths import TagPaths
from . import lifecycle

# (gen_dir, num_games, seed) -> subprocess return code (0 on success).
GenerateFn = Callable[[Path, int, int], int]
# (gen_dir) -> number of games committed to disk.
CountFn = Callable[[Path], int]


def ensure_generation(
    paths: TagPaths,
    index: int,
    *,
    target_games: int,
    seed: int,
    player_spec: str,
    generate_fn: GenerateFn,
    count_fn: CountFn,
) -> Path:
    """Return generation `index`'s directory, filled and marked complete.

    A generation whose manifest already marks it complete is reused untouched.
    Otherwise -- missing, or a partial left by a crash -- the directory is wiped
    to a clean slate (discarding any partial games, as their producer's state is
    gone), a fresh `generating` manifest is written, `generate_fn` plays the
    games, the committed count is measured with `count_fn`, and the manifest is
    flipped to complete. Raises if `generate_fn` reports a nonzero exit.
    """
    gen_dir = paths.generation_dir(index)
    if lifecycle.is_complete(gen_dir):
        return gen_dir

    shutil.rmtree(gen_dir, ignore_errors=True)
    lifecycle.open_generation(
        paths, index, target_games=target_games, seed=seed, player_spec=player_spec
    )
    rc = generate_fn(gen_dir, target_games, seed)
    if rc != 0:
        raise RuntimeError(f"game generation for generation {index} failed (exit code {rc})")
    lifecycle.mark_complete(gen_dir, count_fn(gen_dir))
    return gen_dir

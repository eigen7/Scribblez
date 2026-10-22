"""A small policy_set chunk for tests: HastyBot games surveyed into .pool
sidecars under uniform beliefs, against a logistic win table.

Deterministic: the games are seeded, and the .slog is renamed to a fixed stem
before the survey, which seeds itself from the stem (policy_set.write_pools).
play_game names its files from the clock, so without the rename every run
would survey with a different seed.
"""

from pathlib import Path

import numpy as np
from scribblez.leave_inference import win_table
from scribblez.selfplay import run_games
from scribblez.workloads import policy_set
from scribblez.workloads.policy_set import BELIEFS_UNIFORM, PolicySetParams

GAMES_SEED = 1
STEM = "chunk"


def surveyed_chunk(tmp: Path, *, rollouts: int, positions_per_game: int) -> Path:
    """One game in `tmp`/chunk, surveyed into `positions_per_game` pools of
    `rollouts` rollouts per candidate."""
    chunk = tmp / "chunk"
    assert (
        run_games(
            chunk, num_games=1, threads=2, player_spec="--type=hastybot-endgame", seed=GAMES_SEED
        )
        == 0
    )
    (slog,) = chunk.glob("*.slog")
    slog.rename(chunk / f"{STEM}.slog")
    table = tmp / "table.csv"
    spread = np.arange(-win_table.max_spread(), win_table.max_spread() + 1)
    grid = np.repeat(1 / (1 + np.exp(-spread / 30))[:, None], win_table.max_unseen() + 1, axis=1)
    win_table.write(table, grid, {"tag": "li", "generation": 3})
    p = PolicySetParams(
        leave_beliefs=BELIEFS_UNIFORM, rollouts=rollouts, positions_per_game=positions_per_game
    )
    policy_set.write_pools(p, table, None, chunk, threads=4)
    return chunk

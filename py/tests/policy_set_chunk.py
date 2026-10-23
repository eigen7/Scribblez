"""A small policy_set chunk for tests: HastyBot games labeled into .pool
sidecars under uniform beliefs, by policy_set_pool_tool's CPU reference with
its synthetic leaf (the suites hide the GPU).

Deterministic: the games are seeded, and the .slog is renamed to a fixed stem
before labeling, which seeds itself from the stem (policy_set.write_pools).
play_game names its files from the clock, so without the rename every run
would label with a different seed.
"""

from pathlib import Path

from scribblez.selfplay import run_games
from scribblez.workloads import policy_set
from scribblez.workloads.policy_set import BELIEFS_UNIFORM, LEAF_SYNTHETIC, PolicySetParams

GAMES_SEED = 1
STEM = "chunk"


def surveyed_chunk(tmp: Path, *, rollouts: int, positions_per_game: int) -> Path:
    """One game in `tmp`/chunk, labeled into `positions_per_game` pools of up
    to `rollouts` rollouts per candidate."""
    chunk = tmp / "chunk"
    assert (
        run_games(
            chunk, num_games=1, threads=2, player_spec="--type=hastybot-endgame", seed=GAMES_SEED
        )
        == 0
    )
    (slog,) = chunk.glob("*.slog")
    slog.rename(chunk / f"{STEM}.slog")
    p = PolicySetParams(
        leave_beliefs=BELIEFS_UNIFORM, rollouts=rollouts, positions_per_game=positions_per_game
    )
    policy_set.write_pools(p, LEAF_SYNTHETIC, None, chunk, threads=4)
    return chunk

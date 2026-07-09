"""Running the C++ `play_game` self-play binary.

The library home for shelling out to play_game, shared by every driver that
generates .slog data: the one-shot CLI (scripts/generate_data.py), the
kill-test cycle, and the generational generator role.
"""

import subprocess
from pathlib import Path

PLAY_GAME = "/workspace/repo/target/engine/play_game"


def hasty_player_spec(temperature: float = 0.0, top_k: int = 10, temp_min_bag: int = 0) -> str:
    """The `--player` value for a HastyBot seat, optionally temperature-sampled
    over equity (temperature > 0) for exploration."""
    if temperature > 0:
        return (
            f"--type=hastybot --temperature={temperature} "
            f"--top-k={top_k} --temperature-min-bag={temp_min_bag}"
        )
    return "--type=hastybot"


def run_games(
    out_dir: Path,
    num_games: int,
    games_per_file: int,
    threads: int,
    player_spec: str,
    seed: int = 0,
    random_opening_mean: float = 0.0,
) -> int:
    """Run `num_games` self-play games, logging .slog files to out_dir.

    Both seats use `player_spec` (the value of a `--player` flag); each seat is a
    fresh agent, so two neural seats draw independent sampling seeds. `seed` is
    the PRNG seed passed to play_game; 0 (the default) lets the binary pick one
    from std::random_device, so successive default-seeded runs differ.
    `random_opening_mean` > 0 opens each game with K uniformly-random plies
    (K ~ round(Exp(mean)) per game) before the agents take over; positions
    before the last random ply are excluded from the training-eligible region.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # fmt: off
    cmd = [
        PLAY_GAME,
        "--player", player_spec,
        "--player", player_spec,
        "--binary-log-dir", str(out_dir),
        "--games-per-file", str(games_per_file),
        "--games", str(num_games),
        "--threads", str(threads),
        "--seed", str(seed),
        "--random-opening-mean", str(random_opening_mean),
    ]
    # fmt: on
    cmd_str = " ".join(f'"{t}"' if " " in t else t for t in cmd)
    print(f"Running: {cmd_str}")
    return subprocess.run(cmd, capture_output=False).returncode

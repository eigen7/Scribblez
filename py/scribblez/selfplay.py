"""Launching the C++ `play_game` self-play binary, shared by every Python
driver that generates games (scripts/generate_data.py, the dashboard
workloads, match evaluation)."""

import subprocess
from pathlib import Path

from scribblez.paths import ENGINE_DIR

PLAY_GAME = str(ENGINE_DIR / "play_game")


def run_play_game(args: list[str]) -> int:
    """Run play_game with `args`, echoing a shell-pasteable command line first.
    Returns the exit code."""
    cmd = [PLAY_GAME, *args]
    cmd_str = " ".join(f'"{t}"' if " " in t else t for t in cmd)
    print(f"Running: {cmd_str}")
    return subprocess.run(cmd, capture_output=False).returncode


def hasty_player_spec(temperature: float = 0.0, top_k: int = 10, endgame: bool = False) -> str:
    """The `--player` value for a HastyBot seat. temperature > 0 samples over
    the top-k moves by equity, for exploration. `endgame` selects
    hastybot-endgame, which hands empty-bag positions to the endgame solver."""
    bot_type = "hastybot-endgame" if endgame else "hastybot"
    if temperature > 0:
        return f"--type={bot_type} --temperature={temperature} --top-k={top_k}"
    return f"--type={bot_type}"


def run_games(
    out_dir: Path,
    num_games: int,
    threads: int,
    player_spec: str,
    seed: int = 0,
    random_opening_mean: float = 0.0,
    face_up_leaves: bool = False,
) -> int:
    """Run `num_games` self-play games, logging .slog files to out_dir.

    Both seats are built from `player_spec` (a `--player` value) as separate
    agents, so two sampling seats draw independent seeds. seed=0 lets
    play_game pick a random seed, so successive default runs differ.
    `random_opening_mean` > 0 opens each game with a random number of
    uniformly random plies (mean `random_opening_mean`) for position
    diversity; positions before the last random ply are not training-eligible.
    `face_up_leaves` plays the variant in which retained tiles are public.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # fmt: off
    args = [
        "--player", player_spec,
        "--player", player_spec,
        "--binary-log-dir", str(out_dir),
        "--games", str(num_games),
        "--threads", str(threads),
        "--seed", str(seed),
        "--random-opening-mean", str(random_opening_mean),
    ]
    # fmt: on
    if face_up_leaves:
        args.append("--face-up-leaves")
    return run_play_game(args)

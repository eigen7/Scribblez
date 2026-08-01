"""Running the C++ `play_game` self-play binary.

The library home for shelling out to play_game, shared by every driver that
generates .slog data: the one-shot CLI (scripts/generate_data.py), the
kill-test cycle, and the generational generator role.
"""

import subprocess
from pathlib import Path

PLAY_GAME = "/workspace/repo/target/engine/play_game"


def run_play_game(args: list[str]) -> int:
    """Run the play_game binary with `args`, echoing the command line (quoted
    the way a shell would want it) first. Returns the exit code."""
    cmd = [PLAY_GAME, *args]
    cmd_str = " ".join(f'"{t}"' if " " in t else t for t in cmd)
    print(f"Running: {cmd_str}")
    return subprocess.run(cmd, capture_output=False).returncode


def hasty_player_spec(temperature: float = 0.0, top_k: int = 10, endgame: bool = False) -> str:
    """The `--player` value for a HastyBot seat, optionally temperature-sampled
    over equity (temperature > 0) for exploration. `endgame` selects the
    hastybot-endgame agent, which plays identically until the bag empties and
    then hands the position to the endgame solver at the engine's default
    node budget."""
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

    Both seats use `player_spec` (the value of a `--player` flag); each seat is a
    fresh agent, so two neural seats draw independent sampling seeds. `seed` is
    the PRNG seed passed to play_game; 0 (the default) lets the binary pick one
    from std::random_device, so successive default-seeded runs differ.
    `random_opening_mean` > 0 opens each game with K uniformly-random plies
    (K ~ round(Exp(mean)) per game) before the agents take over; positions
    before the last random ply are excluded from the training-eligible region.
    `face_up_leaves` plays the variant in which retained tiles are public; the
    .slog header records it.
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

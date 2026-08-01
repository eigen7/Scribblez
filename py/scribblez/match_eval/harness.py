"""Head-to-head match harness: paired play_game rounds (roadmap E2).

The harness owns the match-play discipline so experiments do not reinvent it.
Games run in the engine's --paired mode (games 2k and 2k+1 share a game seed
with the seats swapped, cancelling per-seed tile luck), the base seed is
caller-fixed so different arms -- e.g. successive training generations against
one baseline -- face identical deals, and outcomes come back as the pair
scores scribblez.stats' sequential test consumes.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scribblez.selfplay import PLAY_GAME

# Game scores for the player under test, and the per-pair average of two of
# them; see scribblez.stats.PAIR_SCORES.
_WIN, _DRAW, _LOSS = 1.0, 0.5, 0.0


@dataclass(frozen=True)
class RoundResult:
    """One paired round from the perspective of player 0 (the player under
    test): per-pair scores plus the per-game W/D/L tally behind them."""

    pair_scores: list[float]
    wins: int
    draws: int
    losses: int


def _game_score_for_player0(line: dict) -> float:
    p0_seat = line["seat_players"].index(0)
    own, opp = line["seat_scores"][p0_seat], line["seat_scores"][1 - p0_seat]
    if own == opp:
        return _DRAW
    return _WIN if own > opp else _LOSS


def _pair_by_seed(lines: list[dict]) -> list[float]:
    """Collapse per-game records to per-pair scores. Games arrive in completion
    order (the engine plays them on a thread pool), so the pair identity is the
    shared game seed, not adjacency."""
    by_seed: dict[int, list[float]] = {}
    for line in lines:
        by_seed.setdefault(line["seed"], []).append(_game_score_for_player0(line))
    for seed, scores in by_seed.items():
        if len(scores) != 2:
            raise RuntimeError(f"seed {seed} produced {len(scores)} games, expected a pair")
    return [(a + b) / 2.0 for a, b in by_seed.values()]


def play_round(
    player0_spec: str,
    player1_spec: str,
    num_pairs: int,
    threads: int,
    seed: int,
    results_file: Path,
    face_up_leaves: bool = False,
) -> RoundResult:
    """Play `num_pairs` mirrored pairs of player0 vs player1 and return player
    0's results. `seed` must be nonzero (0 asks the engine for entropy, which
    would unfix the deals the pairing discipline relies on)."""
    if seed == 0:
        raise ValueError("seed 0 means 'random' to play_game; matches need fixed seeds")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    # fmt: off
    cmd = [
        PLAY_GAME,
        "--player", player0_spec,
        "--player", player1_spec,
        "--games", str(2 * num_pairs),
        "--threads", str(threads),
        "--seed", str(seed),
        "--paired",
        "--results-file", str(results_file),
    ]
    # fmt: on
    if face_up_leaves:
        cmd.append("--face-up-leaves")
    cmd_str = " ".join(f'"{t}"' if " " in t else t for t in cmd)
    print(f"Running: {cmd_str}")
    rc = subprocess.run(cmd, capture_output=False).returncode
    if rc != 0:
        raise RuntimeError(f"play_game failed with exit code {rc}")

    lines = [json.loads(ln) for ln in results_file.read_text().splitlines() if ln.strip()]
    if len(lines) != 2 * num_pairs:
        raise RuntimeError(f"expected {2 * num_pairs} game records, got {len(lines)}")
    scores = [_game_score_for_player0(line) for line in lines]
    return RoundResult(
        pair_scores=_pair_by_seed(lines),
        wins=scores.count(_WIN),
        draws=scores.count(_DRAW),
        losses=scores.count(_LOSS),
    )

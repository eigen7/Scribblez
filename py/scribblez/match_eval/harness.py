"""Head-to-head match play: rounds of mirrored game pairs through play_game.

Games run in play_game's --paired mode: each pair shares a game seed with the
seats swapped, cancelling most per-seed tile luck. The caller fixes the base
seed, so different contenders (arms, or successive training generations
against one baseline) face identical deals. Results come back as pair scores
for scribblez.stats.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from scribblez.selfplay import run_play_game

# Per-game scores for the player under test; a pair score is the mean of two
# (scribblez.stats.PAIR_SCORES).
_WIN, _DRAW, _LOSS = 1.0, 0.5, 0.0


@dataclass(frozen=True)
class RoundResult:
    """One round's results for player 0, the player under test."""

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
    """Per-pair scores. Games are recorded in completion order across threads,
    so pairs are matched by their shared seed, not by adjacency."""
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
    """Play `num_pairs` mirrored pairs of player0 vs player1. `seed` must be
    nonzero: play_game treats 0 as "pick a random seed", which would break the
    fixed deals comparisons rely on."""
    if seed == 0:
        raise ValueError("seed 0 means 'random' to play_game; matches need fixed seeds")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    # fmt: off
    args = [
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
        args.append("--face-up-leaves")
    rc = run_play_game(args)
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

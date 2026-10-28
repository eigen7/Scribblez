"""Per-arm results of match_arms tags, read from the arms' saved games.

Every arm plays the same deals pair for pair (arms.py), so two arms compare
game by game: a pair's score difference cancels the deal's luck, which the
separate intervals of the two arms both carry. The games are the .slog files
TagPaths.arm_games_dir holds, so an arm still playing reports what it has
played so far.
"""

import dataclasses
import math
import sqlite3
from pathlib import Path

import numpy as np

from scribblez.dashboard import db
from scribblez.paths import TagPaths
from scribblez.sim_evidence.slog_meta import game_metas, read_slog_bytes

WORKLOAD = "match_arms"

# The arm is play_game's first --player (harness.play_round); its seat in a game
# is the seat whose recorded player index is this.
ARM_PLAYER = 0


@dataclasses.dataclass(frozen=True)
class Game:
    """One game from the arm's side: 1 a win, 0.5 a draw, 0 a loss, and the
    final spread."""

    result: float
    spread: int


@dataclasses.dataclass(frozen=True)
class Interval:
    mean: float
    half_width: float  # 95%


@dataclasses.dataclass(frozen=True)
class Summary:
    games: int
    wins: int
    draws: int
    losses: int
    score: Interval
    spread: Interval


@dataclasses.dataclass(frozen=True)
class Comparison:
    """An arm less the baseline, over the pairs both have played."""

    pairs: int
    score: Interval
    spread: Interval
    identical: int  # games whose spread both arms share: decisions that never diverged


# An arm's games by (seed, the arm's seat). A pair's two games share the seed.
ArmGames = dict[tuple[int, int], Game]


def arm_game(meta: np.void) -> tuple[tuple[int, int], Game]:
    """One .slog game's key and result for the arm."""
    players = (int(meta["seat0_player"]), int(meta["seat1_player"]))
    if ARM_PLAYER not in players:
        raise ValueError(f"game with seed {int(meta['seed'])} records no seat for the arm")
    seat = players.index(ARM_PLAYER)
    scores = (int(meta["final_score_p0"]), int(meta["final_score_p1"]))
    spread = scores[seat] - scores[1 - seat]
    return (int(meta["seed"]), seat), Game(result=(float(np.sign(spread)) + 1) / 2, spread=spread)


def read_arm_games(games_dir: Path) -> ArmGames:
    """Every game under an arm's games directory."""
    out: ArmGames = {}
    for path in sorted(games_dir.glob("*.slog")):
        for meta in game_metas(read_slog_bytes(path)):
            key, game = arm_game(meta)
            out[key] = game
    return out


def interval(values: np.ndarray) -> Interval:
    """The mean of `values` with a normal 95% interval; no width from one value."""
    if len(values) < 2:
        return Interval(float(values.mean()) if len(values) else math.nan, math.nan)
    return Interval(float(values.mean()), 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values)))


def pair_means(games: ArmGames, seeds: list[int], field: str) -> np.ndarray:
    """Each complete pair's mean of `field` over its two games, in `seeds` order."""
    return np.array(
        [(getattr(games[s, 0], field) + getattr(games[s, 1], field)) / 2 for s in seeds]
    )


def complete_pairs(games: ArmGames) -> list[int]:
    """The seeds of the pairs with both games played."""
    return sorted(s for s, seat in games if seat == 0 and (s, 1) in games)


def summarize(games: ArmGames) -> Summary:
    """W/D/L over every game, with intervals over complete pairs: the two
    seatings of a deal are correlated, so the pair is the independent unit."""
    results = [g.result for g in games.values()]
    seeds = complete_pairs(games)
    return Summary(
        games=len(results),
        wins=results.count(1.0),
        draws=results.count(0.5),
        losses=results.count(0.0),
        score=interval(pair_means(games, seeds, "result")),
        spread=interval(pair_means(games, seeds, "spread")),
    )


def compare(arm: ArmGames, baseline: ArmGames) -> Comparison:
    """`arm` less `baseline` pair by pair, over the pairs both completed."""
    seeds = sorted(set(complete_pairs(arm)) & set(complete_pairs(baseline)))
    keys = [(s, seat) for s in seeds for seat in (0, 1)]
    return Comparison(
        pairs=len(seeds),
        score=interval(pair_means(arm, seeds, "result") - pair_means(baseline, seeds, "result")),
        spread=interval(pair_means(arm, seeds, "spread") - pair_means(baseline, seeds, "spread")),
        identical=sum(arm[k].spread == baseline[k].spread for k in keys),
    )


@dataclasses.dataclass(frozen=True)
class Arm:
    tag: str
    name: str
    games: ArmGames
    elapsed_s: float | None  # None while the arm is still playing


def finished_elapsed(tag_paths: TagPaths) -> dict[str, float]:
    """Each finished arm's wall time, from the tag's match_arm rows."""
    if not tag_paths.dashboard_db.exists():
        return {}
    conn = sqlite3.connect(f"file:{tag_paths.dashboard_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return {r["arm"]: r["elapsed_s"] for r in db.read_all_match_arms(conn)}
    finally:
        conn.close()


def read_tag(tag: str, mount_root: Path) -> list[Arm]:
    """A tag's arms that have saved games, by name."""
    paths = TagPaths(tag, WORKLOAD, mount_root)
    elapsed = finished_elapsed(paths)
    games_root = paths.data_dir / "arm_games"
    if not games_root.is_dir():
        return []
    return [
        Arm(tag=tag, name=d.name, games=read_arm_games(d), elapsed_s=elapsed.get(d.name))
        for d in sorted(games_root.iterdir())
        if d.is_dir()
    ]


def all_tags(mount_root: Path) -> list[str]:
    root = mount_root / "tags" / WORKLOAD
    return sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []

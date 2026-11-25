"""A release's matches (docs/plans/subtree_bot_generations.md, "The matches"):
what leaf_bot_generation's measure role's worker
(leaf_bot_generation/measure.py) and the controller's tick
(leaf_bot_generation/measure_dispatch.py) agree on. The opponents, how a finished match travels,
where the controller keeps it, and the estimate a chart draws.

Every game is kept: a row per game (play_game's --results-file line: its
seed, seats and scores) and the games' .slog. The estimate is computed from
the rows, so another estimator, such as a correction for the tiles' luck, can
be run over the kept games without replaying them.

Importable without torch or the engine.
"""

import json
import shutil
import time
from pathlib import Path

from scribblez import stats
from scribblez.match_eval import harness
from scribblez.paths import MATCH_RESULTS_DIR, TagPaths

# The opponents every release is matched against, by the name its files and
# rows carry.
HASTYBOT = "hastybot"
RELEASE_0 = "release_0"
OPPONENTS = (HASTYBOT, RELEASE_0)
HASTYBOT_SPEC = "--type=hastybot-endgame"

# The data/ subdirectory a match's .slog is delivered into, beside
# MATCH_RESULTS_DIR for its rows.
MATCH_GAMES_DIR = "match_games"


def opponents_of(release: int) -> tuple[str, ...]:
    """The opponents release `release` is matched against. Release 0 against
    itself would measure nothing."""
    return OPPONENTS if release > 0 else (HASTYBOT,)


def match_stem(release: int, opponent: str) -> str:
    """The name a match's delivered files share."""
    return f"release_{release:06d}-{opponent}"


def result_rel(release: int, opponent: str) -> str:
    """Where a match's rows are delivered, under data/."""
    return f"{MATCH_RESULTS_DIR}/{match_stem(release, opponent)}.json"


def games_rel(release: int, opponent: str) -> str:
    """Where a match's .slog is delivered, under data/."""
    return f"{MATCH_GAMES_DIR}/{match_stem(release, opponent)}.slog"


def kept_dir(paths: TagPaths, release: int) -> Path:
    """Where the controller keeps release `release`'s matches:
    <opponent>.json, its rows, and <opponent>.slog, its games."""
    return paths.root / "matches" / f"release_{release:06d}"


def result_record(release: int, opponent: str, games: list[dict], elapsed_s: float) -> dict:
    """A finished match as it is delivered and kept: `games` are play_game's
    result lines, player 0 the release under test."""
    return {"release": release, "opponent": opponent, "elapsed_s": elapsed_s, "games": games}


def play(
    release: int,
    opponent: str,
    specs: tuple[str, str],
    games: int,
    seed: int,
    threads: int,
    work: Path,
    keep_games: bool,
) -> tuple[dict, Path | None]:
    """A match of `games` mirrored games from `seed` between `specs`, the
    release's --player spec and the opponent's: its record and, if
    `keep_games`, its .slog, both under `work`. A kept match fits one .slog,
    play_game's 1000 games."""
    results = work / "results.jsonl"
    slog_dir = work / "games" if keep_games else None
    if slog_dir is not None:
        shutil.rmtree(slog_dir, ignore_errors=True)
    t0 = time.monotonic()
    harness.play_round(
        *specs,
        num_pairs=games // 2,
        threads=threads,
        seed=seed,
        results_file=results,
        slog_dir=slog_dir,
    )
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    record = result_record(release, opponent, rows, time.monotonic() - t0)
    if slog_dir is None:
        return record, None
    (slog,) = slog_dir.glob("*.slog")
    return record, slog


def deliver(sink, record: dict, slog: Path | None, work: Path) -> int:
    """Hand a finished match to the tag behind `sink`: its games if they
    were kept, then the rows that announce them. Returns the bytes
    delivered."""
    number, opponent = record["release"], record["opponent"]
    rows = work / "result.json"
    rows.write_text(json.dumps(record) + "\n")
    nbytes = 0 if slog is None else sink.deliver(slog, games_rel(number, opponent))
    return nbytes + sink.deliver(rows, result_rel(number, opponent))


def describe(record: dict) -> str:
    """A finished match in a line."""
    s = estimate(record["games"])
    return (
        f"[release {record['release']}] vs {record['opponent']}: {s['wins']}/{s['draws']}/"
        f"{s['losses']} W/D/L, score={s['score']:.3f}+-{s['ci_half_width']:.3f} "
        f"in {record['elapsed_s']:.0f}s"
    )


def read_result(path: Path) -> dict:
    """A delivered or kept match. Raises ValueError if it is not one."""
    record = json.loads(path.read_text())
    missing = [f for f in ("release", "opponent", "elapsed_s", "games") if f not in record]
    if missing:
        raise ValueError(f"{path.name}: result is missing {', '.join(missing)}")
    return record


def estimate(games: list[dict]) -> dict:
    """The release's win rate over `games`, with draws at a half: its mean
    pair score, the confidence interval's half-width, and the counts behind
    them."""
    result = harness.round_result(games)
    counts = stats.pair_score_counts(result.pair_scores)
    score, ci = stats.score_confidence_interval(counts)
    return {
        "games": len(games),
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        "pair_counts": counts,
        "score": score,
        "ci_half_width": ci,
    }

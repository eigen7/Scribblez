"""A leaf export's view of the known positions (deep_bot/known_positions.py),
what deep_bot_leaf_eval's Positions tab shows: LeafBot's value of every legal
move at each position (engine agent/leaf_bot.h, its opponent leaves drawn from
the matches' leave inference model), the move's points, its rank under that
value and under static equity, and the pick the matches would make there: the
best of HastyBot's top `candidates`.

The match slot computes it after each match (workloads/deep_bot_leaf_eval.py)
and delivers it as data/leaf_positions/gen_NNNNNN.json, apart from the match
results the controller ingests.
"""

import json
from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.deep_bot.known_positions import Position

# The data/ subdirectory the analyses are delivered into.
POSITIONS_DIR = "leaf_positions"
# LeafBot's candidates in the analysis: more than any position's legal moves,
# so it values every move.
ALL_MOVES = 1_000_000
# LeafBot's seed in the analysis, which jitters its draws of the opponent's
# leaves. Fixed, so every generation's analysis, and the truth
# (deep_bot_leaf_eval/truth.py), value a position over the same leaves.
ANALYSIS_SEED = 1


def rel(generation: int) -> str:
    """Where generation `generation`'s analysis is delivered, under data/."""
    return f"{POSITIONS_DIR}/gen_{generation:06d}.json"


def generation_of(path: Path) -> int:
    """The generation an analysis file is of."""
    return int(path.stem.removeprefix("gen_"))


def player_spec(
    model: Path | str, leave: Path, opp_leaves: int, candidates: int = ALL_MOVES
) -> str:
    """LeafBot valuing static equity's best `candidates` moves (by default
    every legal move) with leaf `model`, an export or LeafBot's monte-carlo
    leaf."""
    return (
        f"--type=leafbot --leaf-model={model} --leave-inference={leave} "
        f"--candidates={candidates} --opp-leaves={opp_leaves} --seed={ANALYSIS_SEED}"
    )


def match_pick(record: dict, candidates: int) -> str:
    """The move the matches' LeafBot (its `candidates` best by static equity)
    plays at the position the pick tool's `record` (of LeafBot over every
    move) is of: the best valued of those candidates. A position with a single
    legal move has no values, and its record's pick is the matches' too."""
    top = [v for v in record.get("values", []) if 1 <= v["equity_rank"] <= candidates]
    if not top:
        return record["pick"]
    return max(top, key=lambda v: (v["value"], -v["equity_rank"]))["move"]


def entry(p: Position, record: dict, candidates: int) -> dict:
    """The analysis of one position: every valued move, best first, with its
    leaf rank, and the matches' pick with how it reads against the answer."""
    ranked = sorted(record.get("values", []), key=lambda v: (-v["value"], v["equity_rank"]))
    pick = match_pick(record, candidates)
    return {
        "values": [{**v, "leaf_rank": i + 1} for i, v in enumerate(ranked)],
        "pick": pick,
        "reading": p.read(pick),
    }


def analyze(
    positions: list[Position],
    model: Path,
    leave: Path,
    opp_leaves: int,
    candidates: int,
    work: Path,
    threads: int,
) -> dict:
    """`model`'s analysis of every position, keyed by position id."""
    records = known_positions.ask(positions, player_spec(model, leave, opp_leaves), work, threads)
    return {
        "candidates": candidates,
        "positions": {
            p.id: entry(p, r, candidates) for p, r in zip(positions, records, strict=True)
        },
    }


def write(doc: dict, path: Path):
    """Write an analysis atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)

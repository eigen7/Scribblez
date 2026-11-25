"""A leaf export's view of the known positions (deep_bot/known_positions.py),
what deep_bot_leaf_eval's Positions tab shows: DeepBot's value of every legal
move at each position (engine agent/deep_bot.h, its opponent leaves drawn from
the matches' leave inference model), the move's points, its rank under that
value and under static equity, and the pick the matches would make there: the
best of HastyBot's top `candidates`. It values each move over more opponent
leaves than a match's DeepBot does, so that pick is what the matches' DeepBot
would play with that many draws, which a match's might not.

The match slot computes it after each match (workloads/deep_bot_leaf_eval.py)
and delivers it as data/leaf_positions/gen_NNNNNN.json, apart from the match
results the controller ingests. A position the manifest gains later is added
to the analyses already delivered by the positions backfill
(workloads/positions_backfill.py), which rewrites them.
"""

import json
from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.deep_bot.known_positions import Position
from scribblez.paths import TagPaths

# The data/ subdirectory the analyses are delivered into.
POSITIONS_DIR = "leaf_positions"
# DeepBot's candidates in the analysis: more than any position's legal moves,
# so it values every move.
ALL_MOVES = 1_000_000
# DeepBot's seed in the analysis, which jitters its draws of the opponent's
# leaves. Fixed, so every generation's analysis, and the hasty rollouts
# (deep_bot_leaf_eval/hasty_rollouts.py), value a position over the same
# leaves.
ANALYSIS_SEED = 1


def rel(generation: int) -> str:
    """Where generation `generation`'s analysis is delivered, under data/."""
    return f"{POSITIONS_DIR}/gen_{generation:06d}.json"


def generation_of(path: Path) -> int:
    """The generation an analysis file is of."""
    return int(path.stem.removeprefix("gen_"))


def analyzed(paths: TagPaths) -> list[int]:
    """The generations with a delivered analysis, ascending."""
    folder = paths.data_dir / POSITIONS_DIR
    return sorted(generation_of(f) for f in folder.glob("gen_*.json"))


def player_spec(
    model: Path | str, leave: Path, opp_leaves: int, candidates: int = ALL_MOVES
) -> str:
    """DeepBot valuing static equity's best `candidates` moves (by default
    every legal move) with leaf `model`, an export or DeepBot's monte-carlo
    leaf."""
    return (
        f"--type=deepbot --leaf-model={model} --leave-inference={leave} "
        f"--candidates={candidates} --opp-leaves={opp_leaves} --seed={ANALYSIS_SEED}"
    )


def match_pick(record: dict, candidates: int) -> str:
    """The move the matches' DeepBot (its `candidates` best by static equity)
    plays at the position the pick tool's `record` (of DeepBot over every
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
    """`model`'s analysis of every position over `opp_leaves` opponent
    leaves, keyed by position id."""
    records = known_positions.ask(positions, player_spec(model, leave, opp_leaves), work, threads)
    return {
        "candidates": candidates,
        "opp_leaves": opp_leaves,
        "positions": {
            p.id: entry(p, r, candidates) for p, r in zip(positions, records, strict=True)
        },
    }


def read(paths: TagPaths, generation: int) -> dict:
    """Generation `generation`'s delivered analysis."""
    return json.loads((paths.data_dir / rel(generation)).read_text())


def missing(doc: dict, positions: list[Position]) -> list[Position]:
    """The positions an analysis lacks: those the manifest gained since."""
    return [p for p in positions if p.id not in doc["positions"]]


def extend(
    doc: dict, positions: list[Position], model: Path, leave: Path, work: Path, threads: int
) -> dict:
    """Analysis `doc` with `positions` analyzed too, under the same opponent
    leaves and candidates as the rest of it, by its `model` and `leave`."""
    added = analyze(positions, model, leave, doc["opp_leaves"], doc["candidates"], work, threads)
    return {**doc, "positions": {**doc["positions"], **added["positions"]}}


def write(doc: dict, path: Path):
    """Write an analysis atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)

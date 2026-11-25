"""The hasty rollouts of a known position: each legal move's win value when
the game is played out from it by hastybot-endgame. DeepBot computes them
over its monte-carlo leaf (engine sim/monte_carlo_leaf_values.h), playing
each move out to the game's end once against each of many opponent leaves
drawn from the leave inference model's belief.

They are what a deep_bot_leaf_eval leaf would value the moves at if it
learned its targets, HastyBot's self-play, perfectly. They are not the moves'
true values: another continuation would value them otherwise, and a leaf
trained on other games (leaf_bot_generation's, DeepBot's own) targets that
continuation instead. They value a move over the whole belief, where the
network's analysis (positions.py) values it over a few draws; their leaves
are drawn as the analysis draws them (the same model and ANALYSIS_SEED),
only more.

The leaf_eval_hasty_rollouts workload computes them one position at a time
into data/hasty_rollouts/<gcg stem>.json.
py/scripts/deep_bot_leaf_eval/copy_hasty_rollouts.py gathers a finished
tag's files into the repo's ROLLOUTS_FILE, beside the positions, where the
Positions tab reads them. A position's rollouts hold while its GCG and the
leave inference model are those they were computed under, so they record
both: the GCG by its hash, the model in the run's settings.
"""

import hashlib
import json
import time
from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.deep_bot.known_positions import Position
from scribblez.deep_bot_leaf_eval import positions as leaf_positions

# The data/ subdirectory a tag's per-position files are delivered into.
ROLLOUTS_DIR = "hasty_rollouts"
# The repo's copy, which the Positions tab reads.
ROLLOUTS_FILE = known_positions.DIRECTORY / "hasty_rollouts.json"
# DeepBot's leaf that plays out instead of predicting (DeepBot::kMonteCarloLeaf).
MONTE_CARLO_LEAF = "monte-carlo"


def rel(position: Position) -> str:
    """Where `position`'s rollouts are delivered, under data/."""
    return f"{ROLLOUTS_DIR}/{position.gcg.stem}.json"


def gcg_hash(position: Position) -> str:
    """The hash of the GCG the rollouts were computed from."""
    return hashlib.sha256(position.gcg.read_bytes()).hexdigest()[:16]


def run_settings(leave_tag: str, leave_generation: int, playouts: int, candidates: int) -> dict:
    """What the rollouts were computed under, beside the GCG: the leave
    inference model the opponent's leaves are drawn from, the playouts per
    move (one per leave drawn), the seed that jitters the draws, and how many
    of static equity's best moves get them (0: every legal move)."""
    return {
        "leave_inference": {"tag": leave_tag, "generation": leave_generation},
        "playouts": playouts,
        "seed": leaf_positions.ANALYSIS_SEED,
        "candidates": candidates,
    }


def compute(position: Position, leave: Path, run: dict, work: Path, threads: int) -> dict:
    """`position`'s rollouts under `run` (run_settings), with `leave` the
    run's leave inference model: the mover's win value (a win 1, a draw 1/2)
    after each move the run covers."""
    spec = leaf_positions.player_spec(
        MONTE_CARLO_LEAF, leave, run["playouts"], run["candidates"] or leaf_positions.ALL_MOVES
    )
    began = time.monotonic()
    (record,) = known_positions.ask([position], spec, work, threads)
    return {
        "id": position.id,
        "gcg_hash": gcg_hash(position),
        "run": run,
        "seconds": round(time.monotonic() - began, 1),
        "values": {v["move"]: round(v["value"], 5) for v in record.get("values", [])},
    }


def collect(data_dir: Path, positions: list[Position]) -> tuple[dict, list[str]]:
    """The repo file's contents from a tag's data dir: each of `positions`
    whose rollouts the tag holds for its current GCG, and the ids skipped
    (none, or computed from another GCG). Raises ValueError if the tag's
    files were computed under different settings."""
    doc: dict = {"run": None, "positions": {}}
    skipped = []
    for p in positions:
        path = data_dir / rel(p)
        if not path.is_file():
            skipped.append(p.id)
            continue
        rollouts = json.loads(path.read_text())
        if rollouts["gcg_hash"] != gcg_hash(p):
            skipped.append(p.id)
            continue
        if doc["run"] not in (None, rollouts["run"]):
            raise ValueError(f"{path} was computed under {rollouts['run']}, not {doc['run']}")
        doc["run"] = rollouts["run"]
        doc["positions"][p.id] = {"gcg_hash": rollouts["gcg_hash"], "values": rollouts["values"]}
    return doc, skipped


def load() -> dict | None:
    """The repo's rollouts file, or None before any was copied in."""
    return json.loads(ROLLOUTS_FILE.read_text()) if ROLLOUTS_FILE.is_file() else None


def of_position(doc: dict | None, position: Position) -> dict[str, float] | None:
    """`position`'s rollout values by move out of the repo's file, or None
    when it holds none for the position's current GCG."""
    entry = (doc or {}).get("positions", {}).get(position.id)
    if entry is None or entry["gcg_hash"] != gcg_hash(position):
        return None
    return entry["values"]


def same_belief(run: dict, leaf_params: dict) -> bool:
    """Whether the rollouts' opponent leaves are drawn from the belief a
    deep_bot_leaf_eval tag's analysis draws from (its frozen params
    `leaf_params`): the same leave inference model."""
    return run["leave_inference"] == {
        "tag": leaf_params.get("match_leave_inference_tag"),
        "generation": leaf_params.get("match_leave_inference_generation"),
    }

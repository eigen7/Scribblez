"""What a perfect leaf network would value each legal move of a known position
at: LeafBot over its monte-carlo leaf (engine sim/monte_carlo_leaf_values.h),
which plays each (move, opponent leave) out to the game's end with
hastybot-endgame, the policy the network's targets come from, over the same
opponent leaves as the network's analysis (positions.py, ANALYSIS_SEED).

The leaf_eval_truth workload computes it one position at a time into
data/truth/<gcg stem>.json. py/scripts/deep_bot_leaf_eval/copy_truth.py
gathers a finished tag's files into the repo's TRUTH_FILE, beside the
positions, where the Positions tab reads them. A position's truth holds while
its GCG and the leave inference model are those it was computed under, so it
records both: the GCG by its hash, the model in the run's settings.
"""

import hashlib
import json
import time
from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.deep_bot.known_positions import Position
from scribblez.deep_bot_leaf_eval import positions as leaf_positions

# The data/ subdirectory a tag's per-position files are delivered into.
TRUTH_DIR = "truth"
# The repo's copy, which the Positions tab reads.
TRUTH_FILE = known_positions.DIRECTORY / "truth.json"
# LeafBot's leaf that plays out instead of predicting (LeafBot::kMonteCarloLeaf).
MONTE_CARLO_LEAF = "monte-carlo"


def rel(position: Position) -> str:
    """Where `position`'s truth is delivered, under data/."""
    return f"{TRUTH_DIR}/{position.gcg.stem}.json"


def gcg_hash(position: Position) -> str:
    """The hash of the GCG a truth was computed from."""
    return hashlib.sha256(position.gcg.read_bytes()).hexdigest()[:16]


def run_settings(
    leave_tag: str, leave_generation: int, opp_leaves: int, games: int, candidates: int
) -> dict:
    """What a truth was computed under, beside the GCG: the leave inference
    model the opponent's leaves are drawn from, how many, the seed that jitters
    them, the playouts per (move, leave), and how many of static equity's best
    moves get one (0: every legal move)."""
    return {
        "leave_inference": {"tag": leave_tag, "generation": leave_generation},
        "opp_leaves": opp_leaves,
        "seed": leaf_positions.ANALYSIS_SEED,
        "games": games,
        "candidates": candidates,
    }


def compute(position: Position, leave: Path, run: dict, work: Path, threads: int) -> dict:
    """`position`'s truth under `run` (run_settings), with `leave` the run's
    leave inference model: every legal move's value, the mover's win value
    (a win 1, a draw 1/2) after it, for each move the run covers."""
    spec = leaf_positions.player_spec(
        MONTE_CARLO_LEAF, leave, run["opp_leaves"], run["candidates"] or leaf_positions.ALL_MOVES
    )
    began = time.monotonic()
    (record,) = known_positions.ask(
        [position], f"{spec} --monte-carlo-games={run['games']}", work, threads
    )
    return {
        "id": position.id,
        "gcg_hash": gcg_hash(position),
        "run": run,
        "seconds": round(time.monotonic() - began, 1),
        "values": {v["move"]: round(v["value"], 5) for v in record.get("values", [])},
    }


def collect(data_dir: Path, positions: list[Position]) -> tuple[dict, list[str]]:
    """The repo file's contents from a tag's data dir: each of `positions`
    whose truth the tag holds for its current GCG, and the ids skipped (no
    truth, or one computed from another GCG). Raises ValueError if the tag's
    files were computed under different settings."""
    doc: dict = {"run": None, "positions": {}}
    skipped = []
    for p in positions:
        path = data_dir / rel(p)
        if not path.is_file():
            skipped.append(p.id)
            continue
        truth = json.loads(path.read_text())
        if truth["gcg_hash"] != gcg_hash(p):
            skipped.append(p.id)
            continue
        if doc["run"] not in (None, truth["run"]):
            raise ValueError(f"{path} was computed under {truth['run']}, not {doc['run']}")
        doc["run"] = truth["run"]
        doc["positions"][p.id] = {"gcg_hash": truth["gcg_hash"], "values": truth["values"]}
    return doc, skipped


def load() -> dict | None:
    """The repo's truth file, or None before any truth was copied in."""
    return json.loads(TRUTH_FILE.read_text()) if TRUTH_FILE.is_file() else None


def of_position(doc: dict | None, position: Position) -> dict[str, float] | None:
    """`position`'s truth values by move out of the repo's file, or None when
    it holds none for the position's current GCG."""
    entry = (doc or {}).get("positions", {}).get(position.id)
    if entry is None or entry["gcg_hash"] != gcg_hash(position):
        return None
    return entry["values"]


def same_leaves(run: dict, leaf_params: dict) -> bool:
    """Whether a truth's opponent leaves are those a deep_bot_leaf_eval tag's
    analysis draws (its frozen params `leaf_params`): the same leave inference
    model, count and seed."""
    return (
        run["leave_inference"]
        == {
            "tag": leaf_params.get("match_leave_inference_tag"),
            "generation": leaf_params.get("match_leave_inference_generation"),
        }
        and run["opp_leaves"] == leaf_params.get("match_opp_leaves")
        and run["seed"] == leaf_positions.ANALYSIS_SEED
    )

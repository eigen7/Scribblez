"""The opponent panel of the leaf Positions tab (dashboard/
deep_bot_leaf_eval_api.py): at each known position, the mover's belief about
the opponent's rack (LeaveBeliefs.chart: each tile's chance of being on it
and the value of the leave they kept, beside the same under no information)
under the leave inference model a generation's analysis (positions.py) drew
the opponent's leaves from. A deep_bot_leaf_eval tag pins one model for all
its generations; a leaf_bot_generation release has its own.

Made on the controller, on the CPU, the first time the tab asks for a
generation (about a second for every position), and kept as
data/opp_beliefs/gen_NNNNNN.json; a position the manifest gains later is
added the next time the tab asks for it.
"""

import json
from pathlib import Path

import numpy as np

from scribblez.deep_bot import known_positions
from scribblez.deep_bot.position_view import board_view
from scribblez.deep_bot.release_positions import KnownPositionsReplay, belief_at
from scribblez.deep_bot_leaf_eval import positions as leaf_positions
from scribblez.leave_inference.beliefs import LeaveBeliefs
from scribblez.paths import TagPaths

# The data/ subdirectory the charts are kept in.
BELIEFS_DIR = "opp_beliefs"


def rel(generation: int) -> str:
    """Where generation `generation`'s charts are kept, under data/."""
    return f"{BELIEFS_DIR}/gen_{generation:06d}.json"


def chart_positions(replay: KnownPositionsReplay, leave: Path, indices: list[int]) -> dict:
    """The chart at each known position in `indices` (the replay's order)
    under leave inference model `leave`, by position id. Each one's sample
    racks are seeded by its index, as deep_bot's are."""
    beliefs = LeaveBeliefs(leave)
    out = {}
    for i in indices:
        b = belief_at(beliefs, replay, i, board_view(replay.positions[i]))
        out[replay.positions[i].id] = beliefs.chart(b, np.random.default_rng(i))
    return out


def charts(paths: TagPaths, generation: int, leave: Path) -> dict[str, dict]:
    """Every known position's chart under generation `generation`'s leave
    inference model `leave`, by position id: its file's, with the positions
    it lacks charted and written first."""
    path = paths.data_dir / rel(generation)
    held = json.loads(path.read_text()) if path.is_file() else {}
    if all(p.id in held for p in known_positions.load()):
        return held
    replay = KnownPositionsReplay()
    lacking = [i for i, p in enumerate(replay.positions) if p.id not in held]
    doc = {**held, **chart_positions(replay, leave, lacking)}
    leaf_positions.write(doc, path)
    return doc

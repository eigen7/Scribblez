"""The opponent panel of the leaf Positions tab (dashboard/
deep_bot_leaf_eval_api.py): at each known position, the mover's belief about
the opponent's rack (LeaveBeliefs.chart: each tile's chance of being on it
and the value of the leave they kept, beside the same under no information)
under the tag's fixed leave inference model, the one its games draw the
opponent's leaves from: a leaf_bot_generation tag's release 0, which its
self-play plays, or a deep_bot_leaf_eval tag's pinned matches' model. So it
is one chart a position for the tag's whole life, whatever generation the
tab shows.

Made on the controller, on the CPU, the first time the tab asks for one
(about a second for every position), and kept in the tag's dashboard.db
(opp_belief); a position the manifest gains, or whose GCG changes, is charted
on the next request.
"""

from pathlib import Path

import numpy as np

from scribblez.dashboard import db
from scribblez.deep_bot import known_positions
from scribblez.deep_bot.known_position_beliefs import KnownPositionsReplay, belief_at
from scribblez.deep_bot.position_view import board_view
from scribblez.deep_bot_leaf_eval.hasty_rollouts import gcg_hash
from scribblez.leave_inference.beliefs import LeaveBeliefs
from scribblez.paths import TagPaths


def chart_positions(replay: KnownPositionsReplay, leave: Path, indices: list[int]) -> dict:
    """The (gcg_hash, chart) at each known position in `indices` (the
    replay's order) under leave inference model `leave`, by position id. Each
    one's sample racks are seeded by its index, as deep_bot's are."""
    beliefs = LeaveBeliefs(leave)
    out = {}
    for i in indices:
        p = replay.positions[i]
        b = belief_at(beliefs, replay, i, board_view(p))
        out[p.id] = (gcg_hash(p), beliefs.chart(b, np.random.default_rng(i)))
    return out


def lacking(held: dict[str, tuple[str, dict]], positions: list[known_positions.Position]):
    """The indices of the positions `held` has no chart of as their GCGs now are."""
    return [i for i, p in enumerate(positions) if held.get(p.id, ("",))[0] != gcg_hash(p)]


def charts(paths: TagPaths, leave: Path) -> dict[str, dict]:
    """Every known position's chart under the tag's leave inference model
    `leave`, by position id: the database's, with those it lacks charted and
    written first."""
    conn = db.connect(paths.dashboard_db)
    try:
        held = db.read_opp_beliefs(conn)
        if lacking(held, known_positions.load()):
            replay = KnownPositionsReplay()
            added = chart_positions(replay, leave, lacking(held, replay.positions))
            db.write_opp_beliefs(conn, added)
            held |= added
        return {p: chart for p, (_, chart) in held.items()}
    finally:
        conn.close()

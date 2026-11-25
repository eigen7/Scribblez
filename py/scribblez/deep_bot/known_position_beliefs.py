"""The mover's belief about the opponent's rack at a known position
(deep_bot/known_positions.py), under a leave inference model
(LeaveBeliefs). The known positions replay from one .slog, game i the i-th
position, with its .forgone sidecar (KnownPositionsReplay).
"""

import tempfile
from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.leave_inference import forgone
from scribblez.leave_inference.beliefs import LeaveBeliefs, RackBelief, unseen_pool


class KnownPositionsReplay:
    """The known positions' games as one .slog with its .forgone sidecar,
    under a temporary directory that lives as long as this object."""

    def __init__(self, threads: int = 1):
        self.positions = known_positions.load()
        self._dir = tempfile.TemporaryDirectory(prefix="known_positions_")
        work = Path(self._dir.name)
        self.slog = known_positions.write_slog(self.positions, work)
        forgone.write_sidecars(work, threads)


def belief_at(
    beliefs: LeaveBeliefs, replay: KnownPositionsReplay, i: int, view: dict
) -> RackBelief:
    """The mover's belief about the opponent's rack at known position `i`,
    whose board_view is `view`."""
    (b,) = beliefs.beliefs(
        replay.slog,
        [(i, view["turn"] - 1)],
        [unseen_pool(view["board"], view["rack"])],
        [view["opp_rack_count"]],
    )
    return b

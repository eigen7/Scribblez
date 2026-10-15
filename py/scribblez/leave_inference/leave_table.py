"""The leave table the leave inference model scores and samples over: every
leave drawable from a full bag, for each length 1..6, with its static leave
value.

The engine builds it in memory (engine/include/belief/leave_table.h) and this
class copies its rows out through the FFI, so the trainer and the engine's
sampler see the same rows in the same order, and nothing here re-derives
them.
"""

from dataclasses import dataclass

import numpy as np

from scribblez.ffi import LeaveTableHandle

MAX_LEAVE_LENGTH = LeaveTableHandle.MAX_LENGTH


@dataclass
class LeaveRows:
    """One length's leaves, in the engine's order."""

    counts: np.ndarray  # (R, 27) uint8, indexed by tile code; the blank is 26
    value: np.ndarray  # (R,) float32, the static leave value


class LeaveTable:
    """The engine's leave table for the session's lexicon, its rows copied
    into numpy arrays."""

    def __init__(self):
        handle = LeaveTableHandle()
        try:
            self.klv_hash: str = handle.klv_hash
            self.rows: dict[int, LeaveRows] = {
                k: LeaveRows(*handle.rows(k)) for k in range(1, MAX_LEAVE_LENGTH + 1)
            }
        finally:
            handle.close()

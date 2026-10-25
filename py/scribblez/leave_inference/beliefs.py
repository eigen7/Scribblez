"""The leave inference model's belief about the opponent's rack at .slog
positions, on the CPU, for viewers: each position's model row decoded from
the .slog and its .forgone sidecar (ffi.decode_rows, the trainer's own rows),
the exported model run with onnxruntime, and the engine's sampler
(ffi.LeaveValueArrays.sampler_query, belief/leave_sampler.h) over the leave
table.

The model predicts only the leave the opponent kept at their last move; their
rack is that leave plus a refill drawn uniformly from the rest of the
observer's unseen pool (docs/plans/deep_bot.md, "The leave inference model").
Where the row carries no belief (turn 0, a pass, an empty leave, or an empty
bag) the leave is empty and the whole rack is a uniform draw. Zero parameters
are the model's own no-information prior, so the uniform reading goes through
the same sampler.
"""

from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import onnxruntime as ort

from scribblez import ffi

TILE_KINDS = 27  # A..Z, then the blank
NUM_PARAMS = TILE_KINDS + 2  # theta per tile, then the tilt's a and b
# The standard English distribution by tile index (A..Z, blank).
DISTRIBUTION = np.array(
    [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2],
    dtype=np.int64,
)
OUTPUT = "leave_params"
CAUSE_VALID = ffi.format_layout()["constants"]["leave_inference"]["cause"]["valid"]


@dataclass(frozen=True)
class RackBelief:
    """What the observer believes about the opponent's rack at one position."""

    params: np.ndarray  # (29,) the model's leave parameters; zeros where not modelled
    modelled: bool  # whether the model has a belief here, rather than a uniform rack
    leave_length: int  # tiles the opponent kept (0 where not modelled)
    leave_pool: np.ndarray  # (27,) what the leave is drawn from
    unseen: np.ndarray  # (27,) the observer's unseen pool: the bag plus the opponent's rack
    rack_size: int  # the opponent's

    @property
    def refill(self) -> int:
        return self.rack_size - self.leave_length


def unseen_pool(board: list[list[str | None]], rack: str) -> np.ndarray:
    """(27,) the tiles neither on `board` (web encoding: lowercase is a blank)
    nor on the observer's `rack` ('?' a blank)."""
    out = DISTRIBUTION.copy()
    for tile in [c for row in board for c in row if c is not None] + list(rack):
        out[26 if tile == "?" or tile.islower() else ord(tile) - ord("A")] -= 1
    return out


def _target(row: np.ndarray, name: str) -> np.ndarray:
    offset = ffi.task_input_floats("leave_inference")
    for s in ffi.task_target_shapes("leave_inference"):
        size = int(np.prod(s.dims))
        if s.name == name:
            return row[offset : offset + size]
        offset += size
    raise KeyError(name)


class LeaveBeliefs:
    """The exported model at `onnx_path` and the leave table its beliefs are
    over."""

    def __init__(self, onnx_path: Path):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.table = ffi.LeaveValueArrays()
        self._letters: dict[int, np.ndarray] = {}
        self._presence: dict[int, np.ndarray] = {}

    def rows(self, slog: Path, positions: list[tuple[int, int]]) -> np.ndarray:
        """The model rows of (game, turn) `positions` of `slog`, whose .forgone
        sidecar must sit beside it."""
        games, turns = (np.array(c, dtype=np.int64) for c in zip(*positions, strict=True))
        return ffi.decode_rows(slog, games, turns, post_move=False, task="leave_inference")

    def params(self, rows: np.ndarray) -> np.ndarray:
        """(N, 29) the model's parameters for decoded `rows`."""
        shapes = ffi.task_input_shapes("leave_inference")
        spatial_size = int(np.prod(shapes[0].dims))
        feeds = {
            shapes[0].name: rows[:, :spatial_size].reshape(-1, *shapes[0].dims),
            shapes[1].name: rows[:, spatial_size : spatial_size + int(np.prod(shapes[1].dims))],
        }
        return self.session.run([OUTPUT], feeds)[0]

    def beliefs(
        self, slog: Path, positions: list[tuple[int, int]], unseen: list[np.ndarray], rack_sizes
    ) -> list[RackBelief]:
        """The belief at each of `positions` of `slog`, given each one's
        observer `unseen` pool and the opponent's rack size."""
        rows = self.rows(slog, positions)
        params = self.params(rows)
        out = []
        for row, p, u, size in zip(rows, params, unseen, rack_sizes, strict=True):
            modelled = (
                int(_target(row, "opp_leave_cause")[0]) == CAUSE_VALID
                and int(_target(row, "bag_size")[0]) > 0
            )
            out.append(
                RackBelief(
                    params=p if modelled else np.zeros(NUM_PARAMS, dtype=np.float32),
                    modelled=modelled,
                    leave_length=int(_target(row, "opp_leave_length")[0]) if modelled else 0,
                    leave_pool=_target(row, "leave_pool").astype(np.int64),
                    unseen=u,
                    rack_size=size,
                )
            )
        return out

    def _leave_letters(self, length: int) -> np.ndarray:
        """(length, rows) the tile index in each slot of each length-`length`
        leave."""
        if length not in self._letters:
            codes, _ = self.table.of_length(length)
            self._letters[length] = codes // self.table.copies_per_tile
        return self._letters[length]

    def _leave_presence_matrix(self, length: int) -> np.ndarray:
        """(rows, 27) whether each length-`length` leave holds the tile."""
        if length not in self._presence:
            letters = self._leave_letters(length)
            presence = np.zeros((letters.shape[1], TILE_KINDS))
            for slot in letters:
                presence[np.arange(len(slot)), slot] = 1.0
            self._presence[length] = presence
        return self._presence[length]

    def leave_probs(self, params: np.ndarray, b: RackBelief) -> np.ndarray:
        """Every length-b.leave_length leave's probability under `params`."""
        n = self._leave_presence_matrix(b.leave_length).shape[0]
        _, log_probs, _ = self.table.sampler_query(
            params, b.leave_length, b.leave_pool, rows=np.arange(n)
        )
        return np.exp(log_probs)

    def rack_presence(self, b: RackBelief, uniform: bool = False) -> np.ndarray:
        """(27,) P(the opponent's rack holds the tile): the leave's, or else the
        refill's, k tiles drawn from the unseen pool less the leave. Under
        `uniform` the leave is the no-information prior's."""
        leave = np.zeros(TILE_KINDS)
        if b.leave_length > 0:
            params = np.zeros(NUM_PARAMS, dtype=np.float32) if uniform else b.params
            probs = self.leave_probs(params, b)
            leave = probs @ self._leave_presence_matrix(b.leave_length)
        rest = int(b.unseen.sum()) - b.leave_length
        k = b.refill
        # A leave lacking the tile leaves every unseen copy of it in the pool.
        miss = np.array([comb(rest - int(u), k) / comb(rest, k) for u in b.unseen])
        return leave + (1 - leave) * (1 - miss)

    def sample_racks(self, b: RackBelief, n: int, rng: np.random.Generator) -> list[dict]:
        """Up to `n` distinct racks, each a leave from the belief and a refill
        drawn without replacement from the unseen pool less it, as
        {"leave", "refill"} strings ('?' a blank), sorted. Fewer when the
        belief holds fewer distinct racks."""
        seen, out = set(), []
        for leave in self._sample_leaves(b, 50 * n, rng):
            if len(out) == n:
                break
            pool = b.unseen.copy()
            for t in leave:
                pool[t] -= 1
            refill = rng.choice(np.repeat(np.arange(TILE_KINDS), pool), b.refill, replace=False)
            rack = (_spell(leave), _spell(refill))
            if rack not in seen:
                seen.add(rack)
                out.append({"leave": rack[0], "refill": rack[1]})
        return out

    def _sample_leaves(self, b: RackBelief, m: int, rng: np.random.Generator) -> list[list[int]]:
        """`m` leaves drawn from the belief, as tile indices."""
        if b.leave_length == 0:
            return [[] for _ in range(m)]
        _, _, rows = self.table.sampler_query(
            b.params, b.leave_length, b.leave_pool, quantiles=rng.random(m)
        )
        return [list(col) for col in self._leave_letters(b.leave_length)[:, rows].T]


def _spell(tiles) -> str:
    """Tile indices as letters in tile order, the blank ('?') last."""
    return "".join("?" if t == 26 else chr(ord("A") + int(t)) for t in sorted(tiles))

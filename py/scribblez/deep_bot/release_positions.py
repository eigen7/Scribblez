"""A release's models at the known positions (deep_bot/known_positions.py),
computed once as the trainer publishes the release and kept beside its
exports as positions.json, which deep_bot's Positions tab
(deep_bot/position_view.py) reads. Two of the release's models, run on the CPU
under onnxruntime from the exports just written:

- the leave inference model's belief about the opponent's rack
  (LeaveBeliefs.chart);
- the policy over every candidate PolicyBot chooses among there
  (ffi.policy_candidates): each one's probability and its value head's win
  probability, the opponent-rack input summarized from the same belief, as
  PolicyBot reads it.

The leaf network has no output of its own at a position: it values the
rollout leaves two plies on.

The known positions replay from one .slog, game i the i-th position, with its
.forgone sidecar (KnownPositionsReplay), built once per process.
"""

import functools
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from scribblez import ffi
from scribblez.deep_bot import LEAVE, POLICY, known_positions, release
from scribblez.deep_bot.position_view import board_view
from scribblez.leave_inference import forgone
from scribblez.leave_inference.beliefs import TILE_KINDS, LeaveBeliefs, RackBelief, unseen_pool
from scribblez.paths import TagPaths
from scribblez.policy_set.onnx_export import BLANK_PRUNE_KEY, LEAVE_TAG_KEY, OUTPUTS
from scribblez.sim_evidence.sobs import move_notation

# onnxruntime's input type names, as numpy dtypes.
_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
}


class KnownPositionsReplay:
    """The known positions' games as one .slog with its .forgone sidecar,
    under a temporary directory that lives as long as this object."""

    def __init__(self, threads: int = 1):
        self.positions = known_positions.load()
        self._dir = tempfile.TemporaryDirectory(prefix="deep_bot_positions_")
        work = Path(self._dir.name)
        self.slog = known_positions.write_slog(self.positions, work)
        forgone.write_sidecars(work, threads)


@functools.cache
def _replay() -> KnownPositionsReplay:
    return KnownPositionsReplay()


@dataclass
class ReleaseModels:
    """A release's leave inference model and policy, on the CPU."""

    beliefs: LeaveBeliefs
    policy: ort.InferenceSession
    blank_prune_points: int
    # Whether the policy trained on the leave inference model's beliefs, not
    # uniform ones (policy_set/onnx_export.py's metadata).
    inferred: bool

    @classmethod
    def load(cls, paths: TagPaths, number: int) -> "ReleaseModels":
        policy = ort.InferenceSession(
            str(release.export_path(paths, number, POLICY)), providers=["CPUExecutionProvider"]
        )
        meta = policy.get_modelmeta().custom_metadata_map
        return cls(
            beliefs=LeaveBeliefs(release.export_path(paths, number, LEAVE)),
            policy=policy,
            blank_prune_points=int(meta[BLANK_PRUNE_KEY]),
            inferred=bool(meta.get(LEAVE_TAG_KEY)),
        )

    def opp_rack(self, b: RackBelief) -> np.void:
        """The policy's opponent-rack input record at belief `b`: the belief's
        where it is modelled and the policy reads beliefs, else a uniform
        rack, as PolicyBot summarizes it."""
        if self.inferred and b.modelled:
            return self.beliefs.table.opp_rack_summary(
                b.unseen, b.params, b.leave_length, b.leave_pool
            )
        return self.beliefs.table.opp_rack_summary(
            b.unseen, None, 0, np.zeros(TILE_KINDS, dtype=np.int32)
        )

    def policy_feed(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The policy graph's inputs from ffi.policy_set_positions' arrays for
        one position."""
        spatial, scalar = (s.dims for s in ffi.task_input_shapes()[:2])
        rows = inputs["rows"]
        size = int(np.prod(spatial))
        arrays = {
            "input_spatial": rows[:, :size].reshape(1, *spatial),
            "input_scalar": rows[:, size : size + scalar[0]],
            "opp_rack": inputs["opp_rack"],
            "move_letters": inputs["letters"],
            "move_blanks": inputs["blanks"],
            "move_squares": inputs["squares"],
            "move_tile_mask": inputs["tile_mask"],
            "move_scalars": inputs["scalars"],
            "move_cross_cells": inputs["cells"],
            "move_cross_letters": inputs["letters_cross"],
            "move_equity": inputs["equity"],
        }
        return {
            i.name: np.ascontiguousarray(
                arrays[i.name].reshape(len(arrays[i.name]), -1), dtype=_ORT_DTYPES[i.type]
            ).reshape([d if isinstance(d, int) else -1 for d in i.shape])
            for i in self.policy.get_inputs()
        }


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


def policy_candidates(
    models: ReleaseModels, replay: KnownPositionsReplay, i: int, view: dict, b: RackBelief
) -> list[dict]:
    """Every candidate at known position `i` with the policy's outputs,
    likeliest first: its probability, and its value head's win probability
    and the spread of one rollout's value about it, in percent."""
    turn = view["turn"] - 1
    moves, ranks = ffi.policy_candidates(replay.slog, i, turn, models.blank_prune_points)
    inputs = ffi.policy_set_positions(
        replay.slog,
        np.array([i]),
        np.array([turn]),
        np.array([len(moves)]),
        moves,
        np.array([models.opp_rack(b)]),
    )
    logit, value = models.policy.run(list(OUTPUTS), models.policy_feed(inputs))
    z = logit[:, 0] - logit[:, 0].max()
    prob = np.exp(z) / np.exp(z).sum()
    out = [
        {
            "move": move_notation(m, view["board"]),
            "prob": float(p),
            "win_pct": 100 * float(v[0]),
            "win_sd_pct": 100 * float(np.sqrt(max(v[1], 0.0))),
            "equity_rank": int(r) + 1,  # 0 for a pass outside the ranking
        }
        for m, p, v, r in zip(moves, prob, value, ranks, strict=True)
    ]
    return sorted(out, key=lambda c: -c["prob"])


def analyze(models: ReleaseModels, replay: KnownPositionsReplay, i: int) -> dict:
    """The release's models at known position `i`."""
    view = board_view(replay.positions[i])
    b = belief_at(models.beliefs, replay, i, view)
    return {
        # Seeded by the position, so every release draws its racks alike.
        "belief": models.beliefs.chart(b, np.random.default_rng(i)),
        "candidates": policy_candidates(models, replay, i, view, b),
    }


def path(paths: TagPaths, number: int) -> Path:
    return release.release_dir(paths, number) / release.POSITIONS


def write(paths: TagPaths, number: int):
    """Write release `number`'s analysis of every known position from its
    exports, which must be in place: whether its policy reads the belief, and
    each position's analysis by position id."""
    replay = _replay()
    models = ReleaseModels.load(paths, number)
    doc = {
        "inferred": models.inferred,
        "positions": {p.id: analyze(models, replay, i) for i, p in enumerate(replay.positions)},
    }
    out = path(paths, number)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(out)

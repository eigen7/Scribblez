"""What deep_bot's Positions tab (web DeepBotPositions.tsx) shows of a known
position under one release: the board before the move made there, the
release's picks at it (the database's release_pick rows), and two of the
release's models, run on the CPU under onnxruntime:

- the leave inference model's belief about the opponent's rack
  (LeaveBeliefs.chart);
- the policy over the candidates PolicyBot chooses among there
  (ffi.policy_candidates): each one's probability and its value head's win
  probability, the opponent-rack input summarized from the same belief, as
  PolicyBot reads it.

The leaf network has no output of its own at a position: it values the
rollout leaves two plies on.

The known positions replay from one .slog, game i the i-th position, with its
.forgone sidecar (KnownPositionsReplay).
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from scribblez import ffi
from scribblez.blind_spots_viewer import readable
from scribblez.deep_bot import LEAVE, POLICY, known_positions, release
from scribblez.deep_bot.known_positions import Position
from scribblez.leave_inference import forgone
from scribblez.leave_inference.beliefs import TILE_KINDS, LeaveBeliefs, RackBelief, unseen_pool
from scribblez.paths import TagPaths
from scribblez.policy_set.onnx_export import BLANK_PRUNE_KEY, LEAVE_TAG_KEY, OUTPUTS
from scribblez.sim_evidence.sobs import move_notation
from scribblez.sim_survey_viewer import (
    TOTAL_TILES,
    board_before,
    gcg_events,
    last_move,
    opp_rack_count,
    tiles_json,
    word_squares,
)

RACK_SIZE = 7
# The likeliest candidates the policy table lists, besides any that a
# reference, static equity or a pick names.
SHOWN_CANDIDATES = 25
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


def scores_before(events: list[list[str]], turn: int) -> list[int]:
    """Each seat's total before 0-based `turn`: its last event's running
    total. The seats alternate, the first event being seat 0's."""
    scores = [0, 0]
    for j, fields in enumerate(events[:turn]):
        scores[j % 2] = int(fields[-1])
    return scores


def board_view(p: Position) -> dict:
    """The position before its move, in the survey viewer's fields, with the
    manifest's answer."""
    text = p.gcg.read_text()
    turn = p.turn
    rack = gcg_events(text)[turn][1]
    board = board_before(text, turn)
    on_board = sum(cell is not None for row in board for cell in row)
    bag = max(0, TOTAL_TILES - on_board - len(rack) - RACK_SIZE)
    return {
        "id": p.id,
        "title": p.title,
        "turn": turn + 1,
        "mover": turn % 2,
        "rack": rack,
        "scores": scores_before(gcg_events(text), turn),
        "bag_size": bag,
        "opp_rack_count": opp_rack_count(board, rack, bag),
        "board": board,
        "last_move": last_move(text, turn),
        "reference": [move_json(m, board) for m in p.reference],
        "static_equity_best": move_json(p.static_equity_best, board),
    }


def spelled(notation: str, board: list[list[str | None]]) -> str:
    """A move as GCG spells it on `board`, the tiles it plays through in
    parentheses ("8H (CLINK)ER"); an exchange or a pass readable."""
    squares = word_squares(notation)
    if not squares:
        return readable(notation)
    out, through = [], False
    for r, c, ch in squares:
        if (ch == ".") != through:
            through = ch == "."
            out.append("(" if through else ")")
        out.append(board[r][c] if through else ch)
    return f"{notation.split()[0]} {''.join(out)}{')' if through else ''}"


def move_json(notation: str, board: list[list[str | None]]) -> dict:
    """A move as the tab names and draws it on the pre-move `board`."""
    return {"move": notation, "display": spelled(notation, board), "tiles": tiles_json(notation)}


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


def belief_at(models: ReleaseModels, replay: KnownPositionsReplay, i: int, view: dict):
    """The mover's belief about the opponent's rack at known position `i`."""
    (b,) = models.beliefs.beliefs(
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
            **move_json(move_notation(m, view["board"]), view["board"]),
            "prob": float(p),
            "win_pct": 100 * float(v[0]),
            "win_sd_pct": 100 * float(np.sqrt(max(v[1], 0.0))),
            "equity_rank": int(r) + 1,  # 0 for a pass outside the ranking
        }
        for m, p, v, r in zip(moves, prob, value, ranks, strict=True)
    ]
    out.sort(key=lambda c: -c["prob"])
    for rank, c in enumerate(out, start=1):
        c["policy_rank"] = rank
    return out


def shown(candidates: list[dict], named: set[str]) -> list[dict]:
    """The candidates the table lists: the SHOWN_CANDIDATES likeliest, and any
    move in `named`, in policy order."""
    return [c for c in candidates if c["policy_rank"] <= SHOWN_CANDIDATES or c["move"] in named]


@dataclass(frozen=True)
class Analysis:
    """A release's models at one known position."""

    belief: dict  # LeaveBeliefs.chart
    candidates: list[dict]  # policy_candidates
    inferred: bool  # ReleaseModels.inferred


def analyze(models: ReleaseModels, replay: KnownPositionsReplay, i: int, view: dict) -> Analysis:
    """The release's models at known position `i`, whose board_view is
    `view`."""
    b = belief_at(models, replay, i, view)
    # Seeded by the position, so a reload draws the same racks.
    chart = models.beliefs.chart(b, np.random.default_rng(i))
    return Analysis(chart, policy_candidates(models, replay, i, view, b), models.inferred)


def pick_json(row: dict, board: list[list[str | None]]) -> dict:
    """A release_pick row as the tab shows it."""
    return {**move_json(row["pick"], board), "reading": row["reading"], "seconds": row["seconds"]}


def payload(view: dict, number: int, pick_rows: dict[str, dict], a: Analysis) -> dict:
    """The tab's view of a known position under release `number`: its
    board_view, the release's picks there (release_pick rows by reply model),
    and the analysis, the policy table cut to the moves worth listing."""
    picks = {reply: pick_json(row, view["board"]) for reply, row in pick_rows.items()}
    named = {m["move"] for m in [*view["reference"], view["static_equity_best"], *picks.values()]}
    return {
        **view,
        "release": number,
        "picks": picks,
        "belief": {**a.belief, "model": f"release {number}"},
        "policy": {
            "num_candidates": len(a.candidates),
            "inferred": a.inferred,
            "candidates": shown(a.candidates, named),
        },
    }


def release_list(paths: TagPaths, picked: set[int]) -> list[dict]:
    """The tag's complete releases, ascending, each with its models' rows
    trained and whether its picks (the releases in `picked`) are recorded."""
    return [
        {
            "release": n,
            "rows_trained": release.read_manifest(paths, n)["rows_trained"],
            "picked": n in picked,
        }
        for n in release.numbers(paths)
    ]

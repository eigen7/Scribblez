"""The data behind the blind-spots viewer: the sim-survey viewer (web
?tool=survey) fed from a hidden_blind_spots tag instead of a CPU survey.

Each kept position of the tag's .blindspots.json reports
(blind_spot_confirm_tool) becomes one viewer entry: the board and the mover's
rack, rebuilt from the position's exported .gcg as sim_survey_viewer does, and
the confirmed moves, outside moves first. The GPU sims keep only win rates, so
a move carries its confirming and screening win% and the pool's q instead of a
CPU survey's rollout summaries. Every outside move is compared with the best
top move on the same rollouts.

BeliefPanel adds what the mover believes about the opponent's rack, from the
leave inference model the reports name (leave_inference/beliefs.py): per tile,
the chance the rack holds one under the model and under no information, and
sample racks on request.
"""

import json
from pathlib import Path

import numpy as np

from scribblez.leave_inference.beliefs import LeaveBeliefs, RackBelief, unseen_pool
from scribblez.paths import TagPaths
from scribblez.sim_survey_viewer import (
    TILE_SCORES,
    board_before,
    bonuses,
    gcg_events,
    last_move,
    opp_rack_count,
    tiles_json,
)

REPORT_SUFFIX = ".blindspots.json"


def played_move(gcg_text: str) -> str:
    """The move the game played at a position, in GCG notation: the last event
    of its export, which runs through that move."""
    fields = gcg_events(gcg_text)[-1]
    return fields[2] if fields[2].startswith("-") else " ".join(fields[2:4])


def readable(notation: str) -> str:
    """GCG's "-" (a pass) and "-ABC" (an exchange) spelled out; a play as is."""
    if notation == "-":
        return "pass"
    return f"exch {notation[1:]}" if notation.startswith("-") else notation


def move_entry(position: dict, c: int, header: dict) -> dict:
    cand = position["candidates"][c]
    best_cut = position["candidates"][position["best_cut"]]
    out = {
        "move": cand["move"],
        "display": readable(cand["display"]),
        "hasty_rank": cand["equity_rank"] + 1,  # 0 for a pass outside the ranking
        "score": cand["score"],
        "leave": cand["leave"],
        "is_setup": False,
        "tiles": tiles_json(cand["move"]),
        "outside": not cand["in_cut"],
        "win_pct": 100 * cand["win_prob"],
        "rollouts": header["rollouts"],
        "screen_win_pct": 100 * cand["screen_win_prob"],
        "screen_q": cand["screen_q"],
    }
    if not cand["in_cut"]:
        gain = 100 * (cand["win_prob"] - best_cut["win_prob"])
        out |= {
            "gain_pct": gain,
            "beats_cut": gain >= header["min_gain"],
            "versus": readable(best_cut["display"]),
        }
    return out


def position_entry(stem: str, position: dict, header: dict, gcg_dir: Path) -> dict:
    """The viewer's entry for one kept position."""
    gcg = f"{stem}-g{position['game']}-turn{position['turn'] + 1}.gcg"
    text = (gcg_dir / gcg).read_text()
    board = board_before(text, position["turn"])
    moves = [move_entry(position, c, header) for c in range(len(position["candidates"]))]
    outside = sorted((m for m in moves if m["outside"]), key=lambda m: -m["win_pct"])
    inside = sorted((m for m in moves if not m["outside"]), key=lambda m: m["hasty_rank"])
    return {
        "name": gcg.removesuffix(".gcg"),
        "gcg": gcg,
        "stem": stem,
        "game": position["game"],
        "turn": position["turn"] + 1,
        "mover": position["mover"],
        "rack": position["rack"],
        "opp_known_leave": "",  # hidden leaves
        "scores": position["scores"],
        "bag_size": position["bag_size"],
        "solved_endgames": False,  # the pool tool's rollouts play endgames out by HastyBot
        "played": played_move(text),
        "board": board,
        "last_move": last_move(text, position["turn"]),
        "opp_rack_count": opp_rack_count(board, position["rack"], position["bag_size"]),
        "moves": outside + inside,
    }


def viewer_data(found_dir: Path, gcg_dir: Path, min_bag: int = 0) -> dict:
    """Everything the viewer loads: the kept positions of `found_dir`'s reports
    with at least `min_bag` tiles in the bag, in name order (by file stem, then
    game and turn), their games read from `gcg_dir`."""
    found = []  # ((stem, game, turn), entry)
    header = {}
    for path in sorted(found_dir.glob(f"*{REPORT_SUFFIX}")):
        report = json.loads(path.read_text())
        header = {k: v for k, v in report.items() if k != "positions"}
        stem = path.name.removesuffix(REPORT_SUFFIX)
        for position in report["positions"]:
            if position["kept"] and position["bag_size"] >= min_bag:
                entry = position_entry(stem, position, header, gcg_dir)
                found.append(((stem, position["game"], position["turn"]), entry))
    # Turns compare as numbers, so turn9 precedes turn10 unlike in a string sort.
    positions = [entry for _, entry in sorted(found, key=lambda kv: kv[0])]
    return {
        "survey": header,
        "bonuses": bonuses(),
        "tile_scores": TILE_SCORES,
        "positions": positions,
    }


TILE_NAMES = [chr(ord("A") + i) for i in range(26)] + ["?"]


def report_header(found_dir: Path) -> dict:
    """The first report's header; every report of a tag has the same."""
    path = next(iter(sorted(found_dir.glob(f"*{REPORT_SUFFIX}"))), None)
    if path is None:
        return {}
    return {k: v for k, v in json.loads(path.read_text()).items() if k != "positions"}


def leave_model_path(header: dict, mount_root: Path) -> Path | None:
    """The leave inference export the reports' pools were labeled under, or
    None for uniform beliefs."""
    model = header.get("leave_model", {})
    if not model.get("tag"):
        return None
    return TagPaths(model["tag"], "leave_inference", mount_root).onnx_path(model["generation"])


class BeliefPanel:
    """The viewer's opponent-rack panel for `data`'s positions, from `model`:
    each position's letter chart, added to its entry as "belief", and sample
    racks on request (samples)."""

    def __init__(self, data: dict, found_dir: Path, model: LeaveBeliefs):
        self.model = model
        self.beliefs = self._beliefs(data["positions"], found_dir)
        for entry, belief in zip(data["positions"], self.beliefs, strict=True):
            entry["belief"] = self._chart(belief)

    def _beliefs(self, entries: list[dict], found_dir: Path) -> list[RackBelief]:
        by_stem: dict[str, list[int]] = {}
        for i, e in enumerate(entries):
            by_stem.setdefault(e["stem"], []).append(i)
        out: list[RackBelief | None] = [None] * len(entries)
        for stem, indices in by_stem.items():
            es = [entries[i] for i in indices]
            beliefs = self.model.beliefs(
                found_dir / f"{stem}.slog",
                [(e["game"], e["turn"] - 1) for e in es],
                [unseen_pool(e["board"], e["rack"]) for e in es],
                [e["opp_rack_count"] for e in es],
            )
            for i, b in zip(indices, beliefs, strict=True):
                out[i] = b
        return out

    def _chart(self, b: RackBelief) -> dict:
        inferred = self.model.rack_presence(b)
        uniform = self.model.rack_presence(b, uniform=True)
        return {
            "modelled": b.modelled,
            "leave_length": b.leave_length,
            "refill": b.refill,
            "presence": [
                {"tile": name, "inferred": float(i), "uniform": float(u)}
                for name, i, u in zip(TILE_NAMES, inferred, uniform, strict=True)
            ],
        }

    def samples(self, query: dict[str, list[str]]) -> dict:
        """GET /api/leave_samples?position=<index>&seed=<int>[&n=<count>]: up
        to n (10) distinct racks from the position's belief."""
        index = int(query["position"][0])
        rng = np.random.default_rng(int(query["seed"][0]))
        n = int(query.get("n", ["10"])[0])
        return {"racks": self.model.sample_racks(self.beliefs[index], n, rng)}

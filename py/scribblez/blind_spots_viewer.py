"""The data behind the blind-spots viewer: the sim-survey viewer (web
?tool=survey) fed from a hidden_blind_spots tag instead of a CPU survey.

Each kept position of the tag's .blindspots.json reports
(blind_spot_confirm_tool) becomes one viewer entry: the board and the mover's
rack, rebuilt from the position's exported .gcg as sim_survey_viewer does, and
the confirmed moves, outside moves first. The GPU sims keep only win rates, so
a move carries its confirming and screening win% and the pool's q instead of a
CPU survey's rollout summaries. Every outside move is compared with the best
top move on the same rollouts.

A position's entry also carries what the mover believes about the opponent's
rack, as "belief", when its report has a .beliefs.json sidecar
(blind_spots_beliefs.py writes them; the viewer never runs the model).
"""

import json
from pathlib import Path

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
BELIEFS_SUFFIX = ".beliefs.json"


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


def report_entries(report_path: Path, gcg_dir: Path, min_bag: int = 0) -> tuple[dict, list]:
    """The header of the report at `report_path` and the entries of its kept
    positions with at least `min_bag` tiles in the bag, in report order."""
    report = json.loads(report_path.read_text())
    header = {k: v for k, v in report.items() if k != "positions"}
    stem = report_path.name.removesuffix(REPORT_SUFFIX)
    entries = [
        position_entry(stem, p, header, gcg_dir)
        for p in report["positions"]
        if p["kept"] and p["bag_size"] >= min_bag
    ]
    return header, entries


def model_name(model: dict) -> str:
    return f"{model['tag']} gen {model['generation']}"


def attach_beliefs(entries: list[dict], header: dict, beliefs_path: Path):
    """Add each entry's belief from the sidecar at `beliefs_path`, naming the
    model it came from and, for a stand-in, the one the pools were labeled
    under."""
    sidecar = json.loads(beliefs_path.read_text())
    by_position = {(b["game"], b["turn"]): b for b in sidecar["positions"]}
    model, labeled = sidecar["leave_model"], header.get("leave_model", {})
    note = {"model": model_name(model)}
    if (model["tag"], model["generation"]) != (labeled.get("tag"), labeled.get("generation")):
        note["stand_in_for"] = model_name(labeled) if labeled.get("tag") else "uniform beliefs"
    for e in entries:
        b = by_position[(e["game"], e["turn"] - 1)]
        e["belief"] = {k: v for k, v in b.items() if k not in ("game", "turn")} | note


def viewer_data(found_dir: Path, gcg_dir: Path, min_bag: int = 0) -> dict:
    """Everything the viewer loads: the kept positions of `found_dir`'s reports
    with at least `min_bag` tiles in the bag, in name order (by file stem, then
    game and turn), their games read from `gcg_dir`, with their beliefs where
    the reports have sidecars."""
    positions = []
    header = {}
    for path in sorted(found_dir.glob(f"*{REPORT_SUFFIX}")):
        header, entries = report_entries(path, gcg_dir, min_bag)
        beliefs = path.with_name(path.name.removesuffix(REPORT_SUFFIX) + BELIEFS_SUFFIX)
        if beliefs.exists():
            attach_beliefs(entries, header, beliefs)
        positions += entries
    # Turns compare as numbers, so turn9 precedes turn10 unlike in a string sort.
    positions.sort(key=lambda e: (e["stem"], e["game"], e["turn"]))
    return {
        "survey": header,
        "bonuses": bonuses(),
        "tile_scores": TILE_SCORES,
        "positions": positions,
    }


def report_header(found_dir: Path) -> dict:
    """The first report's header; every report of a tag has the same."""
    path = next(iter(sorted(found_dir.glob(f"*{REPORT_SUFFIX}"))), None)
    if path is None:
        return {}
    return {k: v for k, v in json.loads(path.read_text()).items() if k != "positions"}

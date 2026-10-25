"""The data behind the blind-spots viewer: the sim-survey viewer (web
?tool=survey) fed from a hidden_blind_spots tag instead of a CPU survey.

Each kept position of the tag's .blindspots.json reports
(blind_spot_confirm_tool) becomes one viewer entry: the board and the mover's
rack, rebuilt from the position's exported .gcg as sim_survey_viewer does, and
the confirmed moves, outside moves first. The GPU sims keep only win rates, so
a move carries its confirming and screening win% and the pool's q instead of a
CPU survey's rollout summaries. Every outside move is compared with the best
top move on the same rollouts.
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

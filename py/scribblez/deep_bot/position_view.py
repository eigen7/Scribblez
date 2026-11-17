"""What deep_bot's Positions tab (web DeepBotPositions.tsx) shows of a known
position under one release: the board before the move made there, the
release's picks at it (the database's release_pick rows), and what the
trainer computed of the release's models there as it published the release
(deep_bot/release_positions.py): the leave inference model's belief about the
opponent's rack, and the policy over PolicyBot's candidates.
"""

from scribblez.blind_spots_viewer import readable
from scribblez.deep_bot import release
from scribblez.deep_bot.known_positions import Position
from scribblez.paths import TagPaths
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


def policy_json(candidates: list[dict], named: set[str], board) -> list[dict]:
    """The policy table's rows from release_positions' candidates, likeliest
    first: the SHOWN_CANDIDATES likeliest and any move in `named`, each with
    its rank under the policy and drawn on the pre-move `board`."""
    return [
        {**c, **move_json(c["move"], board), "policy_rank": rank}
        for rank, c in enumerate(candidates, start=1)
        if rank <= SHOWN_CANDIDATES or c["move"] in named
    ]


def pick_json(row: dict, board: list[list[str | None]]) -> dict:
    """A release_pick row as the tab shows it."""
    return {**move_json(row["pick"], board), "reading": row["reading"], "seconds": row["seconds"]}


def payload(view: dict, number: int, pick_rows: dict[str, dict], analysis: dict | None) -> dict:
    """The tab's view of a known position under release `number`: its
    board_view, the release's picks there (release_pick rows by reply model),
    and the release's analysis of it (release_positions.write: its
    "inferred" flag and its entry for the position), the policy table cut to
    the moves worth listing. Without an analysis, a release published before
    the trainer wrote one, its model outputs are null."""
    picks = {reply: pick_json(row, view["board"]) for reply, row in pick_rows.items()}
    out = {**view, "release": number, "picks": picks, "belief": None, "policy": None}
    if analysis is None:
        return out
    named = {m["move"] for m in [*view["reference"], view["static_equity_best"], *picks.values()]}
    candidates = analysis["candidates"]
    return out | {
        "belief": {**analysis["belief"], "model": f"release {number}"},
        "policy": {
            "num_candidates": len(candidates),
            "inferred": analysis["inferred"],
            "candidates": policy_json(candidates, named, view["board"]),
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

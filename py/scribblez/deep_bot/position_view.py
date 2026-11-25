"""A known position as the leaf Positions tab draws it
(dashboard/deep_bot_leaf_eval_api.py): the board before the move made there,
with the manifest's answer, and a move as the tab names and draws it.
"""

from scribblez.deep_bot.known_positions import Position
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


def readable(notation: str) -> str:
    """GCG's "-" (a pass) and "-ABC" (an exchange) spelled out; a play as is."""
    if notation == "-":
        return "pass"
    return f"exch {notation[1:]}" if notation.startswith("-") else notation


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

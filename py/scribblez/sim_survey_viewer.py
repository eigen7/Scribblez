"""The data behind the sim-survey viewer (web ?tool=survey).

Joins a finished sim candidate survey (sim_candidate_survey.py) into what the
browser draws, one entry per position where a play from outside the HastyBot
top moves beat them: the board at the decision point, the mover's rack, and
the confirming sim's moves -- the outside plays first, then the top moves --
each with the squares it places and its sim statistics.

The survey files name moves in GCG notation and do not store boards, so the
board is rebuilt here by replaying the position's exported .gcg, which takes
nothing more than reading that notation: a position ("K6" reads down column
K from row 6, "6K" across row 6) and a word whose '.' squares are already on
the board and whose lowercase letters are blanks.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from scribblez.sim_candidate_survey import (
    MIN_SIGMA,
    SURVEY_SUFFIX,
    Finding,
    gcg_name,
    position_findings,
)

BOARD_SIZE = 15

# The standard premium layout, one row per string: '=' triple word, '-' double
# word, '"' triple letter, "'" double letter.
PREMIUM_ROWS = (
    "=  '   =   '  =",
    ' -   "   "   - ',
    "  -   ' '   -  ",
    "'  -   '   -  '",
    "    -     -    ",
    ' "   "   "   " ',
    "  '   ' '   '  ",
    "=  '   -   '  =",
)
PREMIUM_NAMES = {"=": "TW", "-": "DW", '"': "TL", "'": "DL", " ": None}

TILE_SCORES = {
    **dict.fromkeys("AEILNORSTU", 1),
    **dict.fromkeys("DG", 2),
    **dict.fromkeys("BCMP", 3),
    **dict.fromkeys("FHVWY", 4),
    "K": 5,
    **dict.fromkeys("JX", 8),
    **dict.fromkeys("QZ", 10),
}

PLAY_NOTATION = re.compile(r"^(?:([A-O])(\d{1,2})|(\d{1,2})([A-O])) (\S+)$")


def bonuses() -> list[list[str | None]]:
    rows = [*PREMIUM_ROWS, *reversed(PREMIUM_ROWS[:-1])]
    return [[PREMIUM_NAMES[ch] for ch in row] for row in rows]


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    letter: str  # uppercase
    is_blank: bool


def placed_tiles(notation: str) -> list[Tile]:
    """The tiles a move in GCG notation lays on the board; none for an exchange
    ("-ABC") or a pass ("-")."""
    match = PLAY_NOTATION.match(notation)
    if not match:
        return []
    down_col, down_row, across_row, across_col, word = match.groups()
    vertical = down_col is not None
    row = int(down_row if vertical else across_row) - 1
    col = ord(down_col if vertical else across_col) - ord("A")
    tiles = []
    for i, ch in enumerate(word):
        if ch == ".":
            continue
        r, c = (row + i, col) if vertical else (row, col + i)
        tiles.append(Tile(r, c, ch.upper(), ch.islower()))
    return tiles


def board_before(gcg_text: str, turn: int) -> list[list[str | None]]:
    """The board before 0-based `turn` of a .gcg, in the web Board's encoding: a
    letter per tile, lowercase for a blank."""
    board: list[list[str | None]] = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    events = [line for line in gcg_text.splitlines() if line.startswith(">")]
    for line in events[:turn]:
        # ">nick: RACK POS WORD +score total"; an exchange or pass has no WORD.
        fields = line.split()
        for t in placed_tiles(" ".join(fields[2:4])):
            board[t.row][t.col] = t.letter.lower() if t.is_blank else t.letter
    return board


def per_rollout(stats: dict, key: str, n: int) -> float:
    return stats[key] / n


def move_stats(summary: dict) -> dict:
    """The confirming sim's statistics for one move, as the viewer's side panel
    shows them: rates and per-rollout means rather than raw sums."""
    n = summary["n"]
    out = {
        "rollouts": n,
        "win_pct": 100 * (summary["wins"] + 0.5 * summary["draws"]) / n,
        "spread": summary["delta_sum"] / n,
        "spread_sd": max(summary["delta_sq_sum"] / n - (summary["delta_sum"] / n) ** 2, 0) ** 0.5,
        "delta_hist": summary["delta_hist"],
        "end_swing": summary["end_swing_sum"] / n,
        "opp_stranded": summary["opp_stranded_sum"] / n,
        "self_stranded": summary["self_stranded_sum"] / n,
        "self_went_out_pct": 100 * summary["self_went_out"] / n,
        "opp_went_out_pct": 100 * summary["opp_went_out"] / n,
    }
    for side in ("opp_reply", "self_next"):
        s = summary[side]
        out[side] = {
            "score": s["score_sum"] / n,
            "bingo_pct": 100 * s["bingos"] / n,
            "non_play_pct": 100 * s["non_plays"] / n,
            "adjacent_pct": 100 * s["adjacent"] / n,
            "score_hist": s["score_hist"],
        }
    return out


def move_entry(position: dict, entry: dict, finding: Finding | None) -> dict:
    c = position["candidates"][entry["candidate"]]
    out = {
        "move": c["move"],
        "hasty_rank": c["equity_rank"] + 1,
        "equity": c["equity"],
        "score": c["score"],
        "leave": c["leave"],
        "is_setup": c["is_setup"],
        "tiles": [
            {"row": t.row, "col": t.col, "letter": t.letter, "isBlank": t.is_blank}
            for t in placed_tiles(c["move"])
        ],
        "stats": move_stats(entry["summary"]),
    }
    if finding:  # an outside play: how it fared against the best top move
        out |= {
            "gain_pct": 100 * finding.gain,
            "sigmas": finding.sigmas,
            "beats_cut": finding.beats_cut,
            "versus": finding.inside.move,
        }
    return out


def position_entry(
    stem: str, position: dict, cut: int, gcg_dir: Path, min_sigmas: float
) -> dict | None:
    """The viewer's entry for one surveyed position, or None when no outside play
    sat `min_sigmas` above the cut's best there."""
    key = (stem, position["game"], position["turn"])
    found = {f.outside.move: f for f in position_findings(key, position, cut)}
    if not any(f.sigmas >= min_sigmas for f in found.values()):
        return None
    moves = [
        move_entry(position, e, found.get(position["candidates"][e["candidate"]]["move"]))
        for e in position["confirm"]
    ]
    outside = sorted((m for m in moves if "sigmas" in m), key=lambda m: -m["stats"]["win_pct"])
    inside = sorted((m for m in moves if "sigmas" not in m), key=lambda m: m["hasty_rank"])
    gcg = gcg_name(key)
    return {
        "name": gcg.removesuffix(".gcg"),
        "gcg": gcg,
        "turn": position["turn"] + 1,
        "mover": position["mover"],
        "rack": position["rack"],
        "opp_known_leave": position["opp_known_leave"],
        "scores": position["scores"],
        "bag_size": position["bag_size"],
        "num_legal_moves": position["num_legal_moves"],
        "played": position["played"],
        "board": board_before((gcg_dir / gcg).read_text(), position["turn"]),
        "moves": outside + inside,
    }


def viewer_data(survey_dir: Path, min_sigmas: float = MIN_SIGMA) -> dict:
    """Everything the viewer loads: the positions of `survey_dir`'s survey files
    where an outside play sat `min_sigmas` standard errors above the cut's best,
    strongest first."""
    positions = []
    header = {}
    for path in sorted(survey_dir.glob(f"*{SURVEY_SUFFIX}")):
        survey = json.loads(path.read_text())
        header = {k: v for k, v in survey.items() if k != "positions"}
        stem = path.name.removesuffix(SURVEY_SUFFIX)
        for position in survey["positions"]:
            entry = position_entry(stem, position, survey["cut"], survey_dir / "gcg", min_sigmas)
            if entry:
                positions.append(entry)
    positions.sort(key=lambda p: -max(m.get("sigmas", 0) for m in p["moves"]))
    return {
        "survey": header,
        "bonuses": bonuses(),
        "tile_scores": TILE_SCORES,
        "positions": positions,
    }

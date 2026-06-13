"""Parse Scribblez GCG game logs into a replayable game dict.

GCG is the standard Scrabble game-log format (Macondo/Quackle). It records each
play as a position + word (with ``.`` for tiles already on the board and a
lowercase letter for a designated blank), but not the per-turn placed-tile list,
bag size, or both players' running scores. This parser reconstructs those by
replaying the board forward, so the rest of the pipeline can stay format-blind.

``parse_gcg`` returns a dict shaped like the old JSON log::

    {
      "players": [{"name": ...}, {"name": ...}],
      "turns": [{"player", "rack_before", "cumulative_scores",
                 "score_delta", "bag_size_before", "move"}, ...],
      "final_scores": [s0, s1],
    }
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Standard English set: 100 tiles, two full 7-tile racks dealt at the start.
_INITIAL_BAG = 86

_ACROSS_RE = re.compile(r"^(\d+)([A-O])$")  # e.g. "8D"  (horizontal)
_DOWN_RE = re.compile(r"^([A-O])(\d+)$")  # e.g. "D8"  (vertical)


def _parse_position(pos: str) -> tuple[bool, int, int]:
    """Return (horizontal, start_row, start_col) for a GCG coordinate."""
    if pos[0].isdigit():
        m = _ACROSS_RE.match(pos)
        assert m, f"bad position {pos!r}"
        return True, int(m.group(1)) - 1, ord(m.group(2)) - ord("A")
    m = _DOWN_RE.match(pos)
    assert m, f"bad position {pos!r}"
    return False, int(m.group(2)) - 1, ord(m.group(1)) - ord("A")


def parse_gcg(path) -> Dict[str, Any]:
    nick_to_player: Dict[str, int] = {}
    player_names: List[str] = ["", ""]

    board: List[List[int | None]] = [[None] * 15 for _ in range(15)]  # letter index or None
    cumulative = [0, 0]
    bag = _INITIAL_BAG
    turns: List[Dict[str, Any]] = []

    def record(player: int, rack: str, move: Dict[str, Any], score_delta: int,
               bag_before: int):
        turns.append({
            "player": player,
            "rack_before": rack,
            "cumulative_scores": list(cumulative),  # after the move
            "score_delta": score_delta,
            "bag_size_before": bag_before,
            "move": move,
        })

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#player1") or line.startswith("#player2"):
                pn = 0 if line.startswith("#player1") else 1
                parts = line.split(maxsplit=2)
                nick = parts[1]
                nick_to_player[nick] = pn
                player_names[pn] = parts[2] if len(parts) > 2 else nick
                continue
            if not line.startswith(">"):
                continue  # other pragmata / notes

            colon = line.index(":")
            nick = line[1:colon]
            player = nick_to_player.get(nick)
            if player is None:
                continue
            toks = line[colon + 1:].split()
            if not toks:
                continue

            # End-of-game rack adjustment: "(TILES) +pts cum" -- updates the
            # final score but is not a turn.
            if toks[0].startswith("("):
                cumulative[player] = int(toks[-1])
                continue

            rack = toks[0]
            arg = toks[1] if len(toks) > 1 else ""

            if arg.startswith("("):  # END_RACK_PENALTY: "RACK (RACK) -pts cum"
                cumulative[player] = int(toks[-1])
                continue
            if arg == "-":  # pass: "RACK - +0 cum"
                cumulative[player] = int(toks[-1])
                record(player, rack, {"type": "pass"}, 0, bag)
                continue
            if arg.startswith("-"):  # exchange: "RACK -TILES +0 cum"
                cumulative[player] = int(toks[-1])
                record(player, rack, {"type": "exchange", "exchanged": arg[1:]}, 0, bag)
                continue

            # Tile placement: "RACK POS WORD +score cum"
            pos, word, score_tok, cum_tok = toks[1], toks[2], toks[3], toks[4]
            score = int(score_tok.lstrip("+"))
            cumulative[player] = int(cum_tok)
            horizontal, sr, sc = _parse_position(pos)

            tiles: List[Dict[str, Any]] = []
            main_word: List[str] = []
            for i, ch in enumerate(word):
                r = sr if horizontal else sr + i
                c = sc + i if horizontal else sc
                if ch == ".":  # a tile already on the board
                    existing = board[r][c]
                    assert existing is not None, f"play-through on empty square at {r},{c}"
                    main_word.append(chr(ord("A") + existing))
                    continue
                letter = ch.upper()
                tiles.append({"row": r, "col": c, "letter": letter, "is_blank": ch.islower()})
                main_word.append(letter)
                board[r][c] = ord(letter) - ord("A")

            move = {
                "type": "play",
                "horizontal": horizontal,
                "start_row": sr,
                "start_col": sc,
                "main_word": "".join(main_word),
                "score": score,
                "tiles": tiles,
            }
            bag_before = bag
            bag = max(0, bag - len(tiles))  # refill draws one tile per placed tile
            record(player, rack, move, score, bag_before)

    return {
        "players": [{"name": player_names[0]}, {"name": player_names[1]}],
        "turns": turns,
        "final_scores": [cumulative[0], cumulative[1]],
    }

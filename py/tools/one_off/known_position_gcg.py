#!/usr/bin/env python3
"""Cut a recorded game down to one of its positions, as a GCG the engine can
replay (gcg_to_slog), for positions/NWL23/known-positions.

    py/tools/one_off/known_position_gcg.py game.gcg --ply 21 --out richards-eellt.gcg

The output keeps the game's pragmas and its moves up to and including ply
`--ply` (0-based), the move played at the position, and drops the notes.

A human game's record often cannot be replayed: a rack written down for the
opponent, from memory or by inference, lacks tiles the same player kept the
turn before. Each player's racks are therefore made to follow from one
another, working back from the last: where a turn's kept tiles are not all in
the next rack, the missing ones are replaced by tiles that are. The moves,
the scores and the last rack of each player are untouched, so the position
asked about is exactly the record's: its board, the mover's rack, the scores
and the unseen tiles.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

MOVE = re.compile(r"^>(?P<name>[^:]+): (?P<rack>\S+) (?P<rest>.*)$")


def played_tiles(rest: str) -> Counter:
    """The rack tiles a move line's remainder spends: a play's new tiles (a
    lowercase letter is the blank), an exchange's, or none for a pass."""
    first, *more = rest.split()
    if first == "-":
        return Counter()
    if first.startswith("-"):
        return Counter(first[1:])
    return Counter("?" if c.islower() else c for c in more[0] if c != ".")


def following(rack: Counter, spent: Counter, next_rack: Counter) -> Counter:
    """`rack` with the tiles it keeps after spending `spent` made a part of
    `next_rack`: the kept tiles `next_rack` lacks give way to ones it has."""
    kept = rack - spent
    stays = kept & next_rack
    fill = list((next_rack - stays).elements())[: sum(kept.values()) - sum(stays.values())]
    return spent + stays + Counter(fill)


def consistent(moves: list[re.Match]) -> list[str]:
    """Each move's rack, with every player's racks following from one another
    (the module docstring's rule)."""
    racks = [Counter(m["rack"]) for m in moves]
    later: dict[str, Counter] = {}
    for i in reversed(range(len(moves))):
        name = moves[i]["name"]
        if name in later:
            racks[i] = following(racks[i], played_tiles(moves[i]["rest"]), later[name])
        later[name] = racks[i]
    return ["".join(sorted(r.elements())) for r in racks]


def cut(text: str, ply: int) -> str:
    lines = text.splitlines()
    first_move = next(i for i, line in enumerate(lines) if line.startswith(">"))
    pragmas = [ln for ln in lines[:first_move] if ln.startswith("#") and not ln.startswith("#note")]
    moves = [m for m in map(MOVE.match, lines) if m][: ply + 1]
    if len(moves) != ply + 1:
        raise SystemExit(f"the game has only {len(moves)} moves")
    body = [
        f">{m['name']}: {rack} {m['rest']}"
        for m, rack in zip(moves, consistent(moves), strict=True)
    ]
    return "\n".join([*pragmas, *body]) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("gcg", type=Path)
    p.add_argument("--ply", type=int, required=True, help="the position's move, 0-based")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.write_text(cut(args.gcg.read_text(), args.ply))
    return 0


if __name__ == "__main__":
    sys.exit(main())

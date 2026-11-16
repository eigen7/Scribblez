#!/usr/bin/env python3
"""What an agent plays at the known positions
(positions/NWL23/known-positions): asks it for its move at each one and reads
every pick against the position's answer.

    python3 -m scripts.deep_bot.known_position_picks \\
        --player "--type=deepbot-endgame --leaf-model leaf.onnx --reply hasty"

A pick is the position's reference move, static equity's move, or another.
--out keeps the picks as JSON lines, one per position, with how each was
read.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from scribblez.deep_bot import known_positions
from scribblez.hardware import default_thread_count


def table(positions, picks) -> str:
    """The picks as lines, then the count of each reading."""
    readings = (known_positions.REFERENCE, known_positions.EQUITY, known_positions.OTHER)
    lines, counts = [], dict.fromkeys(readings, 0)
    for p, pick in zip(positions, picks, strict=True):
        reading = p.read(pick["pick"])
        counts[reading] += 1
        wanted = f"{' or '.join(p.reference)} (#{p.reference_equity_rank})"
        lines.append(
            f"{p.id:16} {pick['pick']:18} {reading:10} reference {wanted:20} "
            f"{pick['seconds']:6.1f} s"
        )
    summary = ", ".join(f"{n} {name}" for name, n in counts.items())
    return "\n".join([*lines, f"{len(picks)} positions: {summary}"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--player", required=True, help="the agent, as play_game's --player names it")
    p.add_argument("--threads", type=int, default=default_thread_count())
    p.add_argument("--out", type=Path, help="write the picks here, a JSON line each")
    argv = sys.argv[1:] if argv is None else list(argv)
    # A one-word spec starts with "--", which argparse takes for an option.
    if "--player" in argv[:-1]:
        i = argv.index("--player")
        argv[i : i + 2] = [f"--player={argv[i + 1]}"]
    args = p.parse_args(argv)
    positions = known_positions.load()
    with tempfile.TemporaryDirectory() as work:
        picks = known_positions.ask(positions, args.player, Path(work), args.threads)
    if args.out:
        rows = [
            {"id": pos.id, **pick, "reading": pos.read(pick["pick"])}
            for pos, pick in zip(positions, picks, strict=True)
        ]
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(table(positions, picks))
    return 0


if __name__ == "__main__":
    sys.exit(main())

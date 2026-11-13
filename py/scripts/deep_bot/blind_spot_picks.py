#!/usr/bin/env python3
"""What an agent plays at the hidden-leaves blind spots (docs/blind_spots.md):
asks it for its move at a hidden_blind_spots tag's kept positions
(engine position_pick_tool, each game replayed to the position) and reads
every pick against the position's confirming sim.

    python3 -m scripts.deep_bot.blind_spot_picks --tag v1 --out picks.jsonl \\
        --player "--type=deepbot-endgame --leaf-model leaf.onnx --reply hasty"
    python3 -m scripts.deep_bot.blind_spot_picks --tag v1 --picks picks.jsonl

A report sims the static-equity top moves (the cut) and its best few moves
outside them. A pick is the report's best outside move, another of its outside
moves, a cut move, or a move the report never simmed; for the first three the
table gives the pick's confirmed win probability over the best cut move's, in
points. That sim's opponent is HastyBot, so a pick that turns the outside
move down is not thereby wrong: an agent that models a searching opponent
should turn some down. Positions whose best outside move is a pass are left
out unless --passes: against an opponent who can pass back, those are
artifacts of the sim.
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scribblez.hardware import default_thread_count
from scribblez.paths import ENGINE_DIR, TagPaths, add_mount_root_argument

PICK_TOOL = ENGINE_DIR / "position_pick_tool"
PASS = "-"
KINDS = ("best_outside", "other_outside", "cut", "unsimmed")


@dataclass
class Spot:
    """A kept blind spot: where it is, and its report's candidates."""

    slog: str
    game: int
    turn: int
    bag: int
    best_cut: dict
    best_outside: dict
    candidates: dict[str, dict]  # by move notation

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.slog, self.game, self.turn)


def read_spots(found_dir: Path) -> list[Spot]:
    """Every kept position of the .blindspots.json reports in `found_dir`."""
    spots = []
    for report in sorted(found_dir.glob("*.blindspots.json")):
        slog = str(report.with_name(report.name.removesuffix(".blindspots.json") + ".slog"))
        for p in json.loads(report.read_text())["positions"]:
            if not p["kept"]:
                continue
            c = p["candidates"]
            spots.append(
                Spot(
                    slog,
                    p["game"],
                    p["turn"],
                    p["bag_size"],
                    c[p["best_cut"]],
                    c[p["best_outside"]],
                    {x["move"]: x for x in c},
                )
            )
    return spots


def select(spots: list[Spot], args) -> list[Spot]:
    """The spots the flags keep, a seeded sample of `--limit` of them."""
    kept = [
        s
        for s in spots
        if args.min_bag <= s.bag <= args.max_bag and (args.passes or s.best_outside["move"] != PASS)
    ]
    if args.limit and len(kept) > args.limit:
        kept = sorted(random.Random(args.seed).sample(kept, args.limit), key=lambda s: s.key)
    return kept


def run_tool(spots: list[Spot], args) -> list[dict]:
    """The agent's picks at `spots`, written to --out when given."""
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "positions.txt"
        listing.write_text("".join(f"{s.slog} {s.game} {s.turn}\n" for s in spots))
        out = args.out or Path(tmp) / "picks.jsonl"
        cmd = [str(PICK_TOOL), "--positions", str(listing), "--out", str(out)]
        cmd += ["--player", args.player, "--threads", str(args.threads)]
        subprocess.run(cmd, check=True)
        return read_picks(out)


def read_picks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def kind_of(spot: Spot, pick: str) -> str:
    """Which of KINDS the pick is at `spot`."""
    if pick == spot.best_outside["move"]:
        return "best_outside"
    if pick not in spot.candidates:
        return "unsimmed"
    return "cut" if spot.candidates[pick]["in_cut"] else "other_outside"


def bag_bucket(bag: int) -> str:
    if bag >= 8:
        return "bag 8+"
    if bag >= 4:
        return "bag 4-7"
    return "bag 2-3" if bag >= 2 else "bag 1"


def tabulate(spots: list[Spot], picks: list[dict]) -> dict[str, dict[str, float]]:
    """Per bag bucket (and "all"): the positions picked at, the share of each
    kind of pick, and over the picks the report simmed their mean confirmed
    win probability over the best cut move's ("gain") beside the best outside
    move's ("available")."""
    by_key = {s.key: s for s in spots}
    sums = defaultdict(lambda: defaultdict(float))
    for pick in picks:
        spot = by_key.get((pick["slog"], pick["game"], pick["turn"]))
        if spot is None:
            continue
        kind = kind_of(spot, pick["pick"])
        for bucket in (bag_bucket(spot.bag), "all"):
            s = sums[bucket]
            s["positions"] += 1
            s[kind] += 1
            s["seconds"] += pick["seconds"]
            if kind != "unsimmed":
                base = spot.best_cut["win_prob"]
                s["simmed"] += 1
                s["gain"] += spot.candidates[pick["pick"]]["win_prob"] - base
                s["available"] += spot.best_outside["win_prob"] - base
    return {bucket: _means(s) for bucket, s in sums.items()}


def _means(s: dict[str, float]) -> dict[str, float]:
    n, simmed = s["positions"], max(s["simmed"], 1)
    return {
        "positions": n,
        **{kind: s[kind] / n for kind in KINDS},
        "gain": s["gain"] / simmed,
        "available": s["available"] / simmed,
        "seconds": s["seconds"] / n,
    }


def print_table(table: dict[str, dict[str, float]]):
    print(
        f"{'':9} {'positions':>9} {'best out':>9} {'other out':>9} {'cut':>7} {'unsimmed':>9} "
        f"{'gain':>7} {'of':>7} {'s/turn':>7}"
    )
    for bucket in sorted(table, key=lambda b: (b == "all", b)):
        t = table[bucket]
        print(
            f"{bucket:9} {int(t['positions']):9d} {t['best_outside']:9.3f} "
            f"{t['other_outside']:9.3f} {t['cut']:7.3f} {t['unsimmed']:9.3f} "
            f"{100 * t['gain']:+7.1f} {100 * t['available']:7.1f} {t['seconds']:7.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True, help="the hidden_blind_spots tag")
    add_mount_root_argument(parser)
    parser.add_argument("--player", help="the agent, as play_game's --player names it")
    parser.add_argument("--picks", type=Path, help="tabulate these picks instead of asking")
    parser.add_argument("--out", type=Path, help="keep the picks here (JSON lines)")
    parser.add_argument("--min-bag", type=int, default=1)
    parser.add_argument("--max-bag", type=int, default=100)
    parser.add_argument("--passes", action="store_true", help="keep the pass blind spots")
    parser.add_argument("--limit", type=int, default=0, help="a sample of this many (0: all)")
    parser.add_argument("--seed", type=int, default=0, help="the sample's seed")
    parser.add_argument("--threads", type=int, default=default_thread_count())
    args = parser.parse_args()
    if (args.player is None) == (args.picks is None):
        parser.error("give --player to ask an agent, or --picks to tabulate a run")
    found = TagPaths(args.tag, "hidden_blind_spots", args.mount_root).data_dir / "found"
    spots = select(read_spots(found), args)
    if not spots:
        raise SystemExit(f"{found}: no kept position matches")
    picks = read_picks(args.picks) if args.picks else run_tool(spots, args)
    table = tabulate(spots, picks)
    if not table:
        raise SystemExit("no pick is at one of the tag's positions")
    print_table(table)


if __name__ == "__main__":
    sys.exit(main())

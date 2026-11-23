#!/usr/bin/env python3
"""Play one release of a deep_bot tag against an opponent of your choosing
(docs/plans/deep_bot_generations.md, "The matches"), as the measure role
plays it against HastyBot and release 0: the release as the generating agent,
mirrored pairs from the tag's match seed, every game kept.

    python3 -m scripts.deep_bot.match --tag v1 --release 12 \\
        --opponent bestbot --player "--type=bestbot-endgame"

The match is delivered into the tag as a measure slot's is, so the running
dashboard records it on its next pass and the Matches tab draws a panel for
the opponent's name. With --out the two files go under that directory's data/
instead, to be copied into the tag's from a machine that only has the
release.

It fills the cores it is given, and the release's side sims on the GPU: run
it where both are free, or cap --threads.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

from cloud.sinks import LocalSink
from scribblez import params as params_mod
from scribblez import workloads
from scribblez.dashboard import tasks
from scribblez.deep_bot import LEAF, LEAVE, matches, release
from scribblez.hardware import default_thread_count
from scribblez.paths import add_mount_root_argument
from scribblez.workloads import deep_bot

# A name that reads as part of a file's stem and cannot pass for another's.
NAME = re.compile(r"[a-z0-9_]+")


def bound_player(argv: list[str]) -> list[str]:
    """`argv` with "--player <spec>" as one "--player=<spec>" token: argparse
    takes a one-word spec, which starts with "--type", for an option."""
    out = list(argv)
    for i in range(len(out) - 1):
        if out[i] == "--player":
            out[i : i + 2] = [f"--player={out[i + 1]}"]
            break
    return out


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", required=True, help="the deep_bot tag")
    p.add_argument("--release", type=int, required=True, help="the release to play")
    p.add_argument(
        "--opponent", required=True, help="the opponent's name on the chart, lowercase a-z 0-9 _"
    )
    p.add_argument("--player", required=True, help="the opponent's --player spec")
    p.add_argument("--games", type=int, help="games, even (default: the tag's match_games)")
    p.add_argument("--seed", type=int, help="first game seed (default: the tag's match_seed)")
    p.add_argument("--threads", type=int, default=default_thread_count())
    p.add_argument("--out", type=Path, help="deliver under this directory, not into the tag")
    add_mount_root_argument(p)
    args = p.parse_args(bound_player(sys.argv[1:] if argv is None else argv))
    if not NAME.fullmatch(args.opponent):
        p.error("--opponent must be lowercase letters, digits and underscores")
    if args.opponent in matches.OPPONENTS:
        p.error(f"'{args.opponent}' is the measure role's own; choose another name")
    if args.games is not None and (args.games < 2 or args.games % 2):
        p.error("--games must be even and >= 2")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = workloads.get("deep_bot")
    paths = spec.paths(args.tag, args.mount_root)
    stored = tasks.read_params(spec, args.tag, args.mount_root)
    if stored is None or not release.exists(paths, args.release):
        print(f"error: no release {args.release} of deep_bot tag '{args.tag}'", file=sys.stderr)
        return 1
    params = params_mod.validate(spec.params_cls, stored)
    agent = deep_bot.player_spec(
        params,
        release.export_path(paths, args.release, LEAF),
        release.export_path(paths, args.release, LEAVE),
    )
    with tempfile.TemporaryDirectory() as work:
        record, slog = matches.play(
            args.release,
            args.opponent,
            (agent, args.player),
            args.games or params.match_games,
            args.seed or params.match_seed,
            args.threads,
            Path(work),
            keep_games=True,
        )
        matches.deliver(LocalSink(args.out or paths.root), record, slog, Path(work))
    print(matches.describe(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run a head-to-head match between two `play_game` `--player` specs.

The C++ `play_game` binary plays two agents against each other and prints a
W/L/D summary from the first player's perspective. This module wraps that:
build a match, stream its progress live, and parse the final tally. Shared by
the standalone `scripts/match.py` CLI and the `scripts/pi_run.py` benchmark step
so the invocation and parsing live in exactly one place.
"""

import re
import subprocess
import sys

PLAY_GAME = "/workspace/repo/target/engine/play_game"

# play_game's end-of-run summary line: "... W/L/D vs <Opponent>: W / L / D".
WLD_RE = re.compile(r"vs \w+:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)")


def _run_streaming(cmd) -> tuple[int, str]:
    """Run `cmd`, echoing its combined output to stderr live while also capturing
    it, so play_game's periodic progress shows in real time and the full output
    is still available to parse. Returns (returncode, combined_output)."""
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen(cmd, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    chunks = []
    for line in proc.stdout:
        sys.stderr.write(line)
        sys.stderr.flush()
        chunks.append(line)
    proc.wait()
    return proc.returncode, "".join(chunks)


def play_match(spec_a: str, spec_b: str, games: int, threads: int):
    """Play `games` between two --player specs; return (wins, losses, draws) from
    the FIRST player's perspective, or None if the summary line can't be parsed."""
    rc, out = _run_streaming([PLAY_GAME, "--player", spec_a, "--player", spec_b,
                              "--games", str(games), "--threads", str(threads)])
    if rc != 0:
        raise SystemExit(f"command failed (exit {rc}): {PLAY_GAME}")
    m = WLD_RE.search(out)
    if not m:
        print("WARNING: could not parse W/L/D from play_game output", file=sys.stderr)
        return None
    return tuple(int(x) for x in m.groups())


def win_pct(wld) -> float:
    """Win percentage among DECISIVE games (draws excluded), for a (w, l, d) tuple."""
    w, l, _ = wld
    decisive = w + l
    return 100.0 * w / decisive if decisive else float("nan")

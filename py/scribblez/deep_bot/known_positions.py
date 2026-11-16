"""The known positions (positions/NWL23/known-positions, whose README
describes its manifest): asking an agent for its move at each, and reading
the pick against the position's answer.

The positions are GCGs, the game up to the move made there. gcg_to_slog turns
them into one .slog, and position_pick_tool replays each game to its last
move, as the agent would have lived it, and asks.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scribblez.paths import ENGINE_DIR, REPO_ROOT

DIRECTORY = REPO_ROOT / "positions" / "NWL23" / "known-positions"

# How a pick reads against its position's manifest entry.
REFERENCE = "reference"  # the move the analysis favours
EQUITY = "equity"  # HastyBot's move
OTHER = "other"


@dataclass(frozen=True)
class Position:
    """A manifest entry, and where its GCG is."""

    id: str
    title: str
    gcg: Path
    reference: tuple[str, ...]
    reference_equity_rank: int  # static equity's rank of it, HastyBot's move being 1
    static_equity_best: str

    @property
    def turn(self) -> int:
        """The 0-based turn of the move made at the position, the GCG's last."""
        return sum(line.startswith(">") for line in self.gcg.read_text().splitlines()) - 1

    def read(self, pick: str) -> str:
        """What `pick`, a move as the pick tool writes it, is here."""
        if pick in self.reference:
            return REFERENCE
        return EQUITY if pick == self.static_equity_best else OTHER


def load(directory: Path = DIRECTORY) -> list[Position]:
    """The manifest's positions, in its order."""
    entries = json.loads((directory / "positions.json").read_text())
    return [
        Position(
            id=e["id"],
            title=e["title"],
            gcg=directory / e["gcg"],
            reference=tuple(e["reference"]),
            reference_equity_rank=e["reference_equity_rank"],
            static_equity_best=e["static_equity_best"],
        )
        for e in entries
    ]


def files(directory: Path = DIRECTORY) -> list[Path]:
    """The directory's files a reader of the positions needs, the manifest
    last."""
    return [*(p.gcg for p in load(directory)), directory / "positions.json"]


def write_slog(positions: list[Position], work: Path) -> Path:
    """The positions' games as one .slog under `work`, game i the i-th
    position's. Raises subprocess.CalledProcessError if a GCG does not
    replay."""
    work.mkdir(parents=True, exist_ok=True)
    cmd = [str(ENGINE_DIR / "gcg_to_slog"), "--out-dir", str(work)]
    for p in positions:
        cmd += ["--gcg", str(p.gcg)]
    subprocess.run(cmd, check=True, capture_output=True)
    (slog,) = work.glob("*.slog")
    return slog


def ask(positions: list[Position], player: str, work: Path, threads: int) -> list[dict]:
    """`player`'s (a --player spec's) pick at each position: the pick tool's
    record of it (its "pick", "rack", "bag" and "seconds"), in order."""
    shutil.rmtree(work, ignore_errors=True)
    slog = write_slog(positions, work)
    listed = work / "positions.txt"
    listed.write_text("".join(f"{slog} {i} {p.turn}\n" for i, p in enumerate(positions)))
    out = work / "picks.jsonl"
    # A one-word spec starts with "--", so it is bound to its flag.
    subprocess.run(
        [
            str(ENGINE_DIR / "position_pick_tool"),
            "--positions", str(listed),
            "--out", str(out),
            f"--player={player}",
            "--threads", str(threads),
        ],
        check=True,
    )  # fmt: skip
    return [json.loads(line) for line in out.read_text().splitlines()]

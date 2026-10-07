"""M1a's exhibits (docs/plans/supreme_bot_m1a.md, PR 5): hand-built positions
where the right move blocks a threat that only the other moves' rollouts
reveal, and the test of whether a reader learns it.

An exhibit is a GCG whose final move is the decision, and the moves there that
kill the threat (its blockers). Building the exhibits:

    1. each GCG becomes a game of one .slog (engine gcg_to_slog);
    2. the decision's moves are listed by static equity (gcg_sim_evidence) and
       its candidates chosen: the blockers, then the best other moves, 16 in
       all (the reader's slots);
    3. transfer_test_generator probes and labels exactly those, with rollouts
       played to the end and endgames solved, as the near-endgame corpus is;
    4. the teacher's prior cache (scribblez/transfer_test/prior.py).

Scoring a reader: rows that hold out blockers and open moves together, with
every other candidate's probes in context, many times over. For the reader,
the prior and the labels, the gap between the held-out blockers' and the
held-out open moves' mean expected score. A reader that transfers what the
open moves' rollouts show moves its gap from the prior's toward the labels'.
The same rows with the blockers' own probes kept, and with every row's
outcomes shuffled among its probes, are the contrasts.
"""

from __future__ import annotations

import dataclasses
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scribblez.ffi import gcg_sim_evidence, position_eval_board_json
from scribblez.paths import ENGINE_DIR, REPO_ROOT
from scribblez.sim_evidence.sobs import MOVE_EXCHANGE, glyph_char
from scribblez.transfer_test.corpus import CorpusFile, load_file
from scribblez.transfer_test.evaluate import eval_row_config, reader_estimates, shuffled
from scribblez.transfer_test.prior import compute_prior, load_teacher, prior_path, write_prior
from scribblez.transfer_test.probes import read_sprobe
from scribblez.transfer_test.reader import Reader
from scribblez.transfer_test.rows import RowConfig, assemble_row

GCG_TO_SLOG = ENGINE_DIR / "gcg_to_slog"
GENERATOR = ENGINE_DIR / "transfer_test_generator"
CANDIDATES = 16
LISTED_MOVES = 60  # top moves by equity the candidates are chosen from
HELD_PER_GROUP = 2
ROWS_PER_EXHIBIT = 64
# The near-endgame corpora's teacher, so an exhibit's prior matches theirs.
TEACHER_TAG, TEACHER_GENERATION = "transformer-clipped", 2543


@dataclass(frozen=True)
class Exhibit:
    name: str
    gcg: str  # relative to the repo root
    # A move line to append when the file stops before the decision.
    decision_line: str | None
    # GCG notation (position and word, through-tiles as '.') of each blocker.
    blockers: tuple[str, ...]


EXHIBITS = (
    # The opponent's G hook at M7 forms GNU, with -ING words down column M;
    # column-N plays at rows 2-4 kill the lane.
    Exhibit(
        "pos-09",
        "positions/NWL23/position-eval-test-dataset/pos-09.gcg",
        None,
        ("N2 MICE", "N4 OI", "N3 MOI", "N4 MI", "N3 ICE", "N2 COME"),
    ),
    # GAVE opens row 13 to EGOTIZE; these plays close it.
    Exhibit(
        "egotize-lane",
        "positions/NWL23/face-up-trajectory-set/egotize-lane.gcg",
        ">Hasty_1: AEEGSTV E11 G.VE +22 462",
        ("H12 .EVA", "G12 .VE", "13G TEG", "13G TAV"),
    ),
)


def notation(move: np.void, board: list) -> str:
    """A move's GCG notation on the pre-move `board` (the board JSON's rows):
    position and word, through-tiles as '.', or '-TILES' for an exchange."""
    if move["type"] == MOVE_EXCHANGE:
        return "-" + "".join(glyph_char(int(g)) for g in move["glyphs"][: int(move["num_played"])])
    horizontal, lane = bool(move["horizontal"]), int(move["start"])
    placed, along, mask, i = {}, 0, int(move["square_mask"]), 0
    while mask:
        if mask & 1:
            placed[along] = glyph_char(int(move["glyphs"][i]))
            i += 1
        mask >>= 1
        along += 1

    def occupied(a: int) -> bool:
        return bool(board[lane][a] if horizontal else board[a][lane])

    lo, hi = min(placed), max(placed)
    while lo > 0 and occupied(lo - 1):
        lo -= 1
    while hi < 14 and occupied(hi + 1):
        hi += 1
    word = "".join(placed.get(a, ".") for a in range(lo, hi + 1))
    where = f"{lane + 1}{chr(65 + lo)}" if horizontal else f"{chr(65 + lane)}{lo + 1}"
    return f"{where} {word}"


def decision_text(e: Exhibit) -> str:
    text = (REPO_ROOT / e.gcg).read_text().rstrip("\n") + "\n"
    return text + e.decision_line + "\n" if e.decision_line else text


def choose_candidates(e: Exhibit, text: str) -> tuple[np.ndarray, np.ndarray]:
    """(moves, is_blocker): every blocker, then the best other moves by
    static equity, CANDIDATES in all. Raises if a blocker is not among the
    LISTED_MOVES."""
    head = "\n".join(text.rstrip("\n").splitlines()[:-1])
    board = position_eval_board_json(head)["board"]
    records, _ = gcg_sim_evidence(text, top_k=LISTED_MOVES, rollouts=1, open_leaves=True)
    names = [notation(r["move"], board) for r in records]
    missing = set(e.blockers) - set(names)
    if missing:
        raise ValueError(f"{e.name}: blockers not among the top {LISTED_MOVES}: {sorted(missing)}")
    blocker = np.array([n in e.blockers for n in names])
    chosen = list(np.flatnonzero(blocker))
    chosen += [i for i in np.flatnonzero(~blocker)][: CANDIDATES - len(chosen)]
    chosen.sort()
    return records["move"][chosen], blocker[chosen]


def build(
    out: Path, mount_root: Path, probes: int = 125, label_rollouts: int = 1000, threads: int = 0
):
    """Write the exhibits' .slog, probe and label sidecars, prior cache and
    blocker lists under `out`. Loads the teacher first: it sets the engine
    session's input arm, which must precede any other engine call."""
    model = load_teacher(TEACHER_TAG, TEACHER_GENERATION, mount_root)
    out.mkdir(parents=True, exist_ok=True)
    gcgs, chosen_lines, blockers = [], [], {}
    for g, e in enumerate(EXHIBITS):
        text = decision_text(e)
        path = out / f"{e.name}.gcg"
        path.write_text(text)
        gcgs.append(path)
        moves, is_blocker = choose_candidates(e, text)
        turn = sum(1 for line in text.splitlines() if line.startswith(">")) - 1
        chosen_lines += [f"{g} {turn} {m.tobytes().hex()}" for m in moves]
        blockers[e.name] = is_blocker
    (out / "chosen.txt").write_text("\n".join(chosen_lines) + "\n")
    np.savez(out / "blockers.npz", **blockers)
    for old in out.glob("*.slog"):
        old.unlink()
    subprocess.run(
        [str(GCG_TO_SLOG), "--out-dir", str(out), "--face-up", *(f"--gcg={g}" for g in gcgs)],
        check=True,
    )
    (slog,) = out.glob("*.slog")
    subprocess.run(
        [
            str(GENERATOR),
            "--mode=corpus",
            f"--slog-file={slog}",
            f"--chosen-moves={out / 'chosen.txt'}",
            "--horizon=0",
            "--solve-max-unseen=100",
            f"--probes={probes}",
            f"--label-rollouts={label_rollouts}",
            *([f"--threads={threads}"] if threads else []),
        ],
        check=True,
    )
    sprobe = slog.with_suffix(".sprobe")
    device = torch.device("cuda")
    write_prior(compute_prior(model.to(device), read_sprobe(sprobe), device), prior_path(sprobe))


def load(out: Path) -> tuple[CorpusFile, dict[str, np.ndarray]]:
    (sprobe,) = out.glob("*.sprobe")
    with np.load(out / "blockers.npz") as z:
        blockers = {k: z[k] for k in z.files}
    return load_file(sprobe), blockers


@dataclass
class GapRows:
    """One exhibit's rows and, per row, which candidates are held-out blockers
    and held-out open moves."""

    rows: list
    blocker: list[np.ndarray]
    open: list[np.ndarray]


def gap_rows(
    f: CorpusFile, p: int, is_blocker: np.ndarray, cfg: RowConfig, rng, keep_blockers: bool
) -> GapRows:
    """ROWS_PER_EXHIBIT rows holding out HELD_PER_GROUP open moves, and as many
    blockers unless `keep_blockers` (then the blockers' probes stay in context
    and they are read, not transferred to)."""
    out = GapRows([], [], [])
    blockers, opens = np.flatnonzero(is_blocker), np.flatnonzero(~is_blocker)
    for _ in range(ROWS_PER_EXHIBIT):
        held = np.zeros(len(is_blocker), dtype=bool)
        held[rng.choice(opens, HELD_PER_GROUP, replace=False)] = True
        b = rng.choice(blockers, HELD_PER_GROUP, replace=False)
        if not keep_blockers:
            held[b] = True
        out.rows.append(assemble_row(f, 0, p, cfg, rng, held=held))
        out.blocker.append(np.isin(np.arange(len(is_blocker)), b))
        out.open.append(held & ~is_blocker)
    return out


def _gap(estimates: list[np.ndarray], g: GapRows) -> float:
    return float(
        np.mean(
            [
                e[b].mean() - e[o].mean()
                for e, b, o in zip(estimates, g.blocker, g.open, strict=True)
            ]
        )
    )


def score(
    reader: Reader, params: dict, f: CorpusFile, blockers: dict, device, seed: int = 0
) -> dict:
    """Per exhibit: the blocker-minus-open gap in expected score for the
    labels, the prior and the reader, with the blockers held out, with their
    probes kept, and with outcomes shuffled."""
    cfg = dataclasses.replace(eval_row_config(params), max_held_out=2 * HELD_PER_GROUP)
    rng = np.random.default_rng(seed)
    result = {}
    for p, e in enumerate(EXHIBITS):
        c0, c1 = f.probes.candidate_start[p], f.probes.candidate_start[p + 1]
        lab = f.labels[c0:c1]
        label = (lab["wins"] + 0.5 * lab["draws"]) / lab["n"]
        prior = f.prior.wld[c0:c1, 0] + 0.5 * f.prior.wld[c0:c1, 1]
        is_blocker = blockers[e.name]
        held = gap_rows(f, p, is_blocker, cfg, rng, keep_blockers=False)
        kept = gap_rows(f, p, is_blocker, cfg, rng, keep_blockers=True)
        n = len(held.rows)
        result[e.name] = {
            "label_gap": _gap([label] * n, held),
            "prior_gap": _gap([prior] * n, held),
            "reader_gap_held_out": _gap(reader_estimates(reader, held.rows, device), held),
            "reader_gap_shuffled": _gap(
                reader_estimates(reader, [shuffled(r, rng) for r in held.rows], device), held
            ),
            "reader_gap_probed": _gap(reader_estimates(reader, kept.rows, device), kept),
        }
    return result

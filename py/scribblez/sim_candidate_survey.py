"""What a static-equity candidate cut costs, read off sim_candidate_survey_tool's rows.

The measurement behind docs/plans/sim_labeled_candidates.md: at each surveyed
position a stratified candidate sample was simmed, and the question is how
often, and by how much, the sim prefers a candidate outside the head of the
HastyBot equity ranking (the top `cut`), which a top-`cut` self-play mover can
never play.

The best of ~64 noisy sim estimates flatters itself, and most candidates lie
outside the cut, so "the sim's best is outside the cut" is true far more often
than the cut costs anything. Every figure here is therefore held out: picks
are made on one replica's rollouts and valued on another's (same candidates,
independent rollouts), over both assignments of the two roles.

The tail is sampled, not swept, so a position's best tail move is usually not
among its candidates: the cost reported is a lower bound on the cut's true
cost.
"""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SURVEY_SUFFIX = ".simsurvey.csv"

# Equity-rank buckets a pick is tallied under; ranks are 0-based.
RANK_BUCKETS = ((0, 1), (1, 10), (10, 32), (32, 100), (100, math.inf))


@dataclass(frozen=True)
class Candidate:
    equity_rank: int  # -1: a played move the generator never ranks
    is_play: bool
    win_equity: float  # (wins + draws / 2) / rollouts
    mean_delta: float


# position key (file stem, game, turn) -> replica -> candidates in stored order
Survey = dict[tuple[str, int, int], dict[int, list[Candidate]]]


def load_survey(paths: list[Path]) -> Survey:
    survey: Survey = defaultdict(lambda: defaultdict(list))
    for path in paths:
        stem = path.name.removesuffix(SURVEY_SUFFIX)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                n = float(row["rollouts"])
                survey[(stem, int(row["game"]), int(row["turn"]))][int(row["replica"])].append(
                    Candidate(
                        equity_rank=int(row["equity_rank"]),
                        is_play=row["is_play"] == "1",
                        win_equity=(float(row["wins"]) + 0.5 * float(row["draws"])) / n,
                        mean_delta=float(row["delta_sum"]) / n,
                    )
                )
    return survey


def inside_cut(c: Candidate, cut: int) -> bool:
    return 0 <= c.equity_rank < cut


def best_index(candidates: list[Candidate], eligible: list[int]) -> int:
    """The eligible index with the highest win equity, ties to the mean delta then
    the stored order."""
    return max(eligible, key=lambda i: (candidates[i].win_equity, candidates[i].mean_delta, -i))


@dataclass(frozen=True)
class HeldOutPick:
    """One (position, replica assignment): the picks made on the selecting replica,
    valued on the other."""

    pick_rank: int  # equity rank of the unrestricted pick
    pick_is_play: bool
    gain: float  # held-out win equity: unrestricted pick minus the pick inside the cut

    def outside(self, cut: int) -> bool:
        return not 0 <= self.pick_rank < cut


def held_out_picks(survey: Survey, cut: int) -> list[HeldOutPick]:
    picks = []
    for replicas in survey.values():
        for select, value in ((0, 1), (1, 0)):
            chosen, scored = replicas[select], replicas[value]
            inside = [i for i, c in enumerate(chosen) if inside_cut(c, cut)]
            if not inside:
                continue
            free = best_index(chosen, list(range(len(chosen))))
            cut_pick = best_index(chosen, inside)
            picks.append(
                HeldOutPick(
                    pick_rank=chosen[free].equity_rank,
                    pick_is_play=chosen[free].is_play,
                    gain=scored[free].win_equity - scored[cut_pick].win_equity,
                )
            )
    return picks


def mean_and_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, math.nan
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def bucket_label(lo: float, hi: float) -> str:
    if hi == math.inf:
        return f"rank {lo}+"
    return f"rank {lo}" if hi == lo + 1 else f"ranks {lo}-{int(hi) - 1}"


def report(survey: Survey, cut: int) -> str:
    """The survey's findings as text. Gains are win-equity points (percent)."""
    if any(set(replicas) != {0, 1} for replicas in survey.values()):
        raise ValueError("the held-out analysis needs exactly two replicas per position")
    picks = held_out_picks(survey, cut)
    outside = [p for p in picks if p.outside(cut)]
    lines = [
        f"{len(survey)} positions, {len(picks)} held-out picks (2 per position), cut = top {cut}",
        "",
        f"sim pick outside the cut: {len(outside) / len(picks):.1%} of picks",
    ]
    if outside:
        confirmed = sum(p.gain > 0 for p in outside)
        mean, se = mean_and_se([100 * p.gain for p in outside])
        lines += [
            f"  of those, held-out replica agrees (gain > 0): {confirmed / len(outside):.1%}",
            f"  their mean held-out gain: {mean:+.2f} +/- {se:.2f} pts",
        ]
    mean, se = mean_and_se([100 * p.gain for p in picks])
    lines += [
        f"cost of the cut, per position (held-out gain of lifting it): "
        f"{mean:+.2f} +/- {se:.2f} pts",
        "",
        "where the sim's picks sit in the equity ranking (share of picks; mean held-out gain):",
    ]
    rows = []
    for lo, hi in RANK_BUCKETS:
        members = [p for p in picks if p.pick_is_play and lo <= p.pick_rank < hi]
        rows.append((bucket_label(lo, hi), members))
    rows.append(("exchange", [p for p in picks if not p.pick_is_play]))
    for label, members in rows:
        if not members:
            lines.append(f"  {label:<12} {0:6.1%}")
            continue
        mean, _ = mean_and_se([100 * p.gain for p in members])
        lines.append(f"  {label:<12} {len(members) / len(picks):6.1%}   {mean:+.2f} pts")
    return "\n".join(lines)

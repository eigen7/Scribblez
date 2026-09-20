"""What a static-equity candidate cut costs, read off sim_candidate_survey_tool's files.

The measurement behind docs/plans/sim_labeled_candidates.md: at each surveyed
position a set of candidates was simmed, and the question is how often, and by
how much, the sim prefers a candidate outside the head of the HastyBot equity
ranking (the top `cut`), which a top-`cut` self-play mover can never play.

The best of hundreds of noisy sim estimates flatters itself, and most
candidates lie outside the cut, so "the sim's best is outside the cut" is true
far more often than the cut costs anything. Every figure here is therefore
held out: picks are made on one replica's rollouts and valued on another's
(same candidates, independent rollouts), over both assignments of the two
roles. A pick outside the cut "beats the cut" only when the valuing replica
puts it at least MIN_SIGMA standard errors above the cut's own pick -- the
standard error being that of the paired win difference, which common random
numbers make much tighter than the two marginal errors.
"""

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

SURVEY_SUFFIX = ".simsurvey.json"
MIN_SIGMA = 2.0

# Equity-rank buckets a pick is tallied under; ranks are 0-based.
RANK_BUCKETS = ((0, 1), (1, 10), (10, 32), (32, 100), (100, math.inf))

PositionKey = tuple[str, int, int]  # (file stem, game, 0-based turn)


@dataclass(frozen=True)
class Candidate:
    move: str  # GCG notation
    equity_rank: int  # -1: a played move the generator never ranks
    is_play: bool
    is_setup: bool  # a high-value setup play
    replicas: tuple[dict, ...]  # the tool's per-replica rollout summaries

    def win_equity(self, replica: int) -> float:
        s = self.replicas[replica]
        return (s["wins"] + 0.5 * s["draws"]) / s["n"]

    def spread(self, replica: int) -> float:
        s = self.replicas[replica]
        return s["delta_sum"] / s["n"]

    def win_se(self, replica: int) -> float:
        """Standard error of win_equity from its own rollouts (draws as 1/2)."""
        s = self.replicas[replica]
        mean = self.win_equity(replica)
        second = (s["wins"] + 0.25 * s["draws"]) / s["n"]
        return math.sqrt(max(second - mean * mean, 0.0) / s["n"])


@dataclass(frozen=True)
class Position:
    key: PositionKey
    candidates: tuple[Candidate, ...]


def load_survey(paths: list[Path]) -> list[Position]:
    positions = []
    for path in paths:
        stem = path.name.removesuffix(SURVEY_SUFFIX)
        for p in json.loads(path.read_text())["positions"]:
            candidates = tuple(
                Candidate(
                    move=c["move"],
                    equity_rank=c["equity_rank"],
                    is_play=c["is_play"],
                    is_setup=c["is_setup"],
                    replicas=tuple(c["replicas"]),
                )
                for c in p["candidates"]
            )
            positions.append(Position((stem, p["game"], p["turn"]), candidates))
    return positions


def inside_cut(c: Candidate, cut: int) -> bool:
    return 0 <= c.equity_rank < cut


def best_index(candidates: tuple[Candidate, ...], eligible: list[int], replica: int) -> int:
    """The eligible index with the highest win equity on `replica`, ties to the
    spread then the stored order."""
    return max(
        eligible,
        key=lambda i: (candidates[i].win_equity(replica), candidates[i].spread(replica), -i),
    )


def win_gain_and_se(pick: Candidate, cut_pick: Candidate, cut_index: int, replica: int):
    """pick's win equity minus cut_pick's on `replica`, with its standard error:
    the paired one when the tool recorded it (cut_pick is then candidate
    `cut_index`, the tool listing the cut first), else the marginal errors'."""
    gain = pick.win_equity(replica) - cut_pick.win_equity(replica)
    paired = pick.replicas[replica]["win_diff_vs_cut"]
    if cut_index >= len(paired):
        return gain, math.hypot(pick.win_se(replica), cut_pick.win_se(replica))
    total, sq_total = paired[cut_index]
    n = pick.replicas[replica]["n"]
    return gain, math.sqrt(max(sq_total / n - (total / n) ** 2, 0.0) / n)


@dataclass(frozen=True)
class HeldOutPick:
    """One (position, replica assignment): the best candidate outside the cut and
    the best inside it, both chosen on the selecting replica and valued on the
    other."""

    outside: Candidate
    inside: Candidate
    free_pick_outside: bool  # the unrestricted best was the outside candidate
    gain: float  # held-out win equity, outside minus inside
    gain_se: float
    spread_gain: float

    @property
    def sigmas(self) -> float:
        if self.gain_se > 0:
            return self.gain / self.gain_se
        return math.copysign(math.inf, self.gain) if self.gain else 0.0

    @property
    def beats_cut(self) -> bool:
        """The selecting replica preferred the outside candidate to everything in
        the cut, and the valuing replica put it MIN_SIGMA above the cut's pick."""
        return self.free_pick_outside and self.sigmas >= MIN_SIGMA


@dataclass(frozen=True)
class Finding:
    """A position's two held-out picks (one per assignment of the replicas)."""

    key: PositionKey
    picks: tuple[HeldOutPick, HeldOutPick]

    @property
    def mean_gain(self) -> float:
        return sum(p.gain for p in self.picks) / 2

    @property
    def mean_spread_gain(self) -> float:
        return sum(p.spread_gain for p in self.picks) / 2

    @property
    def min_sigmas(self) -> float:
        return min(p.sigmas for p in self.picks)

    @property
    def beats_cut_once(self) -> bool:
        return any(p.beats_cut for p in self.picks)

    @property
    def beats_cut_twice(self) -> bool:
        """Both replicas chose the same outside move and each valued the other's
        choice MIN_SIGMA above the cut's."""
        same_move = self.picks[0].outside.move == self.picks[1].outside.move
        return same_move and all(p.beats_cut for p in self.picks)


def held_out_pick(position: Position, cut: int, setups_only: bool, select: int, value: int):
    cands = position.candidates
    inside = [i for i, c in enumerate(cands) if inside_cut(c, cut)]
    outside = [
        i for i, c in enumerate(cands) if not inside_cut(c, cut) and (c.is_setup or not setups_only)
    ]
    if not inside or not outside:
        return None
    i, o = best_index(cands, inside, select), best_index(cands, outside, select)
    gain, se = win_gain_and_se(cands[o], cands[i], i, value)
    return HeldOutPick(
        outside=cands[o],
        inside=cands[i],
        free_pick_outside=cands[o].win_equity(select) > cands[i].win_equity(select),
        gain=gain,
        gain_se=se,
        spread_gain=cands[o].spread(value) - cands[i].spread(value),
    )


def findings(survey: list[Position], cut: int, setups_only: bool = False) -> list[Finding]:
    """Every position with candidates on both sides of the cut, strongest first
    (by the weaker of its two held-out sigmas)."""
    out = []
    for position in survey:
        if any(len(c.replicas) != 2 for c in position.candidates):
            raise ValueError("the held-out analysis needs exactly two replicas per position")
        picks = (
            held_out_pick(position, cut, setups_only, 0, 1),
            held_out_pick(position, cut, setups_only, 1, 0),
        )
        if picks[0] and picks[1]:
            out.append(Finding(position.key, picks))
    return sorted(out, key=lambda f: -f.min_sigmas)


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


def finding_line(f: Finding) -> str:
    a, b = f.picks
    moves = a.outside.move
    if b.outside.move != moves:
        moves += f" / {b.outside.move}"
    return (
        f"{gcg_name(f.key)}: {moves} (#{a.outside.equity_rank + 1}) over {a.inside.move}, "
        f"win {100 * a.gain:+.1f}/{100 * b.gain:+.1f} ({a.sigmas:+.1f}/{b.sigmas:+.1f} sigma), "
        f"spread {a.spread_gain:+.1f}/{b.spread_gain:+.1f}"
    )


def report(survey: list[Position], cut: int, setups_only: bool = False) -> str:
    """The survey's findings as text. Win gains are win-equity points (percent)."""
    found = findings(survey, cut, setups_only)
    picks = [p for f in found for p in f.picks]
    free = [p for p in picks if p.free_pick_outside]
    cost, cost_se = mean_and_se([100 * p.gain if p.free_pick_outside else 0.0 for p in picks])
    what = "high-value setup play" if setups_only else "candidate"
    lines = [
        f"{len(found)} positions with a {what} outside the top {cut}; "
        f"{len(picks)} held-out picks (2 per position)",
        "",
        f"sim's best lies outside the cut: {len(free) / len(picks):.1%} of picks",
        f"cost of the cut, per position (held-out win gain of lifting it): "
        f"{cost:+.2f} +/- {cost_se:.2f} pts",
    ]
    once = [f for f in found if f.beats_cut_once]
    twice = [f for f in found if f.beats_cut_twice]
    lines += [
        f"outside pick beats the cut's by >= {MIN_SIGMA:g} sigma on the held-out replica: "
        f"{len(once)} positions on at least one assignment ({len(once) / len(found):.1%}), "
        f"{len(twice)} on both with the same move ({len(twice) / len(found):.1%})",
        "",
        "where the confirmed-once outside picks sit in the equity ranking:",
    ]
    winners = [p for f in once for p in f.picks if p.beats_cut]
    for lo, hi in RANK_BUCKETS[2:]:
        members = [p for p in winners if lo <= p.outside.equity_rank < hi]
        lines.append(f"  {bucket_label(lo, hi):<12} {len(members)}")
    lines += ["", "positions that beat the cut at least once, strongest first:"]
    lines += [f"  {finding_line(f)}" for f in once]
    return "\n".join(lines)


def gcg_name(key: PositionKey) -> str:
    """The file sim_candidate_survey_tool --gcg-dir wrote for a position (its turn
    is 1-based there, as neural_rank_tool --turn takes it)."""
    stem, game, turn = key
    return f"{stem}-g{game}-turn{turn + 1}.gcg"


def pooled(c: Candidate) -> tuple[float, float]:
    """A candidate's (win equity in percent, spread) over both replicas' rollouts."""
    return 50 * (c.win_equity(0) + c.win_equity(1)), (c.spread(0) + c.spread(1)) / 2


def write_review_dir(
    survey: list[Position],
    cut: int,
    setups_only: bool,
    gcg_dir: Path,
    review_dir: Path,
    command: str,
):
    """Replace `review_dir` with the GCGs of the positions whose outside pick beat
    the cut at least once and a README table of what the sims said about each.
    `command` is the invocation that produced them, recorded for regeneration."""
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True)
    what = "high-value setup play" if setups_only else "play"
    lines = [
        "# Sim survey examples",
        "",
        f"Positions from HastyBot self-play where a {what} ranked outside the hasty top {cut}",
        f"beat the best top-{cut} move: chosen on one sim replica, it sat at least {MIN_SIGMA:g}",
        "standard errors above the cut's pick on the other, independent one. Strongest first;",
        "`twice` marks positions where both replicas chose the same move and both confirmed",
        "it. Each GCG ends on the move the game actually played; `neural_rank_tool --gcg",
        "<file> --turn <turn>` opens the decision point. Win% and spread (mean final score",
        "differential, mover's view) pool both replicas' rollouts; the gains and sigmas are",
        "held out, one figure per replica assignment.",
        "",
        "Regenerate this directory (games, sims and all; `--slog-dir` is scratch space) with:",
        "",
        "```",
        command,
        "```",
        "",
        "| file | turn | outside play (hasty rank) | win% | spread | best top-10 move | win% "
        "| spread | win gains | sigmas | spread gains | twice |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in findings(survey, cut, setups_only):
        if not f.beats_cut_once:
            continue
        shutil.copy(gcg_dir / gcg_name(f.key), review_dir / gcg_name(f.key))
        a, b = f.picks
        best = a if a.beats_cut and (a.sigmas >= b.sigmas or not b.beats_cut) else b
        (out_win, out_spread), (in_win, in_spread) = pooled(best.outside), pooled(best.inside)
        lines.append(
            f"| {gcg_name(f.key)} | {f.key[2] + 1} | {best.outside.move} "
            f"(#{best.outside.equity_rank + 1}) | {out_win:.1f} | {out_spread:+.1f} "
            f"| {best.inside.move} | {in_win:.1f} | {in_spread:+.1f} "
            f"| {100 * a.gain:+.1f}, {100 * b.gain:+.1f} | {a.sigmas:+.1f}, {b.sigmas:+.1f} "
            f"| {a.spread_gain:+.1f}, {b.spread_gain:+.1f} | {'yes' if f.beats_cut_twice else ''} |"
        )
    (review_dir / "README.md").write_text("\n".join(lines) + "\n")

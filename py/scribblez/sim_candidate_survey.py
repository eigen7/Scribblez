"""What a static-equity candidate cut costs, read off sim_candidate_survey_tool's files.

The measurement behind docs/plans/sim_labeled_candidates.md: at each surveyed
position a set of candidates was simmed, and the question is how often, and by
how much, the sim prefers a candidate outside the head of the HastyBot equity
ranking (the top `cut`), which a top-`cut` self-play mover can never play.

The best of hundreds of noisy sim estimates flatters itself, and most
candidates lie outside the cut, so "the screen's best is outside the cut" is
true far more often than the cut costs anything. The tool therefore sims in
two stages, and every figure here reads the second: the screen singles out
its few best moves outside the cut (racing away the clearly beaten), and a
longer confirming sim on fresh rollouts re-sims those moves beside the cut's.
An outside move "beats the cut" when
the confirming sim puts it at least MIN_SIGMA standard errors above the cut's
best move -- the standard error being that of the paired win difference,
which common random numbers make much tighter than the two marginal errors.
The cut's best is taken on the confirming sim itself; a best-of-ten flatters
the cut a little, so the bar errs on the strict side. Against that, each
position tests a few outside moves, not one, so a few percent of positions
clear the bar by chance alone; read single positions near the bar with that
in mind.
"""

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

SURVEY_SUFFIX = ".simsurvey.json"
MIN_SIGMA = 2.0

# Equity-rank buckets an outside move is tallied under; ranks are 0-based.
RANK_BUCKETS = ((10, 32), (32, 100), (100, math.inf))

PositionKey = tuple[str, int, int]  # (file stem, game, 0-based turn)


def win_equity(summary: dict) -> float:
    return (summary["wins"] + 0.5 * summary["draws"]) / summary["n"]


def spread(summary: dict) -> float:
    return summary["delta_sum"] / summary["n"]


@dataclass(frozen=True)
class ConfirmedMove:
    """One move of a position's confirming sim."""

    move: str  # GCG notation
    equity_rank: int
    is_setup: bool
    summary: dict  # the tool's rollout summary, confirming stage
    win_diff_vs_cut: list[list[float]]  # [sum, sum of squares] against each cut move

    @property
    def win_equity(self) -> float:
        return win_equity(self.summary)

    @property
    def spread(self) -> float:
        return spread(self.summary)


@dataclass(frozen=True)
class Finding:
    """One of the screen's picks from outside the cut, against the cut's best move,
    both as the position's confirming sim read them."""

    key: PositionKey
    outside: ConfirmedMove
    inside: ConfirmedMove
    gain: float  # win equity, outside minus inside
    gain_se: float  # of the paired difference

    @property
    def sigmas(self) -> float:
        if self.gain_se > 0:
            return self.gain / self.gain_se
        return math.copysign(math.inf, self.gain) if self.gain else 0.0

    @property
    def spread_gain(self) -> float:
        return self.outside.spread - self.inside.spread

    @property
    def beats_cut(self) -> bool:
        return self.sigmas >= MIN_SIGMA


def confirmed_moves(position: dict) -> list[ConfirmedMove]:
    moves = []
    for entry in position["confirm"]:
        c = position["candidates"][entry["candidate"]]
        moves.append(
            ConfirmedMove(
                move=c["move"],
                equity_rank=c["equity_rank"],
                is_setup=c["is_setup"],
                summary=entry["summary"],
                win_diff_vs_cut=entry["win_diff_vs_cut"],
            )
        )
    return moves


def position_findings(key: PositionKey, position: dict, cut: int) -> list[Finding]:
    """A finding per outside move of the position's confirming sim (none when the
    screen left nothing to confirm)."""
    moves = confirmed_moves(position)
    cut_moves = [m for m in moves if 0 <= m.equity_rank < cut]  # the tool lists them first
    if not cut_moves:
        return []
    best = max(range(len(cut_moves)), key=lambda i: (cut_moves[i].win_equity, -i))
    out = []
    for outside in moves[len(cut_moves) :]:
        total, sq_total = outside.win_diff_vs_cut[best]
        n = outside.summary["n"]
        mean = total / n
        se = math.sqrt(max(sq_total / n - mean * mean, 0.0) / n)
        out.append(Finding(key, outside, cut_moves[best], mean, se))
    return out


@dataclass(frozen=True)
class Survey:
    positions: int  # surveyed
    findings: list[Finding]  # every confirmed outside move, strongest first

    @property
    def winners(self) -> list[Finding]:
        return [f for f in self.findings if f.beats_cut]


def load_survey(paths: list[Path]) -> Survey:
    positions, found = 0, []
    for path in paths:
        stem = path.name.removesuffix(SURVEY_SUFFIX)
        survey = json.loads(path.read_text())
        for position in survey["positions"]:
            positions += 1
            key = (stem, position["game"], position["turn"])
            found += position_findings(key, position, survey["cut"])
    return Survey(positions, sorted(found, key=lambda f: -f.sigmas))


def mean_and_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, math.nan
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def bucket_label(lo: float, hi: float) -> str:
    return f"rank {lo}+" if hi == math.inf else f"ranks {lo}-{int(hi) - 1}"


def finding_line(f: Finding) -> str:
    return (
        f"{gcg_name(f.key)}: {f.outside.move} (#{f.outside.equity_rank + 1}) over "
        f"{f.inside.move}, win {100 * f.gain:+.1f} ({f.sigmas:+.1f} sigma), "
        f"spread {f.spread_gain:+.1f}"
    )


def report(survey: Survey) -> str:
    """The survey's findings as text. Win gains are win-equity points (percent)."""
    winners = survey.winners
    winning_positions = {f.key for f in winners}
    contested = {f.key for f in survey.findings}
    best_gain = {}
    for f in winners:
        best_gain[f.key] = max(best_gain.get(f.key, 0.0), 100 * f.gain)
    no_gain = [0.0] * (survey.positions - len(best_gain))
    cost, cost_se = mean_and_se([*best_gain.values(), *no_gain])
    lines = [
        f"{survey.positions} positions surveyed; at {len(contested)} the screen left an outside "
        f"move standing, {len(survey.findings)} such moves confirmed in all",
        f"outside move beats the cut by >= {MIN_SIGMA:g} sigma: {len(winners)} moves at "
        f"{len(winning_positions)} positions ({len(winning_positions) / survey.positions:.1%})",
        f"what playing the best of them would gain, per surveyed position: "
        f"{cost:+.2f} +/- {cost_se:.2f} win pts",
        "",
        "where they sit in the equity ranking:",
    ]
    for lo, hi in RANK_BUCKETS:
        members = [f for f in winners if lo <= f.outside.equity_rank < hi]
        lines.append(f"  {bucket_label(lo, hi):<12} {len(members)}")
    lines += ["", "strongest first:"]
    lines += [f"  {finding_line(f)}" for f in winners]
    return "\n".join(lines)


def gcg_name(key: PositionKey) -> str:
    """The file sim_candidate_survey_tool --gcg-dir wrote for a position (its turn
    is 1-based there, as neural_rank_tool --turn takes it)."""
    stem, game, turn = key
    return f"{stem}-g{game}-turn{turn + 1}.gcg"


def write_review_dir(survey: Survey, cut: int, gcg_dir: Path, review_dir: Path, command: str):
    """Replace `review_dir` with the GCGs of the positions where an outside move beat
    the cut and a README table of what the confirming sim said about each.
    `command` is the invocation that produced them, recorded for regeneration."""
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True)
    lines = [
        "# Sim survey examples",
        "",
        f"Positions from HastyBot self-play where a play ranked outside the hasty top {cut} beat",
        f"the best top-{cut} move. A screening sim of every candidate singled out its few best",
        "outside plays; a longer confirming sim on fresh rollouts, of those and the top moves, put",
        f"it at least {MIN_SIGMA:g} standard errors (of the paired win difference) above the best",
        "of them. Strongest first. Each GCG ends on the move the game actually played;",
        "`neural_rank_tool --gcg <file> --turn <turn>` opens the decision point. Win% and spread",
        "(mean final score differential, mover's view) are the confirming sim's.",
        "",
        "Regenerate this directory (games, sims and all; `--slog-dir` is scratch space) with:",
        "",
        "```",
        command,
        "```",
        "",
        "| file | turn | outside play (hasty rank) | win% | spread | best top move | win% "
        "| spread | win gain | sigmas | spread gain |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in survey.winners:
        shutil.copy(gcg_dir / gcg_name(f.key), review_dir / gcg_name(f.key))
        lines.append(
            f"| {gcg_name(f.key)} | {f.key[2] + 1} | {f.outside.move} "
            f"(#{f.outside.equity_rank + 1}) | {100 * f.outside.win_equity:.1f} "
            f"| {f.outside.spread:+.1f} | {f.inside.move} | {100 * f.inside.win_equity:.1f} "
            f"| {f.inside.spread:+.1f} | {100 * f.gain:+.1f} | {f.sigmas:+.1f} "
            f"| {f.spread_gain:+.1f} |"
        )
    (review_dir / "README.md").write_text("\n".join(lines) + "\n")

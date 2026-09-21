"""The analysis of sim_candidate_survey_tool's .simsurvey.json files."""

import json
from pathlib import Path

import pytest
from scribblez.sim_candidate_survey import gcg_name, load_survey, report, write_review_dir

ROLLOUTS = 100


def summary(wins: int) -> dict:
    return {
        "n": ROLLOUTS,
        "wins": wins,
        "draws": 0,
        "losses": ROLLOUTS - wins,
        "delta_sum": 10.0 * wins,
    }


def paired(wins: int, ref_wins: int) -> list[float]:
    """Paired moments against a cut move this one's rollouts disagree with as
    little as the two win counts allow (the tightest CRN can make them)."""
    return [float(wins - ref_wins), float(abs(wins - ref_wins))]


def position(cut_wins: list[int], outside: dict[int, int]) -> dict:
    """A position whose confirming sim covers a cut of len(cut_wins) moves and the
    `outside` moves (equity rank -> wins), stored as the candidates after them."""
    ranks = [*range(len(cut_wins)), *outside]
    wins = [*cut_wins, *outside.values()]
    candidates = [
        {
            "move": f"M{rank}",
            "display": f"M({rank})",
            "equity_rank": rank,
            "is_setup": False,
            "screen": summary(50),
        }
        for rank in ranks
    ]
    confirm = [
        {
            "candidate": i,
            "summary": summary(w),
            "win_diff_vs_cut": [paired(w, ref) for ref in cut_wins],
        }
        for i, w in enumerate(wins)
    ]
    return {"game": 0, "turn": 3, "candidates": candidates, "confirm": confirm}


def write_survey(tmp_path: Path, positions: list[dict]) -> list[Path]:
    path = tmp_path / "chunk.simsurvey.json"
    path.write_text(json.dumps({"version": 2, "cut": 10, "positions": positions}))
    return [path]


def test_the_outside_move_is_measured_against_the_cuts_best(tmp_path):
    (f,) = load_survey(write_survey(tmp_path, [position([40, 44], {62: 54})])).findings
    assert (f.outside.move, f.inside.move) == ("M62", "M1")
    assert f.gain == pytest.approx(0.10)
    # The paired SE: 10 disagreements in 100 rollouts, mean 0.1.
    assert f.gain_se == pytest.approx((0.1 - 0.01) ** 0.5 / 10)
    assert f.sigmas == pytest.approx(10 / 3)
    assert f.spread_gain == pytest.approx(1.0)
    assert f.beats_cut


def test_each_pick_is_its_own_finding_and_a_small_edge_does_not_beat_the_cut(tmp_path):
    survey = load_survey(write_survey(tmp_path, [position([40], {62: 42, 80: 55})]))
    assert [f.outside.move for f in survey.findings] == ["M80", "M62"]  # strongest first
    assert [f.beats_cut for f in survey.findings] == [True, False]
    assert 0 < survey.findings[1].sigmas < 2


def test_the_report_counts_every_surveyed_position(tmp_path):
    nothing_standing = position([40], {62: 50}) | {"confirm": [], "game": 1}
    survey = load_survey(write_survey(tmp_path, [position([40], {62: 50}), nothing_standing]))
    assert (survey.positions, len(survey.findings)) == (2, 1)
    text = report(survey)
    assert "1 moves at 1 positions (50.0%)" in text
    assert "per surveyed position: +5.00" in text


def test_review_dir_collects_the_gcg_and_a_readme(tmp_path):
    survey = load_survey(write_survey(tmp_path, [position([40], {62: 50})]))
    gcg_dir = tmp_path / "gcg"
    gcg_dir.mkdir()
    name = gcg_name(("chunk", 0, 3))
    assert name == "chunk-g0-turn4.gcg"
    (gcg_dir / name).write_text("#note a game\n")
    write_review_dir(survey, 10, gcg_dir, tmp_path / "review", "the command")
    assert (tmp_path / "review" / name).exists()
    readme = (tmp_path / "review" / "README.md").read_text()
    assert "the command" in readme
    assert "M(62) (#63) | 50.0 | +5.0 | M(0) | 40.0 | +4.0 | +10.0 | +3.3 | +1.0 |" in readme


def test_winning_positions_counts_positions_not_moves(tmp_path):
    two_winners = position([40], {62: 55, 80: 56})
    survey = load_survey(
        write_survey(tmp_path, [two_winners, position([40], {62: 41}) | {"game": 1}])
    )
    assert len(survey.winners) == 2
    assert survey.winning_positions == {("chunk", 0, 3)}

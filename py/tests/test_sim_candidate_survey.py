"""The analysis of sim_candidate_survey_tool's .simsurvey.json files."""

import json
from pathlib import Path

import pytest
from scribblez.sim_candidate_survey import gcg_name, load_findings, report, write_review_dir

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


def position(cut_wins: list[int], outside_rank: int, outside_wins: int) -> dict:
    """A position whose confirming sim covers a cut of len(cut_wins) moves and one
    outside move, stored as the candidate after them."""
    ranks = [*range(len(cut_wins)), outside_rank]
    wins = [*cut_wins, outside_wins]
    candidates = [
        {"move": f"M{rank}", "equity_rank": rank, "is_setup": False, "screen": summary(50)}
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
    path.write_text(json.dumps({"version": 2, "positions": positions}))
    return [path]


def test_the_outside_move_is_measured_against_the_cuts_best(tmp_path):
    (f,) = load_findings(write_survey(tmp_path, [position([40, 44], 62, 54)]))
    assert (f.outside.move, f.inside.move) == ("M62", "M1")
    assert f.gain == pytest.approx(0.10)
    # The paired SE: 10 disagreements in 100 rollouts, mean 0.1.
    assert f.gain_se == pytest.approx((0.1 - 0.01) ** 0.5 / 10)
    assert f.sigmas == pytest.approx(10 / 3)
    assert f.spread_gain == pytest.approx(1.0)
    assert f.beats_cut


def test_a_small_edge_does_not_beat_the_cut(tmp_path):
    (f,) = load_findings(write_survey(tmp_path, [position([40], 62, 42)]))
    assert 0 < f.sigmas < 2
    assert not f.beats_cut


def test_unconfirmed_positions_are_skipped_and_the_report_counts_winners(tmp_path):
    unconfirmed = position([40], 62, 50) | {"confirm": [], "game": 1}
    found = load_findings(write_survey(tmp_path, [position([40], 62, 50), unconfirmed]))
    assert len(found) == 1
    assert "beats the cut by >= 2 sigma: 1 positions (100.0%)" in report(found)


def test_review_dir_collects_the_gcg_and_a_readme(tmp_path):
    found = load_findings(write_survey(tmp_path, [position([40], 62, 50)]))
    gcg_dir = tmp_path / "gcg"
    gcg_dir.mkdir()
    name = gcg_name(("chunk", 0, 3))
    assert name == "chunk-g0-turn4.gcg"
    (gcg_dir / name).write_text("#note a game\n")
    write_review_dir(found, 10, gcg_dir, tmp_path / "review", "the command")
    assert (tmp_path / "review" / name).exists()
    readme = (tmp_path / "review" / "README.md").read_text()
    assert "the command" in readme
    assert "M62 (#63) | 50.0 | +5.0 | M0 | 40.0 | +4.0 | +10.0 | +3.3 | +1.0 |" in readme

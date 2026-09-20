"""The held-out analysis of sim_candidate_survey_tool's .simsurvey.json files."""

import json
from pathlib import Path

import pytest
from scribblez.sim_candidate_survey import findings, gcg_name, load_survey, report, write_review_dir

ROLLOUTS = 100


def summary(wins: int, vs_cut: list[list[float]]) -> dict:
    return {
        "n": ROLLOUTS,
        "wins": wins,
        "draws": 0,
        "losses": ROLLOUTS - wins,
        "delta_sum": 10.0 * wins,
        "delta_sq_sum": 0.0,
        "win_diff_vs_cut": vs_cut,
    }


def paired(wins: int, ref_wins: int) -> list[float]:
    """Paired moments against a reference this candidate's rollouts disagree with
    as little as the two win counts allow (the tightest CRN can make them)."""
    return [float(wins - ref_wins), float(abs(wins - ref_wins))]


def candidate(rank: int, wins: tuple[int, int], ref_wins: tuple[int, int], is_setup=False) -> dict:
    return {
        "move": f"M{rank}",
        "equity_rank": rank,
        "is_play": True,
        "is_setup": is_setup,
        "replicas": [summary(w, [paired(w, r)]) for w, r in zip(wins, ref_wins, strict=True)],
    }


def write_survey(tmp_path: Path, candidates: list[dict]) -> list[Path]:
    path = tmp_path / "chunk.simsurvey.json"
    position = {"game": 0, "turn": 3, "candidates": candidates}
    path.write_text(json.dumps({"version": 1, "positions": [position]}))
    return [path]


def test_a_lucky_outside_pick_does_not_beat_the_cut(tmp_path):
    # Rank 40 looks best on replica 0 only; replica 1 prefers rank 0. Each pick
    # is valued on the replica that did not make it.
    ref = (50, 52)
    survey = load_survey(
        write_survey(tmp_path, [candidate(0, ref, ref), candidate(40, (60, 45), ref)])
    )
    (finding,) = findings(survey, cut=1)
    first, second = finding.picks
    assert first.free_pick_outside and not second.free_pick_outside
    assert first.gain == pytest.approx(-0.07)
    # The second assignment's outside pick does well on the valuing replica, but
    # the selecting replica never preferred it, so it confirms nothing.
    assert second.sigmas > 2 and not second.beats_cut
    assert not finding.beats_cut_once


def test_a_real_outside_play_beats_the_cut_on_both_replicas(tmp_path):
    ref = (40, 40)
    survey = load_survey(
        write_survey(tmp_path, [candidate(0, ref, ref), candidate(62, (50, 50), ref)])
    )
    (finding,) = findings(survey, cut=1)
    # The paired SE: 10 disagreements in 100 rollouts, mean 0.1.
    assert finding.picks[0].gain_se == pytest.approx((0.1 - 0.01) ** 0.5 / 10)
    assert finding.picks[0].sigmas == pytest.approx(10 / 3)
    assert finding.beats_cut_twice
    assert "1 on both with the same move" in report(survey, cut=1)


def test_setups_only_ignores_other_outside_plays(tmp_path):
    ref = (40, 40)
    cands = [
        candidate(0, ref, ref),
        candidate(62, (48, 48), ref, is_setup=True),
        candidate(80, (90, 90), ref),
    ]
    survey = load_survey(write_survey(tmp_path, cands))
    (finding,) = findings(survey, cut=1, setups_only=True)
    assert finding.picks[0].outside.move == "M62"
    assert findings(survey, cut=1)[0].picks[0].outside.move == "M80"


def test_review_dir_collects_the_gcg_and_a_readme(tmp_path):
    ref = (40, 40)
    survey = load_survey(
        write_survey(tmp_path, [candidate(0, ref, ref), candidate(62, (50, 50), ref)])
    )
    gcg_dir = tmp_path / "gcg"
    gcg_dir.mkdir()
    name = gcg_name(("chunk", 0, 3))
    assert name == "chunk-g0-turn4.gcg"
    (gcg_dir / name).write_text("#note a game\n")
    write_review_dir(survey, 1, False, gcg_dir, tmp_path / "review", "the command")
    assert (tmp_path / "review" / name).exists()
    readme = (tmp_path / "review" / "README.md").read_text()
    assert "the command" in readme
    assert "M62 (#63) | 50.0 | +5.0 | M0 | 40.0 | +4.0 | +10.0, +10.0 | +3.3, +3.3" in readme

"""The held-out analysis of sim_candidate_survey_tool's rows."""

from pathlib import Path

import pytest
from scribblez.sim_candidate_survey import held_out_picks, load_survey, report

HEADER = (
    "replica,game,turn,num_legal_moves,candidate,equity_rank,is_play,move_score,rollouts,"
    "wins,draws,losses,delta_sum,delta_sq_sum"
)


def survey_row(replica: int, game: int, rank: int, wins: int, is_play: int = 1) -> str:
    return f"{replica},{game},3,50,{rank},{rank},{is_play},10,100,{wins},0,{100 - wins},0,0"


def write_survey(tmp_path: Path, rows: list[str]) -> list[Path]:
    path = tmp_path / "chunk.simsurvey.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    return [path]


def test_held_out_gain_is_read_off_the_other_replica(tmp_path):
    # Game 0: rank 40 looks best on replica 0 only (a lucky draw); replica 1
    # prefers rank 0. Both picks are valued on the replica that did not pick.
    rows = [
        survey_row(0, 0, 0, 50),
        survey_row(0, 0, 40, 60),
        survey_row(1, 0, 0, 52),
        survey_row(1, 0, 40, 45),
    ]
    picks = held_out_picks(load_survey(write_survey(tmp_path, rows)), cut=10)
    assert [(p.pick_rank, pytest.approx(p.gain)) for p in picks] == [(40, -0.07), (0, 0.0)]
    assert [p.outside(10) for p in picks] == [True, False]


def test_report_prices_a_real_setup_play(tmp_path):
    # The tail move wins on both replicas: every pick is outside the cut and
    # the held-out gain is the true margin.
    rows = [survey_row(r, 0, rank, wins) for r in (0, 1) for rank, wins in ((0, 40), (62, 50))]
    text = report(load_survey(write_survey(tmp_path, rows)), cut=10)
    assert "sim pick outside the cut: 100.0% of picks" in text
    assert "held-out replica agrees (gain > 0): 100.0%" in text
    assert "+10.00" in text


def test_report_rejects_a_single_replica(tmp_path):
    rows = [survey_row(0, 0, 0, 50)]
    with pytest.raises(ValueError, match="two replicas"):
        report(load_survey(write_survey(tmp_path, rows)), cut=10)

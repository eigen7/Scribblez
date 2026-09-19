"""The held-out analysis of sim_candidate_survey_tool's rows."""

from pathlib import Path

import pytest
from scribblez.sim_candidate_survey import (
    gcg_name,
    held_out_picks,
    load_survey,
    report,
    setup_findings,
    write_review_dir,
)

HEADER = (
    "replica,game,turn,num_legal_moves,candidate,equity_rank,is_play,is_setup,move,move_score,"
    "rollouts,wins,draws,losses,delta_sum,delta_sq_sum"
)


def survey_row(replica: int, game: int, rank: int, wins: int, is_setup: int = 0) -> str:
    return (
        f"{replica},{game},3,50,{rank},{rank},1,{is_setup},M{rank},10,100,{wins},0,{100 - wins},0,0"
    )


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


def test_setup_findings_compare_the_best_setup_with_the_best_cut_move(tmp_path):
    # Rank 62 is a setup and rank 80 is not: only the setup is compared with
    # the cut, however well the other move sims.
    rows = []
    for replica in (0, 1):
        rows += [
            survey_row(replica, 0, 0, 40),
            survey_row(replica, 0, 62, 48, is_setup=1),
            survey_row(replica, 0, 80, 90),
        ]
    (finding,) = setup_findings(load_survey(write_survey(tmp_path, rows)), cut=10)
    assert finding.setup_moves == ("M62", "M62")
    assert finding.cut_moves == ("M0", "M0")
    assert finding.gains == pytest.approx((0.08, 0.08))
    assert finding.confirmed


def test_review_dir_collects_the_gcg_and_a_readme(tmp_path):
    rows = [
        survey_row(r, 0, rank, wins, is_setup=int(rank > 0))
        for r in (0, 1)
        for rank, wins in ((0, 40), (62, 48))
    ]
    survey = load_survey(write_survey(tmp_path, rows))
    gcg_dir = tmp_path / "gcg"
    gcg_dir.mkdir()
    name = gcg_name(("chunk", 0, 3))
    assert name == "chunk-g0-turn4.gcg"
    (gcg_dir / name).write_text("#note a game\n")
    write_review_dir(survey, 10, gcg_dir, tmp_path / "review", count=5)
    assert (tmp_path / "review" / name).exists()
    readme = (tmp_path / "review" / "README.md").read_text()
    assert "M62 (#63) | 48.0 | +0.0 | M0 | 40.0 | +0.0 | +8.0, +8.0 | +0.0, +0.0" in readme

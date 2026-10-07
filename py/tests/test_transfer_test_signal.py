"""The model-free transfer-signal measure (scribblez/transfer_test/signal.py)
on hand-made blocking matrices and candidates, and the engine's blocking
matrix over a generated corpus file."""

import shutil
import subprocess

import numpy as np
from scribblez.ffi import reply_blocking
from scribblez.transfer_test import exhibits as ex
from scribblez.transfer_test import signal as sg
from scribblez.transfer_test.probes import read_sprobe
from tests.test_transfer_test_reader_data import fake_file


def test_blocking_columns_count_only_other_candidates_probes():
    # Three candidates, two probes each; record r = c * 2 + i.
    blocked = np.zeros((6, 3), dtype=np.uint8)
    blocked[0, 1] = 1  # candidate 1 blocks candidate 0's probe 0
    blocked[4, 1] = 1  # ... and candidate 2's probe 0
    blocked[2, 1] = 1  # its own probe: ignored
    blocked[1, 2] = 1  # candidate 2 blocks candidate 0's probe 1
    damage = np.array([0.4, -0.4, 0.0, 0.0, 0.2, -0.2])
    damage_blocked, share = sg.blocking_columns(blocked, damage, probes=2)
    np.testing.assert_allclose(damage_blocked, [0.0, (0.4 + 0.2) / 4, -0.4 / 4])
    np.testing.assert_allclose(share, [0.0, 2 / 4, 1 / 4])


def _candidates(damage, miss, group) -> sg.Candidates:
    n = len(damage)
    return sg.Candidates(
        group=np.asarray(group),
        bag=np.full(n, 2),
        plausible=np.ones(n, dtype=bool),
        label=np.asarray(miss, float) + 0.5,
        label_var=np.full(n, 1e-4),
        prior=np.full(n, 0.5),
        damage_blocked=np.asarray(damage, float),
        share_blocked=np.asarray(damage, float),
    )


def test_slice_stats_read_within_row_differences_only():
    # Each row's damage is shifted by a row-wide constant that the miss does
    # not share; centering removes it, leaving a perfect relation.
    damage = [0.1, 0.2, 0.3, 5.1, 5.2, 5.3]
    miss = [-0.1, 0.0, 0.1, -0.1, 0.0, 0.1]
    c = _candidates(damage, miss, [0, 0, 0, 1, 1, 1])
    s = sg.slice_stats(c, np.ones(6, dtype=bool))
    assert s["positions"] == 2
    assert np.isclose(s["corr_damage_miss"], 1.0)
    assert np.isclose(s["slope_damage_miss"], 1.0)
    assert np.isclose(s["residual_sd"], 0.0)


def test_threat_mask_takes_the_widest_rows():
    # Row i's damage spreads in proportion to i: the top 3% are rows 97-99.
    group = np.repeat(np.arange(100), 3)
    damage = (np.arange(100)[:, None] * np.array([0.0, 0.01, 0.02])).reshape(-1)
    c = _candidates(damage, np.zeros(300), group)
    assert sg.threat_mask(c).nonzero()[0].tolist() == list(range(291, 300))


def test_position_columns_measure_damage_against_each_candidates_own_mean(tmp_path):
    f = fake_file(tmp_path, k=3, probes=2)
    rec = f.probes.records
    rec["p_win"], rec["p_draw"] = [0.9, 0.5, 0.2, 0.2, 0.6, 0.6], 0.0
    f.probes.candidates["stratum"] = 0
    f.labels["wins"] = [2, 4, 6]
    blocked = np.zeros((6, 3), dtype=np.uint8)
    blocked[1, 1] = 1  # candidate 1 blocks candidate 0's probe 1, 0.2 below its mean
    blocked[0, 2] = 1  # candidate 2 blocks candidate 0's probe 0, 0.2 above it
    cols = sg.position_columns(f, blocked, 0)
    np.testing.assert_allclose(cols["damage_blocked"], [0.0, 0.2 / 4, -0.2 / 4])
    np.testing.assert_allclose(cols["share_blocked"], [0.0, 1 / 4, 1 / 4])
    np.testing.assert_allclose(cols["label"], [0.2, 0.4, 0.6])
    np.testing.assert_allclose(cols["prior"], 0.5)

    f.labels["wins"] = 4  # every plausible candidate's label equal: decided
    assert sg.position_columns(f, blocked, 0) is None
    small = fake_file(tmp_path, k=2, probes=2, seed=1)
    assert sg.position_columns(small, np.zeros((4, 2), dtype=np.uint8), 0) is None


def _generate(tmp_path, slog, moves_by_turn) -> tuple:
    """A corpus .sprobe of `slog`'s game 0 probing the given candidates at the
    given turns, in its own directory; its .slog and probes."""
    tmp_path.mkdir()
    own = tmp_path / slog.name
    shutil.copy(slog, own)
    chosen = tmp_path / "chosen.txt"
    chosen.write_text(
        "".join(f"0 {turn} {m.tobytes().hex()}\n" for turn, ms in moves_by_turn for m in ms)
    )
    subprocess.run(
        [
            str(ex.GENERATOR),
            "--mode=corpus",
            f"--slog-file={own}",
            f"--chosen-moves={chosen}",
            "--horizon=0",
            "--probes=4",
            "--label-rollouts=2",
            "--threads=2",
        ],
        check=True,
        capture_output=True,
    )
    return own, read_sprobe(own.with_suffix(".sprobe"))


def test_reply_blocking_places_each_positions_rows_at_its_records(tmp_path):
    """Two positions of 2 and 3 candidates in one file: the second position's
    rows are the ones a file of it alone gives, and the first's third column
    is padding."""
    gcg = tmp_path / "pos-09.gcg"
    gcg.write_text(ex.decision_text(ex.EXHIBITS[0]))
    subprocess.run(
        [str(ex.GCG_TO_SLOG), f"--out-dir={tmp_path}", "--face-up", f"--gcg={gcg}"], check=True
    )
    (slog,) = tmp_path.glob("*.slog")
    subprocess.run(
        [
            str(ex.GENERATOR),
            "--mode=corpus",
            f"--slog-file={slog}",
            "--horizon=0",
            "--probes=1",
            "--label-rollouts=1",
            "--positions-per-game=2",
            "--threads=2",
        ],
        check=True,
        capture_output=True,
    )
    sampled = read_sprobe(slog.with_suffix(".sprobe"))
    turns = [int(t) for t in sampled.positions["turn_index"]]
    start, moves = sampled.candidate_start, sampled.candidates["move"]
    picks = [(turns[0], moves[start[0] : start[0] + 2]), (turns[1], moves[start[1] : start[1] + 3])]

    two_slog, two = _generate(tmp_path / "two", slog, picks)
    one_slog, one = _generate(tmp_path / "one", slog, picks[1:])
    both = reply_blocking(two_slog, two.path, len(two.records), stride=3, threads=2)
    alone = reply_blocking(one_slog, one.path, len(one.records), stride=3, threads=1)
    assert both.shape == (2 * 4 + 3 * 4, 3)
    np.testing.assert_array_equal(both[8:], alone)
    assert both[:8, 2].sum() == 0
    assert both.sum() > 0

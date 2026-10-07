"""The model-free transfer-signal measure (scribblez/transfer_test/signal.py)
on hand-made blocking matrices and candidates."""

import numpy as np
from scribblez.transfer_test import signal as sg


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

"""The subtree dump's belief readout (scripts.deep_bot.belief_truth):
rack-weighted means per static-equity rank bucket over the searched
candidates that carry a belief."""

import json

import pytest
from scripts.deep_bot import belief_truth


def _candidate(rank, racks=None, log_prob=0.0, prior=0.0):
    c = {"equity_rank": rank}
    if racks is not None:
        c["belief"] = {"racks": racks, "log_prob": log_prob, "prior_log_prob": prior}
    return c


def test_the_readout_weights_candidates_by_their_racks(tmp_path):
    dump = tmp_path / "dump.jsonl"
    turns = [
        {"candidates": [_candidate(0, 10, -5.0, -6.0), _candidate(3)]},
        {"candidates": [_candidate(4, 30, -7.0, -6.0), _candidate(40, 10, -6.5, -6.0)]},
        {"candidates": [_candidate(0, 30, -5.5, -6.0)]},
    ]
    dump.write_text("".join(json.dumps(t) + "\n" for t in turns))
    table = belief_truth.tabulate(belief_truth.rows(dump))
    assert set(table) == {"rank 0", "rank 1-9", "rank 10+", "all"}
    assert table["rank 0"]["gain"] == pytest.approx((10 * 1.0 + 30 * 0.5) / 40)
    assert (table["rank 0"]["candidates"], table["rank 0"]["worse_share"]) == (2, 0)
    assert table["rank 1-9"] == {"candidates": 1, "racks": 30, "gain": -1.0, "worse_share": 1}
    assert table["all"]["worse_share"] == 0.5

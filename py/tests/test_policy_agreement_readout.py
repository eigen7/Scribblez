"""The subtree dump's policy-agreement readout (scripts.deep_bot
.policy_agreement): rack-weighted means per bag bucket over the searched
candidates that carry a measurement."""

import json

from scripts.deep_bot import policy_agreement


def _turn(bag, racks, candidates):
    return {"bag": bag, "bob_racks": racks, "candidates": candidates}


def _candidate(hasty, pool, simmed, searched=True):
    c = {"searched": searched, "replies": {"hasty_share": hasty}}
    if searched:
        c["policy"] = {"pool_share": pool, "simmed_share": simmed}
    return c


def test_the_readout_weights_candidates_by_their_racks(tmp_path):
    dump = tmp_path / "dump.jsonl"
    turns = [
        _turn(70, 10, [_candidate(0.5, 0.6, 0.7), _candidate(0.1, 0.2, 0.3, searched=False)]),
        _turn(20, 30, [_candidate(0.9, 0.8, 1.0)]),
    ]
    dump.write_text("".join(json.dumps(t) + "\n" for t in turns))
    table = policy_agreement.tabulate(policy_agreement.rows(dump))
    assert set(table) == {"bag 60+", "bag 8-29", "all"}
    assert table["bag 60+"] == {"racks": 10, "hasty": 0.5, "policy_pool": 0.6, "policy_simmed": 0.7}
    assert table["all"]["racks"] == 40
    assert table["all"]["hasty"] == (10 * 0.5 + 30 * 0.9) / 40
    assert table["all"]["policy_pool"] == (10 * 0.6 + 30 * 0.8) / 40


def test_a_dump_without_a_policy_has_no_rows(tmp_path):
    dump = tmp_path / "dump.jsonl"
    dump.write_text(json.dumps(_turn(50, 5, [{"searched": True, "replies": {}}])) + "\n")
    assert policy_agreement.tabulate(policy_agreement.rows(dump)) == {}

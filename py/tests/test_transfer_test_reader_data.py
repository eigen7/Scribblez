"""M1a's reader data (scribblez/transfer_test/): the .sprobe reader, row
assembly and collation, and the token encoder, on synthetic corpora."""

import numpy as np
import pytest
import torch
from scribblez.position_eval.model import FOOTPRINT_CLASSES
from scribblez.sim_evidence.sobs import RECORD_DTYPE
from scribblez.transfer_test import probes as pr
from scribblez.transfer_test.corpus import CorpusFile, slim_labels
from scribblez.transfer_test.prior import Prior
from scribblez.transfer_test.rows import (
    ACTION,
    CANDIDATE,
    CHANCE,
    LEAF,
    ROOT,
    ROOT_TOKENS,
    RowConfig,
    assemble_row,
)
from scribblez.transfer_test.tokens import TokenEncoder, collate

TEACHER_WIDTH = 8


def _rack(codes: list[int]) -> bytes:
    return bytes(codes + [pr.NO_TILE] * (pr.RACK_SIZE - len(codes)))


def _sprobe_bytes(k: int, probes: int, turns_per_record: np.ndarray) -> bytes:
    hdr = np.zeros(1, pr.FILE_HEADER)
    hdr["magic"], hdr["version"], hdr["num_positions"] = pr.SPROBE_MAGIC, pr.SPROBE_VERSION, 1
    hdr["probes"], hdr["horizon_plies"], hdr["lexicon"] = probes, 3, b"NWL23"
    ph = np.zeros(1, pr.POSITION_HEADER)
    ph["game_index"], ph["turn_index"], ph["num_candidates"] = 2, 5, k
    ph["num_turns"] = turns_per_record.sum()
    cands = np.zeros(k, pr.CANDIDATE)
    cands["stratum"] = np.arange(k) % 4
    cands["equity"] = -np.arange(k, dtype=np.float32)
    records = np.zeros(k * probes, pr.RECORD)
    records["candidate"] = np.repeat(np.arange(k), probes)
    records["probe"] = np.tile(np.arange(probes), k)
    records["num_turns"] = turns_per_record
    records["p_win"] = np.linspace(0, 1, k * probes)
    turns = np.zeros(int(turns_per_record.sum()), pr.TURN)
    return b"".join(a.tobytes() for a in (hdr, ph, cands, records, turns))


def fake_file(tmp_path, k: int = 6, probes: int = 8, seed: int = 0) -> CorpusFile:
    """A one-position corpus file: every turn draws one tile except every
    third, and the replayed racks are all a single A."""
    rng = np.random.default_rng(seed)
    path = tmp_path / f"f{seed}.sprobe"
    path.write_bytes(_sprobe_bytes(k, probes, rng.integers(1, 4, size=k * probes)))
    probe_file = pr.read_sprobe(path)
    t = len(probe_file.turns)
    state = pr.ProbeReplay(
        roots=np.zeros(1, pr.struct_dtype("ProbeRootState")),
        candidates=np.zeros(k, pr.struct_dtype("ProbeCandidateState")),
        starts=np.zeros(k * probes, pr.struct_dtype("ProbeStartState")),
        turns=np.zeros(t, pr.struct_dtype("ProbeTurnState")),
    )
    for table, fields in (
        (state.candidates, ("leave",)),
        (state.starts, ("mover_drawn", "mover_rack", "opp_drawn", "opp_rack")),
        (state.turns, ("leave", "rack_after")),
    ):
        for field in fields:
            table[field] = _rack([0])
    state.turns["drawn"] = [_rack([] if i % 3 == 2 else [0]) for i in range(t)]
    state.candidates["bag_size"] = 50
    state.turns["ply"] = 1
    labels = slim_labels(np.zeros(k, RECORD_DTYPE["obs"]))
    labels["n"], labels["wins"], labels["losses"] = 10, 4, 6
    prior = Prior(
        root_board=rng.standard_normal((1, 225, TEACHER_WIDTH)).astype(np.float16),
        root_summary=rng.standard_normal((1, TEACHER_WIDTH)).astype(np.float16),
        wld=np.full((k, 3), 1 / 3, np.float32),
        score=np.zeros((k, 2), np.float32),
        placement=np.zeros((k, 4, FOOTPRINT_CLASSES), np.float16),
    )
    drew = pr.tile_counts(pr.tile_codes(state.turns["drawn"])).sum(axis=1) > 0
    return CorpusFile(probes=probe_file, replay=state, labels=labels, prior=prior, drew=drew)


def test_read_sprobe_addresses_records_and_turns(tmp_path):
    turns = np.array([1, 2, 3, 1, 2, 3])
    path = tmp_path / "x.sprobe"
    path.write_bytes(_sprobe_bytes(k=2, probes=3, turns_per_record=turns))
    f = pr.read_sprobe(path)
    assert (f.num_positions, f.probes, f.horizon_plies, f.lexicon) == (1, 3, 3, "NWL23")
    assert list(f.candidate_start) == [0, 2]
    assert list(f.turn_start) == [0, 1, 3, 6, 7, 9, 12]
    assert list(f.records["candidate"]) == [0, 0, 0, 1, 1, 1]

    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="trailing"):
        pr.read_sprobe(path)


def test_tile_counts():
    codes = pr.tile_codes(np.array([_rack([0, 0, 26]), _rack([])], dtype="S7"))
    counts = pr.tile_counts(codes)
    assert counts.shape == (2, pr.TILE_KINDS)
    assert (counts[0, 0], counts[0, 26], counts[0].sum(), counts[1].sum()) == (2, 1, 3, 0)


def _probe_tokens(row) -> np.ndarray:
    return np.isin(row.kind, [CHANCE, ACTION, LEAF])


def test_held_out_candidates_bring_no_evidence(tmp_path):
    f = fake_file(tmp_path)
    rng = np.random.default_rng(1)
    for _ in range(50):
        row = assemble_row(f, 0, 0, RowConfig(), rng)
        held = np.flatnonzero(row.held_out)
        assert 1 <= len(held) <= 4 and not row.held_out.all()
        assert not np.isin(row.slot[_probe_tokens(row)], held).any()


def test_graded_rows_keep_a_few_held_out_probes(tmp_path):
    f = fake_file(tmp_path)
    rng = np.random.default_rng(2)
    for _ in range(50):
        row = assemble_row(f, 0, 0, RowConfig(graded_max=2), rng)
        for slot in np.flatnonzero(row.held_out):
            leaves = np.sum((row.kind == LEAF) & (row.slot == slot))
            assert 1 <= leaves <= 2


def test_rows_fit_the_budget_and_index_their_tables_in_order(tmp_path):
    f = fake_file(tmp_path)
    cfg = RowConfig(max_tokens=300)
    row = assemble_row(f, 0, 0, cfg, np.random.default_rng(3))
    assert len(row.kind) <= cfg.max_tokens
    assert list(row.ref[row.kind == ROOT]) == list(range(ROOT_TOKENS))
    assert list(row.ref[row.kind == CANDIDATE]) == list(range(6))
    tables = {CHANCE: row.chance["drawn"], ACTION: row.action["move"], LEAF: row.leaf}
    for kind, table in tables.items():
        assert list(row.ref[row.kind == kind]) == list(range(len(table)))
    # Each probe opens with two deals, before its first ply.
    assert np.sum((row.kind == CHANCE) & (row.ply == 0)) == 2 * len(row.leaf)


def test_queries_ask_every_candidate_at_probe_boundaries(tmp_path):
    f = fake_file(tmp_path)
    row = assemble_row(f, 0, 0, RowConfig(query_points=3), np.random.default_rng(4))
    prefixes = np.unique(row.query_prefix)
    assert prefixes[-1] == len(row.kind) and len(prefixes) <= 3
    after_leaf = set(np.flatnonzero(row.kind == LEAF) + 1) | {ROOT_TOKENS + 6}
    assert set(prefixes) <= after_leaf
    for prefix in prefixes:
        assert sorted(row.query_slot[row.query_prefix == prefix]) == list(range(6))


def test_collated_refs_point_at_each_rows_features(tmp_path):
    files = [fake_file(tmp_path, k=k, seed=k) for k in (3, 6)]
    rng = np.random.default_rng(5)
    rows = [assemble_row(f, i, 0, RowConfig(), rng) for i, f in enumerate(files)]
    b = collate(rows)
    for i, row in enumerate(rows):
        n = len(row.kind)
        leaf = (b.kind[i, :n] == LEAF).numpy()
        np.testing.assert_array_equal(b.leaf[b.ref[i, :n][leaf]].numpy(), row.leaf[row.ref[leaf]])
        cand = (b.kind[i, :n] == CANDIDATE).numpy()
        flat = b.ref[i, :n][cand].numpy()
        np.testing.assert_array_equal(b.held_out[flat].numpy(), row.held_out)
        assert b.pad[i, n:].all() and not b.pad[i, :n].any()
    assert b.query_candidate[1, 0] == 3  # the second row's candidates follow the first's three


def test_encoder_embeds_every_token_and_query(tmp_path):
    f = fake_file(tmp_path)
    b = collate([assemble_row(f, 0, 0, RowConfig(), np.random.default_rng(6)) for _ in range(2)])
    enc = TokenEncoder(width=16, teacher_width=TEACHER_WIDTH, max_slots=16)
    context, queries = enc(b)
    assert context.shape == (*b.kind.shape, 16) and queries.shape == (*b.query_slot.shape, 16)
    assert torch.isfinite(context).all() and torch.isfinite(queries).all()

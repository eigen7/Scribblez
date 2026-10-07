"""M1a's exhibits (scribblez/transfer_test/exhibits.py): the GCG-to-.slog
conversion, the chosen-candidate generation, and the candidates chosen."""

import subprocess

import numpy as np
import pytest
from scribblez.ffi import position_eval_board_json, read_file_header
from scribblez.transfer_test import exhibits as ex
from scribblez.transfer_test.probes import read_sprobe, replay
from scribblez.transfer_test.rows import RowConfig, assemble_row
from tests.test_transfer_test_reader_data import fake_file


@pytest.mark.parametrize("exhibit", ex.EXHIBITS, ids=lambda e: e.name)
def test_every_blocker_is_among_the_exhibits_candidates(exhibit):
    text = ex.decision_text(exhibit)
    moves, is_blocker = ex.choose_candidates(exhibit, text)
    assert len(moves) == ex.CANDIDATES
    head = "\n".join(text.rstrip("\n").splitlines()[:-1])
    board = position_eval_board_json(head)["board"]
    names = {ex.notation(m, board) for m in moves[is_blocker]}
    assert names == set(exhibit.blockers)


def test_a_gcg_becomes_a_slog_the_generator_probes_with_chosen_moves(tmp_path):
    """pos-09 through gcg_to_slog, then the generator with two chosen
    candidates at its decision: the probes replay against the .slog from the
    GCG's position (bag 2, the mover 22 ahead)."""
    exhibit = ex.EXHIBITS[0]
    gcg = tmp_path / "pos-09.gcg"
    text = ex.decision_text(exhibit)
    gcg.write_text(text)
    subprocess.run(
        [str(ex.GCG_TO_SLOG), f"--out-dir={tmp_path}", "--face-up", f"--gcg={gcg}"], check=True
    )
    (slog,) = tmp_path.glob("*.slog")
    assert read_file_header(slog)[0] == 1
    moves, _ = ex.choose_candidates(exhibit, text)
    turn = sum(1 for line in text.splitlines() if line.startswith(">")) - 1
    chosen = tmp_path / "chosen.txt"
    chosen.write_text("".join(f"0 {turn} {m.tobytes().hex()}\n" for m in moves[:2]))
    subprocess.run(
        [
            str(ex.GENERATOR),
            "--mode=corpus",
            f"--slog-file={slog}",
            f"--chosen-moves={chosen}",
            "--horizon=0",
            "--probes=2",
            "--label-rollouts=2",
            "--threads=2",
        ],
        check=True,
    )
    probes = read_sprobe(slog.with_suffix(".sprobe"))
    assert probes.num_positions == 1 and probes.positions[0]["turn_index"] == turn
    assert probes.candidates["move"].tobytes() == moves[:2].tobytes()
    root = replay(probes).roots[0]
    assert (int(root["bag_size"]), int(root["score_diff"])) == (2, 22)


def test_a_row_holds_out_exactly_the_given_candidates(tmp_path):
    f = fake_file(tmp_path)
    held = np.zeros(6, dtype=bool)
    held[[1, 4]] = True
    row = assemble_row(f, 0, 0, RowConfig(), np.random.default_rng(0), held=held)
    assert row.held_out.tolist() == held.tolist()

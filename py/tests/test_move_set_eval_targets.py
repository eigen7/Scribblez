"""Tests for the move set evaluation model distillation-target data path:
move_set_eval_target_generator over
.slog fixtures with a tiny teacher ONNX, parsed by the .mset reader. Skipped
when the engine binaries, HastyBot leave values, or torch/TensorRT are
unavailable (the tool builds a TensorRT engine, so this is a GPU test).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scribblez.move_set_eval.targets import MSET_FLAG_OPEN_LEAVES, TARGET_NAMES_V1, read_mset
from scribblez.position_eval.model import PositionEvalModel
from scribblez.position_eval.onnx_export import export_onnx
from scribblez.sim_evidence.slog_meta import game_metas, move_at, read_slog_bytes

# This checkout's own binaries (not the primary checkout's), so a worktree's
# tests exercise the code built beside them -- the external-data test below
# exists precisely to catch a neural_net.cpp regression, which a fixed path to
# /workspace/repo would mask until after merge.
_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
TARGET_GENERATOR = _ENGINE_DIR / "move_set_eval_target_generator"
SLOG_WRITER = _ENGINE_DIR / "test_slog_writer"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

QUOTAS = {"top": 3, "mid": 2, "tail": 2, "exchange": 1}


def _skip_unless_runnable():
    if not TARGET_GENERATOR.exists() or not SLOG_WRITER.exists():
        pytest.skip("engine binaries not built")
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")


def _export_tiny_teacher(onnx_path: Path):
    from scribblez.ffi import get_input_shapes

    shapes = {s.name: s.dims for s in get_input_shapes()}
    torch.manual_seed(0)
    model = PositionEvalModel(
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        trunk_channels=8,
        num_blocks=3,
    ).eval()
    export_onnx(
        model,
        onnx_path,
        spatial_planes=shapes["input_spatial"][0],
        scalar_size=shapes["input_scalar"][0],
        contingent_features=True,
        opp_leave_input=False,
    )


@pytest.fixture(scope="module")
def mset_dir(tmp_path_factory) -> Path:
    """A directory of .slog files labeled by the tool with a tiny teacher."""
    _skip_unless_runnable()
    d = tmp_path_factory.mktemp("move_set_eval_targets")
    subprocess.run([str(SLOG_WRITER), str(d), "8", "4"], check=True, capture_output=True)
    onnx_path = d / "teacher.onnx"
    _export_tiny_teacher(onnx_path)

    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-dir={d}",
            f"--model={onnx_path}",
            "--fast-build",
            f"--quota-top={QUOTAS['top']}",
            f"--quota-mid={QUOTAS['mid']}",
            f"--quota-tail={QUOTAS['tail']}",
            f"--quota-exchange={QUOTAS['exchange']}",
            "--positions-per-game=2",
            "--threads=4",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"move_set_eval_target_generator failed: {result.stderr}"
    assert sorted(d.glob("*.mset")), "no .mset produced"
    return d


def test_mset_positions_parse_and_hold_invariants(mset_dir):
    max_candidates = 1 + sum(QUOTAS.values())  # the played move + the strata
    total = 0
    for path in sorted(mset_dir.glob("*.mset")):
        parsed = read_mset(path)
        assert parsed.record_floats == len(TARGET_NAMES_V1)
        assert len(parsed.model_hash) > 0
        assert parsed.flags == 0
        for pos in parsed.positions:
            total += 1
            assert 1 <= len(pos.moves) <= max_candidates
            assert pos.targets.shape == (len(pos.moves), len(TARGET_NAMES_V1))
            # WLD probabilities: each a distribution summing to ~1.
            wld = pos.targets[:, :3]
            assert np.all(wld >= 0) and np.all(wld <= 1)
            assert np.allclose(wld.sum(axis=1), 1.0, atol=1e-3)
            # The std head is softplus-floored positive.
            assert np.all(pos.targets[:, 4] > 0)
    assert total > 0


def test_mset_includes_the_played_move(mset_dir):
    slog = sorted(mset_dir.glob("*.slog"))[0]
    buf = read_slog_bytes(slog)
    metas = game_metas(buf)
    parsed = read_mset(slog.with_suffix(".mset"))
    for pos in parsed.positions:
        played = move_at(buf, metas[pos.game_index], pos.turn_index).tobytes()
        assert any(pos.moves[i].tobytes() == played for i in range(len(pos.moves)))


def test_mset_model_hash_consistent_across_files(mset_dir):
    hashes = {read_mset(p).model_hash for p in mset_dir.glob("*.mset")}
    assert len(hashes) == 1


def _run_full_sweep(slog_dir: Path, onnx_path: Path, cap: int):
    return subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-dir={slog_dir}",
            f"--model={onnx_path}",
            "--fast-build",
            "--full-sweep",
            f"--sweep-cap={cap}",
            "--positions-per-game=2",
            "--threads=4",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )


def test_full_sweep_labels_every_candidate_and_records_the_legal_count(tmp_path):
    """The evaluation slice: an uncapped sweep labels a position's whole legal
    move set -- orders of magnitude past the stratified quota -- and stamps the
    file full-sweep so the trainer holds it out."""
    _skip_unless_runnable()
    subprocess.run([str(SLOG_WRITER), str(tmp_path), "4", "4"], check=True, capture_output=True)
    onnx_path = tmp_path / "teacher.onnx"
    _export_tiny_teacher(onnx_path)

    result = _run_full_sweep(tmp_path, onnx_path, cap=100000)
    assert result.returncode == 0, f"full-sweep run failed: {result.stderr}"
    msets = sorted(tmp_path.glob("*.mset"))
    assert msets, "no .mset produced"

    biggest = 0
    for path in msets:
        parsed = read_mset(path)
        assert parsed.full_sweep
        for pos in parsed.positions:
            # Uncapped, the sweep is complete: candidates == the legal count.
            assert pos.num_legal_moves == len(pos.moves)
            assert pos.targets.shape == (len(pos.moves), len(TARGET_NAMES_V1))
            biggest = max(biggest, len(pos.moves))
    assert biggest > 1 + sum(QUOTAS.values()), "a sweep should dwarf the stratified sample"


def test_full_sweep_cap_truncates_visibly_and_keeps_the_played_move(tmp_path):
    """Under a cap the sweep is a prefix of the equity ranking plus the
    exchanges and the played move, and the shortfall against the recorded legal
    count is what makes the truncation visible."""
    _skip_unless_runnable()
    subprocess.run([str(SLOG_WRITER), str(tmp_path), "4", "4"], check=True, capture_output=True)
    onnx_path = tmp_path / "teacher.onnx"
    _export_tiny_teacher(onnx_path)

    assert _run_full_sweep(tmp_path, onnx_path, cap=3).returncode == 0
    truncated = 0
    for path in sorted(tmp_path.glob("*.mset")):
        buf = read_slog_bytes(path.with_suffix(".slog"))
        metas = game_metas(buf)
        for pos in read_mset(path).positions:
            assert len(pos.moves) <= pos.num_legal_moves
            truncated += len(pos.moves) < pos.num_legal_moves
            played = move_at(buf, metas[pos.game_index], pos.turn_index).tobytes()
            assert any(pos.moves[i].tobytes() == played for i in range(len(pos.moves)))
    assert truncated > 0, "a cap of 3 should truncate some position"


def test_generator_refuses_face_up_slogs_with_a_blind_teacher(tmp_path):
    """The pre-scan guard: a face-up-leaves .slog paired with a teacher whose
    ONNX does not declare opp_leave_input must fail before any .mset is
    written -- a blind teacher's labels would describe games nobody played."""
    _skip_unless_runnable()
    subprocess.run(
        [str(SLOG_WRITER), str(tmp_path), "2", "2", "--face-up-leaves"],
        check=True,
        capture_output=True,
    )
    onnx_path = tmp_path / "teacher.onnx"
    _export_tiny_teacher(onnx_path)  # a blind teacher: opp_leave_input=False
    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-dir={tmp_path}",
            f"--model={onnx_path}",
            "--fast-build",
            f"--quota-top={QUOTAS['top']}",
            f"--quota-mid={QUOTAS['mid']}",
            f"--quota-tail={QUOTAS['tail']}",
            f"--quota-exchange={QUOTAS['exchange']}",
            "--positions-per-game=2",
            "--threads=2",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "opp_leave_input" in result.stderr
    assert not list(tmp_path.glob("*.mset"))


def test_generator_labels_face_up_slogs_with_an_open_leaves_teacher(tmp_path):
    """The guard's allowed branch: a face-up corpus under a teacher that
    declares opp_leave_input labels successfully, and each .mset is stamped
    open-leaves after its source log."""
    _skip_unless_runnable()
    subprocess.run(
        [str(SLOG_WRITER), str(tmp_path), "2", "2", "--face-up-leaves"],
        check=True,
        capture_output=True,
    )
    onnx_path = tmp_path / "teacher.onnx"
    # The open-leaves input arm changes the FFI session's row layout, which is
    # process-wide and fixed at creation -- export this teacher in a subprocess
    # (the test_sim_evidence.py pattern).
    code = (
        "import torch\n"
        "from scribblez.ffi import set_opp_leave_input, get_input_shapes\n"
        "from scribblez.position_eval.model import PositionEvalModel\n"
        "from scribblez.position_eval.onnx_export import export_onnx\n"
        "set_opp_leave_input(True)\n"
        "shapes = {s.name: s.dims for s in get_input_shapes()}\n"
        "torch.manual_seed(0)\n"
        "model = PositionEvalModel(spatial_planes=shapes['input_spatial'][0],\n"
        "                          scalar_size=shapes['input_scalar'][0],\n"
        "                          trunk_channels=8, num_blocks=3).eval()\n"
        f"export_onnx(model, {str(onnx_path)!r},\n"
        "            spatial_planes=shapes['input_spatial'][0],\n"
        "            scalar_size=shapes['input_scalar'][0],\n"
        "            contingent_features=True, opp_leave_input=True)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-dir={tmp_path}",
            f"--model={onnx_path}",
            "--fast-build",
            f"--quota-top={QUOTAS['top']}",
            f"--quota-mid={QUOTAS['mid']}",
            f"--quota-tail={QUOTAS['tail']}",
            f"--quota-exchange={QUOTAS['exchange']}",
            "--positions-per-game=2",
            "--threads=2",
            "--seed=7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"generator refused the allowed pairing: {result.stderr}"
    msets = sorted(tmp_path.glob("*.mset"))
    assert msets, "no .mset produced"
    assert all(read_mset(p).flags == MSET_FLAG_OPEN_LEAVES for p in msets)


def test_generator_loads_a_teacher_with_external_data(tmp_path):
    """A teacher whose initializers live in an external-data blob beside the
    .onnx must load regardless of the process CWD: TensorRT resolves external
    references against the model path the engine passes (neural_net.cpp), not
    the CWD. A regression re-breaks exactly this run, because the CWD here is
    deliberately far from the model's directory."""
    import onnx

    _skip_unless_runnable()
    subprocess.run([str(SLOG_WRITER), str(tmp_path), "2", "2"], check=True, capture_output=True)
    plain = tmp_path / "teacher.onnx"
    _export_tiny_teacher(plain)

    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    model = onnx.load(str(plain))
    onnx.save_model(
        model,
        str(ext_dir / "teacher.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="teacher_data.bin",
        size_threshold=0,
    )
    plain.unlink()  # only the external-data variant remains loadable

    result = subprocess.run(
        [
            str(TARGET_GENERATOR),
            f"--slog-dir={tmp_path}",
            f"--model={ext_dir / 'teacher.onnx'}",
            "--fast-build",
            "--positions-per-game=1",
            "--limit-games=2",
            "--threads=2",
        ],
        capture_output=True,
        text=True,
        cwd="/",
    )
    assert result.returncode == 0, f"generator failed from a foreign CWD: {result.stderr}"
    assert sorted(tmp_path.glob("*.mset")), "no .mset produced"

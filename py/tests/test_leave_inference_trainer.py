"""The leave_inference workload's trainer pieces: the resume gate, the
face-up refusal, the held-out set's reproducibility, the eval metrics, and
the workload's registration."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez import ffi, workloads  # noqa: E402
from scribblez.leave_inference import holdout, train_loop, trainer  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.workloads import leave_inference  # noqa: E402


def _paths(tmp_path) -> TagPaths:
    return TagPaths("t", "leave_inference", tmp_path)


def _save_checkpoint(paths: TagPaths, config: dict):
    paths.rolling_checkpoint.parent.mkdir(parents=True)
    torch.save({"config": config}, paths.rolling_checkpoint)


def test_resume_gate_accepts_a_matching_checkpoint(tmp_path):
    paths = _paths(tmp_path)
    config = {trainer.FORMAT_KEY: 1, trainer.TABLE_KEY: "abc"}
    trainer.check_resumable(paths, config)  # no checkpoint yet
    _save_checkpoint(paths, config)
    trainer.check_resumable(paths, config)


@pytest.mark.parametrize("key", [trainer.FORMAT_KEY, trainer.TABLE_KEY])
def test_resume_gate_refuses_another_format_or_table(tmp_path, key):
    paths = _paths(tmp_path)
    saved = {trainer.FORMAT_KEY: 1, trainer.TABLE_KEY: "abc"}
    _save_checkpoint(paths, saved)
    with pytest.raises(RuntimeError, match=key):
        trainer.check_resumable(paths, {**saved, key: "other"})


def test_face_up_corpus_is_refused(tmp_path):
    binary = _ENGINE_DIR / "test_slog_writer"
    if not binary.is_file():
        pytest.skip("test_slog_writer not built")
    hidden, face_up = tmp_path / "hidden", tmp_path / "face_up"
    hidden.mkdir()
    subprocess.run([str(binary), str(hidden), "2", "2"], check=True, capture_output=True)
    shutil.copytree(hidden, face_up)
    header = ffi.struct_dtype("SlogFileHeader")
    for path in face_up.glob("*.slog"):
        buf = bytearray(path.read_bytes())
        h = np.frombuffer(buf, header, count=1)
        h["flags"] |= trainer.FACE_UP_FLAG
        path.write_bytes(bytes(buf))
    guard = trainer.FaceUpGuard()
    guard.check([hidden])
    with pytest.raises(RuntimeError, match="face-up"):
        guard.check([face_up])


def test_holdout_set_is_reproducible(tmp_path):
    if not (_ENGINE_DIR / "play_game").is_file():
        pytest.skip("play_game not built")
    spec = "--type=hastybot-endgame"
    a = holdout.ensure(tmp_path / "a", games=20, seed=3, player_spec=spec)
    b = holdout.ensure(tmp_path / "b", games=20, seed=3, player_spec=spec)
    c = holdout.ensure(tmp_path / "c", games=20, seed=4, player_spec=spec)
    assert a.slogs and a.content_hash == b.content_hash != c.content_hash
    # A complete set is reused, not replayed.
    assert holdout.ensure(tmp_path / "a", games=20, seed=3, player_spec=spec).slogs == a.slogs
    with pytest.raises(ValueError):
        holdout.ensure(tmp_path / "d", games=20, seed=0, player_spec=spec)


def test_eval_metrics_breakdowns():
    rows = {
        "nll": np.array([1.0, 2.0, 3.0, 4.0]),
        "uniform": np.array([1.5, 2.5, 3.0, 5.0]),
        "length": np.array([1, 1, 3, 6]),
        "bag": np.array([10.0, 30.0, 50.0, 70.0]),
        "presence": np.full((4, 27), 0.25),
        "true_presence": np.zeros((4, 27)),
        "value_mean": np.array([0.0, 1.0, 2.0, 3.0]),
        "value_sd": np.array([1.0, 1.0, 1.0, 1.0]),
        "true_value": np.array([0.5, 3.0, 2.0, 3.5]),
    }
    causes = {"valid": 4, "no_opponent_move": 1, "empty_leave": 3, "pass": 0}
    m = train_loop.eval_metrics(rows, causes)
    assert m["eval_nll_vs_uniform"] == pytest.approx(-0.5)
    assert m["eval_nll_vs_uniform_len_1"] == pytest.approx(-0.5)
    assert m["eval_nll_vs_uniform_len_6"] == pytest.approx(-1.0)
    assert "eval_nll_vs_uniform_len_2" not in m
    assert m["eval_nll_vs_uniform_bag_61_100"] == pytest.approx(-1.0)
    assert m["eval_letter_calibration"] == pytest.approx(0.25)
    assert m["eval_value_coverage"] == pytest.approx(0.75)
    assert m["eval_invalid_empty_leave"] == pytest.approx(3 / 8)


def test_workload_registration_and_hidden_leaves_generation(monkeypatch):
    spec = workloads.get("leave_inference")
    assert spec.role("train").ingest  # a generational tag
    calls = []
    monkeypatch.setattr(
        leave_inference,
        "generate",
        lambda ctx, player, face_up_leaves, random_opening_mean: calls.append(
            (face_up_leaves, random_opening_mean)
        ),
    )

    class Ctx:
        params = leave_inference.LeaveInferenceParams()

    leave_inference.run_generate(Ctx())
    assert calls == [(False, 0.0)]

"""The leave_inference workload's trainer pieces: the resume gate, the
face-up refusal, the held-out set's reproducibility, the eval metrics, the
per-generation export, and the workload's registration."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez import ffi, workloads  # noqa: E402
from scribblez.dataset import SlogDataset  # noqa: E402
from scribblez.leave_inference import forgone, holdout, loss, train_loop, trainer  # noqa: E402
from scribblez.leave_inference.heads import HEAD_TILTED, HEADS  # noqa: E402
from scribblez.leave_inference.model import LeaveInferenceModel  # noqa: E402
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


def test_holdout_set_is_reproducible(tmp_path, monkeypatch):
    if not (_ENGINE_DIR / "play_game").is_file():
        pytest.skip("play_game not built")
    # The sidecar tool would otherwise take every core.
    monkeypatch.setattr(holdout, "default_thread_count", lambda: 1)
    spec = "--type=hastybot-endgame"
    a = holdout.ensure(tmp_path / "a", games=2, seed=3, player_spec=spec)
    b = holdout.ensure(tmp_path / "b", games=2, seed=3, player_spec=spec)
    c = holdout.ensure(tmp_path / "c", games=2, seed=4, player_spec=spec)
    assert a.slogs and a.content_hash == b.content_hash != c.content_hash
    # A complete set is reused, not replayed.
    assert holdout.ensure(tmp_path / "a", games=2, seed=3, player_spec=spec).slogs == a.slogs
    with pytest.raises(ValueError):
        holdout.ensure(tmp_path / "d", games=2, seed=0, player_spec=spec)


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
        lambda ctx, player, face_up_leaves, random_opening_mean, sidecar: calls.append(
            (face_up_leaves, random_opening_mean, sidecar.suffix)
        ),
    )

    class Ctx:
        params = leave_inference.LeaveInferenceParams()

    leave_inference.run_generate(Ctx())
    assert calls == [(False, 0.0, forgone.SUFFIX)]


@pytest.mark.parametrize("head", HEADS)
def test_each_generation_exports_the_served_head_only(tmp_path, head):
    paths = _paths(tmp_path)
    shapes = {s.name: s.dims for s in ffi.task_input_shapes("leave_inference")}
    model = LeaveInferenceModel(
        shapes["input_spatial"][0], shapes["input_scalar"][0], 8, 1, head=head
    )
    params = leave_inference.LeaveInferenceParams(head=head)
    ctx = {
        "tag": "t",
        "config": {trainer.TABLE_KEY: "abc"},
        "spatial_planes": shapes["input_spatial"][0],
        "scalar_size": shapes["input_scalar"][0],
    }
    trainer.export_generation(paths, params, 3, model, ctx)
    assert paths.onnx_path(3).exists() == (head == HEAD_TILTED)


class _OrderRecorder:
    """Stands in for the output sink and the generation record, logging the
    order of their deliveries."""

    def __init__(self):
        self.events = []

    def deliver_output(self, path, rel, keep=False):
        self.events.append(rel)

    def deliver_staged(self, staged):
        self.events.append("record")


@pytest.mark.parametrize("head", HEADS)
def test_delivery_ships_the_served_model_before_the_record(tmp_path, monkeypatch, head):
    monkeypatch.setattr(trainer.state_pair, "deliver", lambda *args: None)
    paths = _paths(tmp_path)
    paths.win_table_path(3).parent.mkdir(parents=True, exist_ok=True)
    paths.win_table_path(3).touch()
    if head == HEAD_TILTED:
        paths.onnx_path(3).parent.mkdir(parents=True, exist_ok=True)
        paths.onnx_path(3).touch()
    recorder = _OrderRecorder()
    trainer._deliver_generation(recorder, paths, 3, None, None, recorder, None)
    model_rel = str(paths.onnx_path(3).relative_to(paths.root))
    assert (model_rel in recorder.events) == (head == HEAD_TILTED)
    assert recorder.events[-1] == "record"


# The tilted head's readouts, which the autoregressive head does not report.
TILTED_ONLY = {"eval_letter_calibration", "eval_value_bias", "eval_value_coverage"}


@pytest.fixture(scope="module")
def training_corpus(tmp_path_factory) -> tuple[list[dict], loss.TableTensors]:
    """The batches of a small corpus with sidecars, and the leave table on the
    CPU: built once for both heads' runs, which only read them."""
    binary = _ENGINE_DIR / "test_slog_writer"
    if not binary.is_file():
        pytest.skip("test_slog_writer not built")
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    root = tmp_path_factory.mktemp("corpus")
    # A pass streams the whole 737k-row leave table whatever the batch, so the
    # test's cost is its pass count: four games make one batch, replayed once.
    subprocess.run([str(binary), str(root), "4", "4"], check=True, capture_output=True)
    forgone.write_sidecars(root, threads=2)
    ds = SlogDataset(
        root, task="leave_inference", post_move=False, apply_symmetry=False, num_workers=1
    )
    return list(ds.iter_batches(64, seed=0)), loss.TableTensors(ffi.LeaveValueArrays(), "cpu")


@pytest.mark.parametrize("head", HEADS)
def test_epoch_and_eval_train_each_head(training_corpus, head):
    corpus, table = training_corpus
    shapes = {s.name: s.dims for s in ffi.task_input_shapes("leave_inference")}
    torch.manual_seed(0)
    model = LeaveInferenceModel(
        shapes["input_spatial"][0], shapes["input_scalar"][0], 8, 1, head=head
    )

    def batches():
        return iter(corpus)

    before = train_loop.evaluate(model, table, batches(), "cpu")
    assert {"eval_nll", "eval_invalid_no_opponent_move"} <= set(before)
    if head == HEAD_TILTED:
        # The zero-initialized head is the uniform prior, up to b's small start.
        assert abs(before["eval_nll_vs_uniform"]) < 1e-2
        assert TILTED_ONLY <= set(before)
    else:
        assert not TILTED_ONLY & set(before)
    # 3e-3: at 1e-2 the first epochs overshoot on this tiny corpus, so a
    # short smoke test can end above where it started.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rows = sum(len(b["opp_leave_valid"]) for b in batches())
    for _ in range(2):
        result = train_loop.run_epoch(
            model, optimizer, table, batches(), "cpu", lr_fn=None, rows_trained=0, grad_clip=1.0
        )
        assert result.rows_trained == rows
        assert 0 < result.valid_frac < 1 and np.isfinite(result.loss)
    after = train_loop.evaluate(model, table, batches(), "cpu")
    assert after["eval_nll"] < before["eval_nll"]

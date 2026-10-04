"""The transfer_test workload's wiring and its step-0 analysis
(scribblez/transfer_test_measure.py)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scribblez import params as params_mod
from scribblez import transfer_test_measure as tm
from scribblez import workloads
from scribblez.workloads import transfer_test


def test_registered_and_queueable():
    spec = workloads.get("transfer_test")
    assert spec.layout and spec.gpu_need
    slots = transfer_test.layout(
        transfer_test.TransferTestParams(), vcpus=28, generator_threads=None
    )
    assert [(s.role, s.threads) for s in slots] == [("generate", 28)]
    assert slots[0].gpu_gb == transfer_test.GENERATOR_GPU_GB


@pytest.mark.parametrize(
    ("positions", "per_batch", "files"), [(300, 20, 15), (301, 20, 16), (-1, 20, -1)]
)
def test_target_files_covers_the_target_positions(positions, per_batch, files):
    params = transfer_test.TransferTestParams(target_positions=positions, games_per_batch=per_batch)
    assert transfer_test.target_files(params) == files


def test_generator_command(monkeypatch):
    commands = []

    def fake_run(cmd):
        commands.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(transfer_test.subprocess, "run", fake_run)
    # Distinct values per field, so a flag wired to the wrong param fails.
    params = transfer_test.TransferTestParams(
        rollouts=1000, horizon=4, saturation_probes=64, option_k=12
    )
    slogs = [Path("/data/a.slog"), Path("/data/b.slog")]
    assert transfer_test.run_generator(slogs, params, threads=8, model="/m/leaf.onnx") == 0
    assert len(commands) == 2
    cmd = commands[0]
    assert cmd[0] == transfer_test.TRANSFER_TEST_GENERATOR
    flags = dict(arg.split("=", 1) for arg in cmd[1:])
    assert flags == {
        "--mode": "measure",
        "--slog-file": "/data/a.slog",
        "--leaf-model": "/m/leaf.onnx",
        "--horizon": "4",
        "--rollouts": "1000",
        "--saturation-probes": "64",
        "--option-k": "12",
        "--seed": flags["--seed"],
        "--threads": "8",
    }
    # Each file is measured under a seed of its own.
    assert flags["--seed"] != dict(arg.split("=", 1) for arg in commands[1][1:])["--seed"]


def test_corpus_generator_command(monkeypatch):
    commands = []
    monkeypatch.setattr(
        transfer_test.subprocess,
        "run",
        lambda cmd: commands.append(cmd) or SimpleNamespace(returncode=0),
    )
    params = transfer_test.TransferTestParams(
        mode=transfer_test.MODE_CORPUS, probes_per_candidate=64, label_rollouts=1000
    )
    assert transfer_test.run_generator([Path("/d/a.slog")], params, 4, "/m/leaf.onnx") == 0
    flags = dict(arg.split("=", 1) for arg in commands[0][1:])
    assert flags["--mode"] == "corpus"
    assert flags["--probes"] == "64"
    assert flags["--label-rollouts"] == "1000"
    assert "--rollouts" not in flags and "--saturation-probes" not in flags


def test_profiles_set_up_the_corpus_runs():
    spec = workloads.get("transfer_test")
    assert spec.default_profile == "train-corpus"
    train = params_mod.validate(spec.params_cls, spec.profiles["train-corpus"])
    test = params_mod.validate(spec.params_cls, spec.profiles["test-corpus"])
    assert (train.mode, train.label_rollouts) == (transfer_test.MODE_CORPUS, 100)
    assert (test.mode, test.label_rollouts) == (transfer_test.MODE_CORPUS, 1000)
    # A tag stored before modes existed keeps measuring.
    assert params_mod.validate(spec.params_cls, {}).mode == transfer_test.MODE_MEASURE


def test_the_scheduler_counts_the_corpus_store_in_corpus_mode(tmp_path):
    store = tmp_path / transfer_test.CORPUS_DIR
    store.mkdir()
    (store / f"a{transfer_test.PROBE_EXT}").touch()
    finished = []
    params = {"mode": "corpus", "target_positions": 20, "games_per_batch": 20}
    hooks = SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path), finish=finished.append)
    spec = transfer_test.SPEC
    workloads.resolve(spec.scheduler)(spec, SimpleNamespace(params=params), hooks)
    assert finished == ["generate"]


def test_the_scheduler_finishes_the_generators_at_target_positions(tmp_path):
    """An ssh generator delivers into its own container and cannot count the
    store; the controller, which holds it whole, stops them all."""
    store = tmp_path / transfer_test.STORE_DIR
    store.mkdir()
    for stem in ("a", "b"):
        (store / f"{stem}{transfer_test.JSON_EXT}").touch()
    spec = transfer_test.SPEC

    def finished_after_tick(target_positions: int) -> list[str]:
        finished = []
        params = {"target_positions": target_positions, "games_per_batch": 20}
        hooks = SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path), finish=finished.append)
        workloads.resolve(spec.scheduler)(spec, SimpleNamespace(params=params), hooks)
        return finished

    assert finished_after_tick(41) == []  # 3 files
    assert finished_after_tick(40) == ["generate"]  # 2 files
    assert finished_after_tick(params_mod.UNBOUNDED) == []


def _synthetic(rng, means, luck_sd, noise_sd, rollouts):
    """(K, R) expected scores: per-candidate means, luck shared by every
    candidate within a rollout (the common random numbers), and candidate noise."""
    luck = rng.normal(0, luck_sd, rollouts)
    noise = rng.normal(0, noise_sd, (len(means), rollouts))
    return np.asarray(means)[:, None] + luck[None, :] + noise


def test_noise_signal_removes_shared_luck():
    rng = np.random.default_rng(0)
    means = [0.50, 0.52, 0.47, 0.55]
    values = _synthetic(rng, means, luck_sd=0.3, noise_sd=0.05, rollouts=20000)
    ns = tm.noise_signal(values)
    # Centering leaves the candidate noise (variance shrunk by (K-1)/K), not the luck.
    assert ns.noise_per_rollout == pytest.approx(0.05**2 * 3 / 4, rel=0.05)
    assert ns.signal == pytest.approx(np.var(means, ddof=1), rel=0.05)
    assert tm.independent_to_paired_variance(values) > 20


def test_noise_to_signal_scales_with_the_budget():
    ns = tm.NoiseSignal(noise_per_rollout=0.01, signal=0.001)
    assert tm.noise_to_signal(ns, 100) == pytest.approx(0.1)
    assert tm.noise_to_signal(tm.NoiseSignal(0.01, -0.001), 100) == float("inf")
    # One candidate (a stratum filter can leave one) has no spread: not resolvable.
    assert tm.noise_to_signal(tm.noise_signal(np.ones((1, 50))), 100) == float("inf")


def test_noise_signal_reads_no_signal_where_there_is_none():
    """Unbiased at a small budget: the centered labels' spread carries
    K/(K-1) times the per-candidate noise, all of which is removed."""
    rng = np.random.default_rng(1)
    signals = [
        tm.noise_signal(_synthetic(rng, [0.5] * 4, luck_sd=0.3, noise_sd=0.1, rollouts=100)).signal
        for _ in range(4000)
    ]
    # Removing only var_i(d[c]) / n, without the K/(K-1), would leave a 2.5e-5 bias.
    assert abs(np.mean(signals)) < 5e-6


def test_read_file_round_trips_the_sidecars(tmp_path):
    rollouts, k = 4, 2
    floats = np.arange(2 * k * rollouts, dtype="<f4")
    floats.tofile(tmp_path / "a.trollouts")
    doc = {
        "rollouts": rollouts,
        "sim_seconds": 2.0,
        "threads": 1,
        "positions": [
            {
                "game": 0,
                "turn": 3,
                "float_offset": 0,
                "candidates": [{"stratum": "top"}, {"stratum": "exchange"}],
                "couplings": [{"a": 0, "b": 1, "kind": "play_exchange"}],
                "offered_couplings": {"play_exchange": 2},
                "saturation": [[[1, 10], [2, 14]], [[1, 9], [2, 9]]],
            }
        ],
    }
    (tmp_path / "a.tmeasure").write_text(json.dumps(doc))
    files = tm.read_store(tmp_path)
    p = files[0].positions[0]
    assert p.expected.tolist() == [[0, 2, 4, 6], [8, 10, 12, 14]]
    assert p.delta.tolist() == [[1, 3, 5, 7], [9, 11, 13, 15]]
    assert tm.rollouts_per_second(files) == pytest.approx(4.0)
    assert tm.saturation_by_stratum(files) == {
        "top": {1: [10], 2: [14]},
        "exchange": {1: [9], 2: [9]},
    }


def test_rows_in_selects_candidates_by_stratum():
    position = tm.MeasuredPosition(
        game=0,
        turn=0,
        strata=["top", "exchange", "middle"],
        expected=np.zeros((3, 2)),
        delta=np.zeros((3, 2)),
        couplings=[],
        offered={},
        saturation=[],
    )
    values = np.arange(6.0).reshape(3, 2)
    assert tm.rows_in(position, values, {"top", "middle"}).tolist() == [[0, 1], [4, 5]]


def test_decided_positions_are_the_ones_without_any_spread():
    def position(expected):
        return tm.MeasuredPosition(
            game=0,
            turn=0,
            strata=["top"] * expected.shape[0],
            expected=expected,
            delta=np.zeros_like(expected),
            couplings=[],
            offered={},
            saturation=[],
        )

    assert tm.decided(position(np.ones((3, 5))))
    assert not tm.decided(position(np.array([[1.0, 0.0], [1.0, 1.0]])))

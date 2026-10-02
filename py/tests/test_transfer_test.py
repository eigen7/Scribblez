"""The transfer_test workload's wiring and its step-0 analysis
(scribblez/transfer_test_measure.py)."""

import json

import numpy as np
import pytest
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
    ratios = {
        100: np.array([0.3, 0.2]),
        1000: np.array([0.05, 0.08]),
        10000: np.array([0.01, 0.01]),
    }
    assert tm.required_rollouts(ratios, target=0.1, quantile=0.5) == 1000
    assert tm.required_rollouts(ratios, target=0.001, quantile=0.5) is None


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
                "saturation": {"0": [[1, 10], [2, 14]], "1": [[1, 9], [2, 9]]},
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
        saturation={},
    )
    values = np.arange(6.0).reshape(3, 2)
    assert tm.rows_in(position, values, {"top", "middle"}).tolist() == [[0, 1], [4, 5]]

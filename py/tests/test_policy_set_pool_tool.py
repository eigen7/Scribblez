"""policy_set_pool_tool end to end (docs/plans/deep_bot_v0.md, "Data: labeled
pools"), on its CPU reference with the synthetic leaf, since the suites hide
the GPU: HastyBot games in, a .pool sidecar out, whose labels form a
distribution over each position's pool, raced in rounds, independent of how
rounds share launches, and whose header names the leaf. The GPU path
(gpu_rollout_demo's checks) and the leave inference path need a GPU, so only
the latter's refusal is checked here."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
TOOL = _ENGINE_DIR / "policy_set_pool_tool"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file() or not TOOL.is_file():
    pytest.skip("engine not built -- run py/build.py first", allow_module_level=True)
if not LEAVES.exists():
    pytest.skip("HastyBot leave values not installed", allow_module_level=True)

from scribblez import ffi  # noqa: E402
from scribblez.policy_set.pools import read_pools  # noqa: E402
from scribblez.selfplay import run_games  # noqa: E402

ROLLOUTS = 20
MOVE_TYPES = ffi.format_layout()["constants"]["move_type"]
CPU = ["--backend", "cpu", "--leaf-model", "synthetic"]


def _label(slog_dir: Path, *flags: str) -> subprocess.CompletedProcess:
    cmd = [str(TOOL), "--slog-dir", str(slog_dir), *CPU, "--positions-per-game", "1"]
    return subprocess.run(
        [*cmd, "--rollouts", str(ROLLOUTS), "--threads", "4", *flags],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def games(tmp_path_factory) -> Path:
    games = tmp_path_factory.mktemp("pool") / "games"
    assert (
        run_games(games, num_games=2, threads=2, player_spec="--type=hastybot-endgame", seed=5) == 0
    )
    return games


@pytest.fixture(scope="module")
def pooled(games) -> Path:
    run = _label(games)
    assert run.returncode == 0, run.stderr
    (slog,) = games.glob("*.slog")
    return slog


def test_header_names_the_sources(pooled):
    hdr, positions = read_pools(pooled.with_suffix(".pool"))
    assert hdr["flags"] == 0  # uniform opponent leaves
    assert hdr["rollouts"] == ROLLOUTS
    assert hdr["leaf_model_tag"] == b"synthetic" and hdr["leaf_model_hash"] == b""
    assert len(positions) == 2  # one per game


def test_labels_are_a_distribution_over_each_pool(pooled):
    _, positions = read_pools(pooled.with_suffix(".pool"))
    for p in positions:
        assert len(p.moves) > 1
        np.testing.assert_allclose(p.q.sum(), 1.0, atol=1e-5)
        assert p.q.max() < 1 and p.q.std() > 0
        assert ((p.win_prob >= 0) & (p.win_prob <= 1)).all()
        assert (p.moves["type"] == MOVE_TYPES["pass"]).sum() == 1
        assert ((p.rollouts >= 1) & (p.rollouts <= ROLLOUTS)).all()
        leader = int(np.argmax(p.q))
        assert p.rollouts[leader] == ROLLOUTS and p.paired_se[leader] == 0
        assert p.win_prob[leader] == p.win_prob[p.rollouts == ROLLOUTS].max()


def test_rounds_race_the_candidates(pooled):
    _, positions = read_pools(pooled.with_suffix(".pool"))
    # Rounds end at 2, 4, 8 and 14 of 20 rollouts, and some candidates stop at
    # a round's end; never the pass, the baseline of the pool's stakes.
    counts = np.concatenate([p.rollouts for p in positions])
    assert (counts < ROLLOUTS).any()
    for p in positions:
        assert p.rollouts[p.moves["type"] == MOVE_TYPES["pass"]] == ROLLOUTS
    assert set(np.unique(counts)) <= {2, 4, 8, 14, ROLLOUTS}
    win = np.concatenate([p.win_prob for p in positions])
    assert win.std() > 0 and ((win > 0) & (win < 1)).mean() > 0.5


def test_labels_do_not_depend_on_how_rounds_share_launches(games, pooled, tmp_path):
    other = tmp_path / "games"
    other.mkdir()
    (other / pooled.name).write_bytes(pooled.read_bytes())
    run = _label(other, "--launch-tasks", "64", "--max-in-flight", "1")
    assert run.returncode == 0, run.stderr
    assert (other / pooled.name).with_suffix(".pool").read_bytes() == pooled.with_suffix(
        ".pool"
    ).read_bytes()


def test_unraced_every_candidate_runs_the_budget(pooled, tmp_path):
    other = tmp_path / "games"
    other.mkdir()
    (other / pooled.name).write_bytes(pooled.read_bytes())
    run = _label(other, "--race", "false")
    assert run.returncode == 0, run.stderr
    _, positions = read_pools((other / pooled.name).with_suffix(".pool"))
    assert all((p.rollouts == ROLLOUTS).all() for p in positions)


def test_a_finished_file_is_skipped(pooled):
    pool = pooled.with_suffix(".pool")
    before = pool.stat().st_mtime_ns
    assert _label(pooled.parent).returncode == 0
    assert pool.stat().st_mtime_ns == before


def test_a_finished_file_from_other_settings_is_refused(pooled):
    pool = pooled.with_suffix(".pool")
    before = pool.stat().st_mtime_ns
    cmd = [str(TOOL), "--slog-file", str(pooled), *CPU, "--rollouts", str(ROLLOUTS + 1)]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert run.returncode != 0 and "another rollout budget" in run.stderr
    assert pool.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    ("flags", "error"),
    [
        (["--leaf-model", "synthetic"], "needs --backend cpu"),
        (["--backend", "cpu", "--leaf-model", "leaf.onnx"], "needs --leaf-model synthetic"),
        (["--backend", "cpu"], "--leaf-model is required"),
        ([*CPU, "--rollouts", "0"], "--rollouts must be"),
        ([*CPU, "--blank-prune-points", "-1"], "--blank-prune-points must be"),
    ],
)
def test_bad_flag_combinations_are_refused(pooled, flags, error):
    run = subprocess.run(
        [str(TOOL), "--slog-file", str(pooled), *flags], capture_output=True, text=True, check=False
    )
    assert run.returncode != 0 and error in run.stderr


def test_leave_inference_needs_forgone_sidecars(pooled, tmp_path):
    """The refusal comes before the model loads, so no GPU or model is needed."""
    model = tmp_path / "model.onnx"  # never read
    cmd = [str(TOOL), "--slog-file", str(pooled), *CPU, "--leave-inference", str(model)]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert run.returncode != 0 and "forgone_play_tool" in run.stderr

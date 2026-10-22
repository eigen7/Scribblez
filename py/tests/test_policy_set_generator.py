"""The policy_set workload's generate side (scribblez.workloads.policy_set) and
its pool reader (scribblez.policy_set.pools): pinning the source generation, the
role inputs, a chunk's survey into a .pool, and reading pools back into model
inputs."""

import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez import ffi, params  # noqa: E402
from scribblez.leave_inference import win_table  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.policy_set import pools  # noqa: E402
from scribblez.selfplay import run_games  # noqa: E402
from scribblez.workloads import policy_set, selfplay_gen  # noqa: E402
from scribblez.workloads.policy_set import (  # noqa: E402
    BELIEFS_INFERRED,
    BELIEFS_UNIFORM,
    MODEL_INPUT,
    WIN_TABLE_INPUT,
    PolicySetParams,
)


def _source_tag(mount: Path, tables: list[int], models: list[int]) -> TagPaths:
    """A leave_inference tag holding win tables and exports for those
    generations."""
    paths = TagPaths("li", policy_set.LEAVE_INFERENCE, mount)
    for g in tables:
        paths.win_table_path(g).parent.mkdir(parents=True, exist_ok=True)
        paths.win_table_path(g).write_text("table")
    for g in models:
        paths.onnx_path(g).write_bytes(b"onnx")
    return paths


def _pool_tag(mount: Path) -> TagPaths:
    return TagPaths("pools", "policy_set", mount)


def test_the_source_pins_its_latest_usable_generation(tmp_path):
    _source_tag(tmp_path, tables=[3, 5], models=[3])
    inferred = PolicySetParams(leave_inference_tag="li")
    assert policy_set.finalize(None, _pool_tag(tmp_path), inferred).leave_inference_generation == 3
    uniform = replace(inferred, leave_beliefs=BELIEFS_UNIFORM)
    assert policy_set.finalize(None, _pool_tag(tmp_path), uniform).leave_inference_generation == 5


@pytest.mark.parametrize(
    "changes",
    [
        {"leave_inference_tag": ""},
        {"leave_inference_generation": 5},  # a table but no model
        {"leave_inference_generation": 9},  # neither
        {"leave_inference_generation": -2},
        {"leave_inference_tag": "missing"},
    ],
)
def test_an_unusable_source_is_refused_at_task_creation(tmp_path, changes):
    _source_tag(tmp_path, tables=[3, 5], models=[3])
    p = replace(PolicySetParams(leave_inference_tag="li"), **changes)
    with pytest.raises(params.ParamsError):
        policy_set.finalize(None, _pool_tag(tmp_path), p)


def test_inputs_ship_the_model_only_under_inferred_beliefs(tmp_path):
    src = _source_tag(tmp_path, tables=[3], models=[3])
    p = PolicySetParams(leave_inference_tag="li", leave_inference_generation=3)
    assert policy_set.inputs(p, tmp_path) == {
        WIN_TABLE_INPUT: src.win_table_path(3),
        MODEL_INPUT: src.onnx_path(3),
    }
    uniform = replace(p, leave_beliefs=BELIEFS_UNIFORM)
    assert policy_set.inputs(uniform, tmp_path) == {WIN_TABLE_INPUT: src.win_table_path(3)}


def test_inferred_beliefs_write_forgone_sidecars_and_pass_the_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(policy_set.forgone, "write_sidecars", lambda d, t: calls.append(("fg", d)))
    monkeypatch.setattr(policy_set.subprocess, "run", lambda cmd, check: calls.append(cmd))
    (tmp_path / "c.slog").write_bytes(b"")
    p = PolicySetParams(leave_beliefs=BELIEFS_INFERRED)
    policy_set.write_pools(p, Path("t.csv"), Path("m.onnx"), tmp_path, threads=3)
    assert calls[0] == ("fg", tmp_path)
    cmd = calls[1]
    assert cmd[cmd.index("--leave-inference") + 1] == "m.onnx"
    assert cmd[cmd.index("--horizon-plies") + 1] == "3"
    assert cmd[cmd.index("--rollouts") + 1] == "100"


def test_games_per_chunk_is_bounded_by_play_games_file_size(tmp_path):
    with pytest.raises(ValueError, match="games_per_chunk"):
        selfplay_gen.generate(None, "--type=greedy", False, 0.0, games_per_chunk=1001)


@pytest.fixture(scope="module")
def surveyed_chunk(tmp_path_factory) -> Path:
    """A one-game chunk surveyed under uniform beliefs, with a logistic win
    table, as a generator cycle leaves it. Shared by the module's tests: a
    pool sims every candidate of a position, the bulk of this module's time."""
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    tmp = tmp_path_factory.mktemp("chunk")
    chunk = tmp / "chunk"
    assert run_games(chunk, num_games=1, threads=2, player_spec="--type=hastybot-endgame") == 0
    table = tmp / "table.csv"
    spread = np.arange(-win_table.max_spread(), win_table.max_spread() + 1)
    grid = np.repeat(1 / (1 + np.exp(-spread / 30))[:, None], win_table.max_unseen() + 1, axis=1)
    win_table.write(table, grid, {"tag": "li", "generation": 3})
    p = PolicySetParams(leave_beliefs=BELIEFS_UNIFORM, rollouts=8, positions_per_game=1)
    policy_set.write_pools(p, table, None, chunk, threads=4)
    return chunk


def test_a_chunk_surveys_into_pools_that_read_back_as_model_inputs(surveyed_chunk):
    (slog,) = surveyed_chunk.glob("*.slog")
    hdr, read = pools.read_pools(slog.with_suffix(pools.SUFFIX))
    assert hdr["flags"] == 0 and hdr["horizon_plies"] == 3 and hdr["rollouts"] == 8
    assert len(read) == 1  # one position per game
    for pool in read:
        assert len(pool.moves) > 1
        np.testing.assert_allclose(pool.q.sum(), 1.0, atol=1e-5)
        # Uniform beliefs: a full rack drawn uniformly, holding 1 to 7 kinds.
        assert not pool.opp_leave_modelled and pool.opp_leave_value_mean == 0
        assert ((pool.opp_rack_presence >= 0) & (pool.opp_rack_presence <= 1)).all()
        assert 1 <= pool.opp_rack_presence.sum() <= 7
        # The stored rollout values are what each candidate's win_prob averages.
        assert len(pool.rollout_values) == pool.rollouts.sum()
        n = pool.rollouts.astype(np.int64)
        means = np.add.reduceat(pool.rollout_values, np.cumsum(n) - n) / n
        np.testing.assert_allclose(means, pool.win_prob, atol=1e-5)
    inputs = pools.pool_inputs(slog, read)
    assert inputs["rows"].shape == (1, ffi.task_row_floats())
    assert len(inputs["cells"]) == len(inputs["letters"]) == sum(len(p.moves) for p in read)


def test_a_damaged_pool_file_is_refused(surveyed_chunk, tmp_path):
    (pool,) = surveyed_chunk.glob("*" + pools.SUFFIX)
    data = pool.read_bytes()
    truncated = tmp_path / "t.pool"
    truncated.write_bytes(data[:-1])
    with pytest.raises(ValueError, match="truncated"):
        pools.read_pools(truncated)
    stale = bytearray(data)
    stale[4] += 1  # the version field, after the 4-byte magic
    other = tmp_path / "v.pool"
    other.write_bytes(bytes(stale))
    with pytest.raises(ValueError, match="version"):
        pools.read_pools(other)


def test_survey_failure_fails_the_cycle(tmp_path):
    (tmp_path / "c.slog").write_bytes(b"not a slog")
    p = PolicySetParams(leave_beliefs=BELIEFS_UNIFORM)
    with pytest.raises(subprocess.CalledProcessError):
        policy_set.write_pools(p, tmp_path / "no-table.csv", None, tmp_path, threads=1)


def test_replayed_equities_order_the_pool_as_its_equity_ranks_do(surveyed_chunk):
    """ffi.move_set_positions' equities are the survey's static equities: along
    each pool's equity ranks they never increase."""
    (slog,) = surveyed_chunk.glob("*.slog")
    _, read = pools.read_pools(slog.with_suffix(pools.SUFFIX))
    for pool in read:
        got = ffi.move_set_positions(
            slog, np.array([pool.game]), np.array([pool.turn]), np.array([len(pool.moves)]),
            pool.moves, equities=True,
        )["equities"]  # fmt: skip
        ranked = pool.equity_rank >= 0
        along = got[ranked][np.argsort(pool.equity_rank[ranked])]
        assert (np.diff(along) <= 1e-4).all()

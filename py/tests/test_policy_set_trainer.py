"""The policy_set workload's trainer side: the set metrics
(scribblez.policy_set.metrics), the baseline-scoring generation loop
(scribblez.policy_set.trainer), and the workload's registration."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez import ffi, workloads  # noqa: E402
from scribblez.generational import lifecycle  # noqa: E402
from scribblez.paths import TagPaths  # noqa: E402
from scribblez.policy_set import metrics, pools, trainer  # noqa: E402
from scribblez.workloads.policy_set import PolicySetParams  # noqa: E402


def _pool(q, win, rank) -> pools.Pool:
    n = len(q)
    return pools.Pool(
        game=0,
        turn=0,
        num_legal_moves=n,
        moves=np.zeros(n, ffi.struct_dtype("Move")),
        win_prob=np.array(win, np.float32),
        paired_se=np.zeros(n, np.float32),
        q=np.array(q, np.float32),
        equity_rank=np.array(rank, np.int32),
        rollouts=np.full(n, 8, np.uint16),
    )


# The leader (largest q) is equity rank 3 in A, and rank 12 (the tail) in B.
POOL_A = _pool([0.1, 0.6, 0.3], [0.40, 0.50, 0.45], [0, 3, 1])
POOL_B = _pool([0.7, 0.3], [0.60, 0.55], [12, 0])


def test_the_equity_baseline_metrics_by_set_size():
    m = metrics.set_metrics([POOL_A, POOL_B], metrics.equity_top, "equity_")
    assert m["equity_eval_recall_1"] == 0
    assert m["equity_eval_recall_4"] == 0.5  # A's leader enters at rank 3; B's never by 4
    assert m["equity_eval_recall_16"] == 1
    assert m["equity_eval_cover_1"] == pytest.approx((0.1 + 0.3) / 2)
    assert m["equity_eval_regret_1"] == pytest.approx((0.10 + 0.05) / 2)
    assert m["equity_eval_regret_16"] == pytest.approx(0)
    assert (m["equity_eval_tail_recall_4"], m["equity_eval_tail_recall_16"]) == (0, 1)


@pytest.mark.parametrize("bad", [lambda p, n: np.array([], int), lambda p, n: np.arange(n + 1)])
def test_a_chooser_must_return_a_set_of_at_most_n(bad):
    with pytest.raises(ValueError):
        metrics.set_metrics([_pool([0.5] * 20, [0.5] * 20, list(range(20)))], bad, "")


def test_pool_stats():
    s = metrics.pool_stats([POOL_A, POOL_B], "eval_")
    assert (s["eval_pools"], s["eval_pool_size"], s["eval_tail_share"]) == (2, 2.5, 0.5)
    assert s["eval_label_entropy"] > 0


def _write_pool_file(path: Path, pool_list: list[pools.Pool]):
    """A .pool file of `pool_list`, laid out as the engine writes one."""
    hdr = np.zeros(1, ffi.struct_dtype("PoolFileHeader"))
    hdr["magic"], hdr["version"] = pools.POOL["magic"], pools.POOL["version"]
    hdr["num_positions"] = len(pool_list)
    parts = [hdr.tobytes()]
    for p in pool_list:
        ph = np.zeros(1, ffi.struct_dtype("PoolPositionHeader"))
        ph["num_candidates"] = len(p.q)
        c = np.zeros(len(p.q), ffi.struct_dtype("PoolCandidate"))
        c["q"], c["win_prob"], c["equity_rank"] = p.q, p.win_prob, p.equity_rank
        parts += [ph.tobytes(), c.tobytes()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(parts))


def _complete(dir_: Path):
    lifecycle.write_manifest(dir_, {"status": lifecycle.GENERATING})
    lifecycle.mark_complete(dir_, 1)


def test_each_generation_records_the_baseline_and_advances_the_cursor(tmp_path):
    paths = TagPaths("t", "policy_set", tmp_path)
    _write_pool_file(paths.generation_dir(0) / "a.pool", [POOL_A, POOL_B, POOL_A])
    _complete(paths.generation_dir(0))
    _write_pool_file(paths.held_out_chunks_dir / "h.pool", [POOL_A, POOL_B])
    committed = []
    recorder = SimpleNamespace(
        commit_generation=lambda g, rows, rec: committed.append((g, rows, rec))
    )
    home = SimpleNamespace(remote=False, check=lambda: None)
    stats = SimpleNamespace(cycle_done=lambda phases, units, nbytes: None)
    params = PolicySetParams(max_rows=1)
    trainer.run_generations(paths, params, recorder, home, sink=None, stats=stats)
    ((gen, rows, record),) = committed
    assert (gen, rows) == (0, 3)
    assert (record["pools"], record["eval_pools"]) == (3, 2)
    assert record["equity_eval_recall_16"] == 1
    assert lifecycle.read_train_state(paths) == {"generation_index": 1, "rows_trained": 3}


def test_the_workload_registers_a_gpu_generator_and_a_data_home_trainer():
    spec = workloads.get("policy_set")
    assert spec.role("generate").gpu
    assert spec.role("train").singleton and spec.role("train").ingest
    assert spec.finalize and spec.role("generate").inputs

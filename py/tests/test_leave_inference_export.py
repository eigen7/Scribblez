"""Serving the leave inference model: the engine's LeaveSampler reproduces the
trainer's tilted NLL from the exported 29 floats, and the ONNX export carries
the metadata the engine gates on. The TensorRT side is
engine/tests/test_leave_inference_parity.cpp."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "target" / "engine"
LEAVES = Path("/workspace/mount/macondo/data/strategy/NWL23/leaves.klv2")

if not (_ENGINE_DIR / "libscribblez_ffi.so").is_file():
    pytest.skip("libscribblez_ffi.so not built -- run py/build.py first", allow_module_level=True)

from scribblez.ffi import LeaveValueArrays, task_input_shapes  # noqa: E402
from scribblez.leave_inference import loss  # noqa: E402
from scribblez.leave_inference.heads import HEAD_AUTOREGRESSIVE, HEAD_TILTED  # noqa: E402
from scribblez.leave_inference.model import LeaveInferenceModel  # noqa: E402
from scribblez.leave_inference.onnx_export import (  # noqa: E402
    FORMAT_KEY,
    GRAPH,
    TABLE_KEY,
    export_onnx,
    params_in_points,
)

# A full bag's tile counts (A..Z, blank).
FULL_BAG = [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2]
SHAPES = {s.name: s.dims for s in task_input_shapes("leave_inference")}


@pytest.fixture(scope="module")
def arrays() -> LeaveValueArrays:
    if not LEAVES.exists():
        pytest.skip("HastyBot leave values not installed")
    return LeaveValueArrays()


@pytest.fixture(scope="module")
def table(arrays) -> loss.TableTensors:
    return loss.TableTensors(arrays, "cpu")


def _pool_and_leaves(k: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """A mid-game pool (60 tiles of a full bag) and `n` length-k leaves drawn
    from it."""
    unseen = rng.choice(np.repeat(np.arange(27), FULL_BAG), size=60, replace=False)
    leaves = [
        np.bincount(rng.choice(unseen, size=k, replace=False), minlength=27) for _ in range(n)
    ]
    return np.bincount(unseen, minlength=27), np.array(leaves)


@pytest.mark.parametrize("k", [2, 4])
def test_sampler_matches_the_trainers_nll(arrays, table, k):
    rng = np.random.default_rng(k)
    pool, leaves = _pool_and_leaves(k, 5, rng)
    torch.manual_seed(k)
    theta, a, b = torch.randn(1, 27), torch.randn(1), torch.rand(1)
    n = len(leaves)
    want = loss.tilted_nll(
        table,
        theta.expand(n, -1),
        a.expand(n),
        b.expand(n),
        torch.from_numpy(leaves),
        torch.full((n,), k),
        torch.from_numpy(pool).expand(n, -1),
    ).nll.numpy()
    params = params_in_points(theta, a, b)[0].numpy()
    _, log_probs, _ = arrays.sampler_query(params, k, pool, rows=table.row_index(k, leaves))
    np.testing.assert_allclose(-log_probs, want, rtol=1e-5, atol=2e-4)


def _model(head: str) -> LeaveInferenceModel:
    torch.manual_seed(0)
    return LeaveInferenceModel(
        SHAPES["input_spatial"][0], SHAPES["input_scalar"][0], 8, 1, head=head
    )


def test_export_stamps_the_gated_entries_and_keeps_the_mode(tmp_path):
    model = _model(HEAD_TILTED).train()
    path = tmp_path / "model.onnx"
    export_onnx(
        model,
        path,
        SHAPES["input_spatial"][0],
        SHAPES["input_scalar"][0],
        leave_values="abc",
        provenance={"tag": "t", "generation": 7, "policy": "hasty"},
    )
    meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    assert meta["graph"] == GRAPH
    assert int(meta[FORMAT_KEY]) >= 1
    assert meta[TABLE_KEY] == "abc"
    assert (meta["head"], meta["tag"], meta["generation"]) == (HEAD_TILTED, "t", "7")
    assert model.training
    model.eval()
    export_onnx(
        model,
        path,
        SHAPES["input_spatial"][0],
        SHAPES["input_scalar"][0],
        leave_values="abc",
        provenance={},
    )
    assert not model.training


def test_only_the_tilted_head_exports(tmp_path):
    with pytest.raises(ValueError, match="tilted"):
        export_onnx(
            _model(HEAD_AUTOREGRESSIVE),
            tmp_path / "model.onnx",
            SHAPES["input_spatial"][0],
            SHAPES["input_scalar"][0],
            leave_values="abc",
            provenance={},
        )

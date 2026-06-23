"""Tests for the streaming self-play training source and row-slicing helper.

Covers:
  1. slice_row_batch matches SlogDataset's slicing (guards the DRY extraction).
  2. StreamingTrainSource smoke: pull a few slots, check shapes/finiteness,
     release, and stop cleanly.
"""

from pathlib import Path

import numpy as np
import pytest

from scribblez.dataset import row_layout, slice_row_batch
from scribblez.ffi import get_input_shapes, get_target_shapes, row_size_floats

_FFI_LIB = Path("/workspace/repo/target/engine/libscribblez_ffi.so")


def _require_engine():
    if not _FFI_LIB.is_file():
        pytest.skip("libscribblez_ffi.so not built -- run py/build.py first")


def test_slice_row_batch_matches_dataset():
    """slice_row_batch reproduces the named tensors with correct shapes/values."""
    _require_engine()
    rf = row_size_floats()
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((5, rf)).astype(np.float32)

    input_shapes, targets = row_layout()
    out = slice_row_batch(batch, input_shapes, targets)

    # Every input + target tensor is present with the advertised shape.
    expected = {s.name: (5, *s.dims) for s in get_input_shapes()}
    expected.update({s.name: (5, *s.dims) for s in get_target_shapes()})
    assert set(out) == set(expected)
    for name, shape in expected.items():
        assert tuple(out[name].shape) == shape

    # The flat concatenation of all regions reconstructs the original row.
    flat = np.concatenate(
        [out[s.name].numpy().reshape(5, -1) for s in get_input_shapes()]
        + [out[name].numpy().reshape(5, -1) for name, *_ in targets],
        axis=1,
    )
    assert np.array_equal(flat, batch)


def test_streaming_source_smoke():
    """Pull a few full slots, verify shape/finiteness, release, and stop."""
    _require_engine()
    from scribblez.ffi import StreamingTrainSource

    bs = 32
    src = StreamingTrainSource(
        batch_size=bs, num_slots=2, num_threads=2, seed=7, handicap_max=0, pin_memory=False
    )
    src.start()
    rf = row_size_floats()
    try:
        for _ in range(4):
            res = src.next_slot()
            assert res is not None
            idx, tensor = res
            arr = tensor.numpy().copy()
            src.release(idx)
            assert arr.shape == (bs, rf)
            assert np.isfinite(arr).all()
        st = src.stats()
        assert st["rows_committed"] >= bs  # at least one full slot produced
        assert st["games_played"] >= st["rows_committed"]
    finally:
        src.stop()


def test_streaming_source_anchor_planes_populated():
    """A streamed row carries non-zero per-letter anchor planes (encoder ran)."""
    _require_engine()
    from scribblez.ffi import StreamingTrainSource

    bs = 32
    src = StreamingTrainSource(batch_size=bs, num_slots=2, num_threads=2, seed=3, pin_memory=False)
    src.start()
    try:
        res = src.next_slot()
        assert res is not None
        idx, tensor = res
        arr = tensor.numpy().copy()
        src.release(idx)
        spatial = arr[:, : 85 * 225].reshape(bs, 85, 15, 15)
        assert (spatial[:, 33:85] != 0).any()  # horizontal+vertical anchor planes
    finally:
        src.stop()

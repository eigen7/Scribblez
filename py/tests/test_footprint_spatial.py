"""The footprint spatial reshape + sparse top-k codec."""

import numpy as np
import pytest
from scribblez import footprint_spatial as fs


def test_constants_factor_the_grid():
    assert fs.ANCHORED == fs.SIDE * fs.SIDE * fs.SLOTS_PER_CELL
    assert fs.CATCH_ALL == 2
    assert fs.NUM_CLASSES == fs.ANCHORED + fs.CATCH_ALL


def test_spatial_roundtrip_and_cell_major_layout():
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((3, fs.NUM_CLASSES)).astype(np.float32)
    anchored, catch = fs.to_spatial(dense)
    assert anchored.shape == (3, fs.SIDE, fs.SIDE, fs.SLOTS_PER_CELL)
    assert catch.shape == (3, fs.CATCH_ALL)
    np.testing.assert_array_equal(fs.from_spatial(anchored, catch), dense)

    # Cell-major, slot-minor: anchored class c -> (r, col, slot).
    c = (7 * fs.SIDE + 5) * fs.SLOTS_PER_CELL + 4
    onehot = np.zeros(fs.NUM_CLASSES, np.float32)
    onehot[c] = 1.0
    grid, _ = fs.to_spatial(onehot)
    assert grid[7, 5, 4] == 1.0
    assert grid.sum() == 1.0


def test_top_k_exact_when_k_covers_support():
    rng = np.random.default_rng(1)
    dense = np.zeros((4, fs.NUM_CLASSES), np.float32)
    for row in dense:  # 10 nonzero classes per row
        row[rng.choice(fs.NUM_CLASSES, 10, replace=False)] = rng.random(10).astype(np.float32)
    idx, val = fs.top_k_sparse(dense, 32)
    assert idx.shape == val.shape == (4, 32)
    # k >= support -> scatter reconstructs exactly.
    np.testing.assert_allclose(fs.scatter_sparse(idx, val), dense, atol=0)
    # Descending, distinct indices per row.
    assert np.all(np.diff(val, axis=-1) <= 0)
    for r in range(4):
        assert len(set(idx[r].tolist())) == 32


def test_top_k_truncates_the_tail():
    dense = np.zeros((1, fs.NUM_CLASSES), np.float32)
    dense[0, :5] = np.array([5, 4, 3, 2, 1], np.float32)
    idx, val = fs.top_k_sparse(dense, 3)
    np.testing.assert_array_equal(idx[0], [0, 1, 2])
    np.testing.assert_array_equal(val[0], [5, 4, 3])
    assert fs.top_k_mass(dense, 3)[0] == pytest.approx(12 / 15)
    assert fs.top_k_mass(dense, 5)[0] == pytest.approx(1.0)


def test_top_k_mass_handles_zero_rows():
    assert fs.top_k_mass(np.zeros((1, fs.NUM_CLASSES), np.float32), 4)[0] == 1.0

"""Footprint placement classes as a spatial tensor, plus a sparse top-k codec.

A footprint class (engine/include/training/footprint.h) describes where the next
move's tiles go: an anchor cell, an orientation, and a tile count. There are
``side*side*slots_per_cell`` anchored classes (225 cells x 13 slots = 2925),
followed by two non-spatial catch-all classes: ``pass`` (no placement), then
the win heads' not-win class. An anchored class index is ``(r*side + c)*slots + slot``, so the
anchored block reshapes losslessly to ``(side, side, slots)``: a 15x15 grid with
13 channels per square. The teacher target, the sim observation, the student
head and evidence fusion all use this layout.

Frame invariant: every placement path uses the game's natural frame, with no
symmetry transpose. A diagonal transpose swaps rows with columns and also the
horizontal with the vertical slot channels (footprint.h), so a transposed
consumer would have to permute the 13 channels as well as H and W.

The sparse codec keeps each distribution's top-k classes as fixed-size
``(index, value)`` pairs. No on-disk format uses it; it serves offline analysis
(scripts/position_eval/footprint_topk_fidelity.py).
"""

import numpy as np

from scribblez.ffi import format_layout

_F = format_layout()["constants"]["footprint"]
SIDE = _F["side"]  # 15
SLOTS_PER_CELL = _F["slots_per_cell"]  # 13
MAX_K = _F["max_k"]  # 7 (kFootprintMaxK: the largest tile count a slot encodes)
ANCHORED = _F["anchored"]  # 2925
NUM_CLASSES = _F["num_classes"]  # 2927
PASS_CLASS = _F["pass_class"]  # 2925
EXTRA_CLASS = _F["extra_class"]  # 2926
CATCH_ALL = NUM_CLASSES - ANCHORED  # 2

assert ANCHORED == SIDE * SIDE * SLOTS_PER_CELL, "footprint constants inconsistent with FFI"


def to_spatial(dense):
    """Split a ``(..., NUM_CLASSES)`` array into a ``(..., SIDE, SIDE, SLOTS)``
    anchored block and a ``(..., CATCH_ALL)`` catch-all block. Inverse of
    ``from_spatial``."""
    dense = np.asarray(dense)
    lead = dense.shape[:-1]
    anchored = dense[..., :ANCHORED].reshape(*lead, SIDE, SIDE, SLOTS_PER_CELL)
    return anchored, dense[..., ANCHORED:]


def from_spatial(anchored, catch_all):
    """Recombine a ``(..., SIDE, SIDE, SLOTS)`` anchored block and a
    ``(..., CATCH_ALL)`` catch-all block into a ``(..., NUM_CLASSES)`` dense
    array. Inverse of ``to_spatial``."""
    anchored = np.asarray(anchored)
    catch_all = np.asarray(catch_all)
    flat = anchored.reshape(*anchored.shape[:-3], ANCHORED)
    return np.concatenate([flat, catch_all], axis=-1)


def to_slot_planes(dense):
    """A ``(..., NUM_CLASSES)`` array as channels-first
    ``(..., SLOTS_PER_CELL, SIDE, SIDE)`` board maps, dropping the catch-all
    classes."""
    anchored, _ = to_spatial(dense)  # (..., SIDE, SIDE, SLOTS)
    return np.moveaxis(anchored, -1, -3)


def top_k_sparse(dense, k):
    """The ``k`` largest entries of each row of a ``(..., n)`` array, as
    ``(indices (..., k) int32, values (..., k) float32)`` sorted descending.
    The indices are distinct, so ``scatter_sparse`` reconstructs the row exactly
    when ``k`` covers its nonzero support. Past that support, entries carry a
    real index with value 0."""
    dense = np.asarray(dense, dtype=np.float32)
    n = dense.shape[-1]
    k = min(k, n)
    part = np.argpartition(-dense, k - 1, axis=-1)[..., :k]  # k largest, unordered
    part_vals = np.take_along_axis(dense, part, axis=-1)
    order = np.argsort(-part_vals, axis=-1)  # sort those k descending
    indices = np.take_along_axis(part, order, axis=-1).astype(np.int32)
    values = np.take_along_axis(part_vals, order, axis=-1).astype(np.float32)
    return indices, values


def scatter_sparse(indices, values, n=NUM_CLASSES):
    """Dense ``(..., n)`` reconstructed from ``top_k_sparse`` output; unlisted
    classes are 0."""
    indices = np.asarray(indices)
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros((*indices.shape[:-1], n), dtype=np.float32)
    np.put_along_axis(out, indices.astype(np.intp), values, axis=-1)
    return out


def top_k_mass(dense, k):
    """Fraction of each row's total mass carried by its top-k entries. Rows
    summing to 0 report 1.0."""
    dense = np.asarray(dense, dtype=np.float64)
    total = dense.sum(axis=-1)
    _, values = top_k_sparse(dense, k)
    kept = values.sum(axis=-1)
    return np.where(total > 0, kept / np.where(total > 0, total, 1.0), 1.0)

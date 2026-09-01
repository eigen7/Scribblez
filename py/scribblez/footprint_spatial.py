"""Footprint placement as a (side, side, slots) spatial tensor + catch-all, and a
sparse top-k codec.

A placement class (engine/include/training/footprint.h) is (anchor, orientation,
k). The ANCHORED classes number ``side*side*slots_per_cell`` (2925 = 225 cells ×
13 slots), followed by two non-spatial catch-all classes: ``pass`` then the win
heads' not-win/dummy. Because an anchored class factors as ``(cell, slot)`` with
``cell = r*side + c`` (slot minor), the anchored block reshapes losslessly to
``(side, side, slots)`` -- a 15×15 board grid with 13 channels per square. This is
the representation the footprint-native placement path shares across the teacher
target, the sim observation, the student head, and the evidence fusion.

FRAME INVARIANT: every placement path is pinned to the unflipped (``flip=false``)
frame. A diagonal transpose swaps rows↔cols AND the horizontal↔vertical slot
channels (footprint.h), so the SLOT axis -- not only the spatial axes -- permutes
under a flip. Any future flipped consumer must permute the 13 channels too, not
just H/W.

The sparse codec keeps the top-k classes of a per-head distribution as fixed
padded ``(index, value)`` pairs. Both on-disk formats (``.mset``, ``.sobs``)
ended up dense -- the teacher distribution is too broad for any small k, and the
sim observation stayed a verbatim POD -- so the codec currently has no on-disk
consumer; it remains for offline analysis (the fidelity probe). It is generic
over the class count, so both the anchored+catch-all distribution and any
sub-block can use it.
"""

import numpy as np

from scribblez.ffi import format_layout

_F = format_layout()["constants"]["footprint"]
SIDE = _F["side"]  # 15
SLOTS_PER_CELL = _F["slots_per_cell"]  # 13
ANCHORED = _F["anchored"]  # 2925
NUM_CLASSES = _F["num_classes"]  # 2927
PASS_CLASS = _F["pass_class"]  # 2925
EXTRA_CLASS = _F["extra_class"]  # 2926
CATCH_ALL = NUM_CLASSES - ANCHORED  # 2

assert ANCHORED == SIDE * SIDE * SLOTS_PER_CELL, "footprint constants inconsistent with FFI"


def to_spatial(dense):
    """Split a ``(..., NUM_CLASSES)`` array into a ``(..., SIDE, SIDE, SLOTS)``
    anchored block and a ``(..., CATCH_ALL)`` catch-all block. Inverse of
    ``from_spatial``. The anchored reshape is cell-major, slot-minor -- matching
    ``footprint_class = (r*side + c)*slots + slot`` in the unflipped frame."""
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
    """A ``(..., NUM_CLASSES)`` array as ``(..., SLOTS_PER_CELL, SIDE, SIDE)``
    per-slot board maps, the catch-all block dropped. This is the evidence
    fusion's channel layout: anchored class ``(cell, slot)`` lands on channel
    ``slot`` at ``cell``."""
    anchored, _ = to_spatial(dense)  # (..., SIDE, SIDE, SLOTS)
    return np.moveaxis(anchored, -1, -3)


def top_k_sparse(dense, k):
    """The ``k`` largest entries of each row of a ``(..., n)`` array, as fixed
    padded ``(indices (..., k) int32, values (..., k) float32)`` sorted
    descending. The k indices are always distinct board positions (so
    ``scatter_sparse`` reconstructs exactly when ``k`` covers every nonzero
    entry); pad slots past the nonzero support carry a real index with value 0."""
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
    """Dense ``(..., n)`` reconstructed from sparse ``(indices, values)``. The k
    indices per row are distinct, so this is a plain scatter (not scatter-add);
    every unlisted class is 0."""
    indices = np.asarray(indices)
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros((*indices.shape[:-1], n), dtype=np.float32)
    np.put_along_axis(out, indices.astype(np.intp), values, axis=-1)
    return out


def top_k_mass(dense, k):
    """Fraction of each row's total that its top-k entries carry -- the fidelity
    a sparse top-k target preserves. Rows summing to 0 report 1.0."""
    dense = np.asarray(dense, dtype=np.float64)
    total = dense.sum(axis=-1)
    _, values = top_k_sparse(dense, k)
    kept = values.sum(axis=-1)
    return np.where(total > 0, kept / np.where(total > 0, total, 1.0), 1.0)

"""Move-feature inputs for the move set evaluation model.

The engine owns the encoding (engine/include/training/move_set_encoder.h), so
the training dataset and the in-engine agents share one implementation. This
module re-exports its FFI bindings: `encode_moves` turns packed Move records
into per-candidate letter/blank/square/mask/scalar arrays, `move_encoding_dims`
reports the layout constants the model sizes its embeddings from, and
`score_diff_input_layout` locates the score-diff scalar in the board input so
the dataset can read a position's pre-move differential from its encoded row.
"""

from __future__ import annotations

from scribblez.ffi import (
    encode_moves,
    move_encoding_dims,
    move_encoding_version,
    score_diff_input_layout,
)

BOARD = 15

__all__ = [
    "BOARD",
    "encode_moves",
    "move_encoding_dims",
    "move_encoding_version",
    "score_diff_input_layout",
]

"""Move-feature inputs for the move set evaluation model.

The encoding is owned by the engine
(engine/include/scribblez/move_set_encoder.h) so the training dataset
and the eventual move set evaluation agent share one implementation and never
drift. This module is a thin Python view over it: `encode_moves` turns a (M,) array of
packed Move records (plus their pre-move score differentials) into the
per-candidate letter/blank/square/mask/scalar arrays, `move_encoding_dims`
reports the layout constants (max placed tiles, scalar count, letter
vocabulary, board cells) the model builds its embeddings from, and
`score_diff_input_layout` locates the board input's score-diff scalar so the
dataset can read each position's pre-move differential straight out of the
encoded row.
"""

from __future__ import annotations

from scribblez.ffi import encode_moves, move_encoding_dims, score_diff_input_layout

BOARD = 15

__all__ = ["BOARD", "encode_moves", "move_encoding_dims", "score_diff_input_layout"]

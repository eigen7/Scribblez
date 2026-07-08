#pragma once

// Move-feature encoding for the move set evaluation model. The board is
// encoded once by the shared trunk; each candidate move is additionally
// described by a small feature set the model's move encoder embeds:
//   * per placed tile: its letter, a blank flag, and its board square;
//   * a scalar block: the resultant post-move score differential, the tile
//     count, and whether the move is a board play.
//
// This is the single source of truth for that layout: the training dataset
// reaches it through the FFI (scribblez_move_set_encode_moves) and the move
// set evaluation agent encodes candidates the same way at inference, so the
// two never drift. The placement is a pure function of the Move -- a PLAY's
// placed squares come from visit_placed_squares and its tiles from
// Move::glyph in the same lane order; EXCHANGE/PASS place nothing and differ
// only in the scalars.
//
// A tile's letter is stored as its A..Z identity with a separate blank flag,
// rather than one glyph code spanning natural and blank letters, so the model
// shares one letter representation across a natural tile and its blank twin.
//
// The score differential feeds in as the resultant post-move value
// (pre_move_score_diff + move score, mover POV) rather than the raw move score:
// that is the quantity the position evaluation model evaluates, encoded on
// the same scale as the board trunk's score-diff input (input_encoder.h), so
// the two are directly comparable. The leave is intentionally not encoded --
// it is recoverable from the mover's rack, which the board trunk sees, minus
// the placed tiles.

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/tile.h"

#include <cstdint>

namespace scribblez {
namespace move_set {

// Tiles-per-move slot: the letter/blank/square arrays are this wide (a
// full-rack bingo).
inline constexpr int kMoveMaxPlaced = RACK_SIZE;
// Per-move scalar features: [resultant_score_diff, tiles/7, is_play].
inline constexpr int kMoveScalars = 3;
// Letter-embedding vocabulary: 0 the empty/pad slot, 1..26 the letters A..Z.
inline constexpr int kMoveLetterVocab = 27;
// Board-square embedding size: one index per board cell.
inline constexpr int kMoveCells = BOARD_SIZE * BOARD_SIZE;

// Encode one move into caller-owned slices. `pre_move_score_diff` is the
// mover's score advantage before the move (points), used to form the resultant
// post-move differential.
//   letters   int32[kMoveMaxPlaced]  placed-tile letters 1..26 (0 in empty slots)
//   blanks    uint8[kMoveMaxPlaced]  1 iff that placed tile is a blank
//   squares   int32[kMoveMaxPlaced]  placed-tile board indices r*BOARD_SIZE+c
//                                     (0 in empty slots, masked out downstream)
//   tile_mask uint8[kMoveMaxPlaced]  1 for real placed tiles, else 0
//   scalars   float[kMoveScalars]
void encode_move(const Move& m, int pre_move_score_diff, int32_t* letters, uint8_t* blanks,
                 int32_t* squares, uint8_t* tile_mask, float* scalars);

// Encode `n` contiguous moves into the batch arrays; `pre_move_score_diffs` is
// the per-move pre-move differential (one per move, since a flattened batch
// mixes positions). letters/blanks/squares/tile_mask are n*kMoveMaxPlaced,
// scalars is n*kMoveScalars (candidate-major).
void encode_moves(const Move* moves, int64_t n, const int32_t* pre_move_score_diffs,
                  int32_t* letters, uint8_t* blanks, int32_t* squares, uint8_t* tile_mask,
                  float* scalars);

}  // namespace move_set
}  // namespace scribblez

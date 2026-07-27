#pragma once

// Move-feature encoding for the move set evaluation model: on top of the board
// the shared trunk encodes, each candidate move is described by its placed
// tiles (letter, blank flag, board square) and a small scalar block. The single
// source of truth for that layout, reached both by the training dataset through
// the FFI and by the agent at inference, so the two never drift.
//
// A tile's letter is stored as its A..Z identity with a separate blank flag,
// rather than one glyph code spanning natural and blank letters, so the model
// shares one letter representation between a natural tile and its blank twin.
//
// The score differential feeds in as the resultant post-move value rather than
// the raw move score: that is the quantity the position evaluation model
// evaluates, on the same scale as the board trunk's score-diff input, so the
// two compare directly. The leave is deliberately absent, being recoverable
// from the mover's rack (which the trunk sees) minus the placed tiles.

#include "game/board.h"
#include "game/move.h"
#include "game/tile.h"

#include <cstdint>

namespace scribblez {
namespace move_set {

// Tiles per move slot: a full-rack bingo.
inline constexpr int kMoveMaxPlaced = RACK_SIZE;
// [resultant_score_diff, tiles/7, is_play].
inline constexpr int kMoveScalars = 3;
// 0 is the empty/pad slot, 1..26 the letters A..Z.
inline constexpr int kMoveLetterVocab = 27;
// One embedding index per board cell.
inline constexpr int kMoveCells = BOARD_SIZE * BOARD_SIZE;

// `pre_move_score_diff` is the mover's score advantage in points before the
// move, from which the resultant post-move differential is formed.
//   letters   int32[kMoveMaxPlaced]  placed-tile letters 1..26 (0 in empty slots)
//   blanks    uint8[kMoveMaxPlaced]  1 iff that placed tile is a blank
//   squares   int32[kMoveMaxPlaced]  board indices r*BOARD_SIZE+c (0 in empty
//                                     slots, masked out downstream)
//   tile_mask uint8[kMoveMaxPlaced]  1 for real placed tiles, else 0
//   scalars   float[kMoveScalars]
void encode_move(const Move& m, int pre_move_score_diff, int32_t* letters, uint8_t* blanks,
                 int32_t* squares, uint8_t* tile_mask, float* scalars);

// The batch form, candidate-major. `pre_move_score_diffs` carries one entry per
// move, a flattened batch mixing positions.
void encode_moves(const Move* moves, int64_t n, const int32_t* pre_move_score_diffs,
                  int32_t* letters, uint8_t* blanks, int32_t* squares, uint8_t* tile_mask,
                  float* scalars);

}  // namespace move_set
}  // namespace scribblez

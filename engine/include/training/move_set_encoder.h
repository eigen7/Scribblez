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
//
// EXCHANGE moves have no placed squares, but their surrendered tiles fill the
// same letter/blank/tile_mask slots (squares stay 0, masked spatially
// downstream by is_play): without them every same-size exchange from one rack
// encodes identically while the teacher's value depends on exactly which
// tiles are kept, an irreducible target error. An unassigned blank -- an
// exchange surrenders undesignated blanks -- encodes as letters=0/blanks=1,
// represented by the blank flag alone. PASS stays all-zero.

#include "game/board.h"
#include "game/move.h"
#include "game/tile.h"

#include <cstdint>
#include <vector>

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

// The move-feature SEMANTICS version: bumped whenever encode_move changes what
// the same Move encodes to (v1: exchanges carry their surrendered tiles). A
// checkpoint is only valid with the encoder version its training rows used, and
// nothing structural detects a mismatch -- the tensor shapes are unchanged --
// so the version rides the checkpoint config and the exported ONNX metadata,
// where the engine-side loader rejects a stale model instead of silently
// feeding it off-distribution rows.
inline constexpr int kMoveEncodingVersion = 1;

// `pre_move_score_diff` is the mover's score advantage in points before the
// move, from which the resultant post-move differential is formed.
//   letters   int32[kMoveMaxPlaced]  tile letters 1..26 (0 in empty slots, and
//                                     for an exchange's unassigned blanks)
//   blanks    uint8[kMoveMaxPlaced]  1 iff that tile is a blank
//   squares   int32[kMoveMaxPlaced]  board indices r*BOARD_SIZE+c for placed
//                                     tiles (0 in empty slots and for
//                                     exchanges, masked out downstream)
//   tile_mask uint8[kMoveMaxPlaced]  1 for real tiles (placed or surrendered),
//                                     else 0
//   scalars   float[kMoveScalars]
void encode_move(const Move& m, int pre_move_score_diff, int32_t* letters, uint8_t* blanks,
                 int32_t* squares, uint8_t* tile_mask, float* scalars);

// The batch form, candidate-major. `pre_move_score_diffs` carries one entry per
// move, a flattened batch mixing positions.
void encode_moves(const Move* moves, int64_t n, const int32_t* pre_move_score_diffs,
                  int32_t* letters, uint8_t* blanks, int32_t* squares, uint8_t* tile_mask,
                  float* scalars);

// One encoded candidate set, owning the five buffers encode_move fills so it
// crosses an API boundary as a single object (nn::MoveSetEvalService takes
// one). Row-major, `count` rows of kMoveMaxPlaced -- kMoveScalars for scalars.
// The training path keeps writing into its own tensors through the batch
// pointer form above; this is the inference side's convenience.
struct MoveFeatureArrays {
  std::vector<int32_t> letters;
  std::vector<uint8_t> blanks;
  std::vector<int32_t> squares;
  std::vector<uint8_t> tile_mask;
  std::vector<float> scalars;
  int count = 0;

  // Size the buffers for `n` moves and encode them. One differential covers the
  // set: a candidate set is one position's alternatives, so every move in it
  // resolves the same pre-move score advantage.
  void encode(const Move* moves, int n, int pre_move_score_diff);
};

}  // namespace move_set
}  // namespace scribblez

#pragma once

// Per-move features for the move set evaluation model. The shared trunk encodes
// the board once; each candidate move is then described by its tiles (letter,
// blank flag, board square) and a small scalar block. Both the training dataset
// (through the FFI) and the agent at inference encode moves here, so the two
// cannot drift.
//
// A tile's letter is its A..Z identity plus a separate blank flag, so a natural
// tile and a blank playing the same letter share one letter representation.
//
// The score scalar is the resultant post-move differential, not the move's raw
// score: that is what the position evaluation model evaluates, on the same
// scale as the trunk's score-diff input, so the two compare directly. The leave
// is absent because it is recoverable from the mover's rack, which the trunk
// sees, minus the move's tiles.
//
// An EXCHANGE has no placed squares, but its surrendered tiles fill the letter,
// blank and tile_mask slots (squares stay 0; downstream, is_play masks them out
// spatially). Otherwise every same-size exchange from one rack would encode
// identically, though the teacher's value depends on exactly which tiles are
// kept. An exchanged blank has no designated letter, so it encodes as
// letter 0 with the blank flag set. A PASS encodes as all zeros.

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

// The move-feature semantics version, bumped whenever encode_move changes what
// a given Move encodes to (v1: exchanges carry their surrendered tiles). A
// checkpoint is valid only with the version its training rows used, and the
// tensor shapes do not change between versions, so nothing structural would
// catch a mismatch. The version therefore travels in the checkpoint config and
// the exported ONNX metadata, and the engine-side loader rejects a stale model
// rather than feed it off-distribution rows.
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

// One encoded candidate set, owning the five buffers encode_move fills, so it
// crosses an API boundary as one object (inference services take it). `count`
// rows of kMoveMaxPlaced each (kMoveScalars for scalars). The training path
// writes its own tensors through the pointer forms above.
struct MoveFeatureArrays {
  std::vector<int32_t> letters;
  std::vector<uint8_t> blanks;
  std::vector<int32_t> squares;
  std::vector<uint8_t> tile_mask;
  std::vector<float> scalars;
  int count = 0;

  // Size the buffers for `n` moves and encode them. One differential serves the
  // whole set, since all candidates start from the same position.
  void encode(const Move* moves, int n, int pre_move_score_diff);
};

}  // namespace move_set
}  // namespace scribblez

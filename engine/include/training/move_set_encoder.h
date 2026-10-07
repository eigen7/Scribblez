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
//
// Each move also carries the cross-checks it changes. The trunk encodes the
// pre-move board once per position, but the position evaluation teacher scores
// each candidate on its post-move board, cross-check planes included -- and
// part of that change is not inferable from the placed tiles: the cross-checks
// at the two ends of the word a move forms are that word's hooks. So the
// changed entries (training/cross_check_delta.h, at most kMoveCrossSlots per
// move) ride along per move as `cross_cells` and `cross_letters`. Squares the
// move fills are not entries: the tile features already say they are taken.

#include "game/board.h"
#include "game/move.h"
#include "game/tile.h"
#include "training/cross_check_delta.h"

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

// Cross-check entries per move: the most a move can change.
inline constexpr int kMoveCrossSlots = kMoveMaxCrossDeltas;
// cross_letters elements per move: one legal-letter flag per slot and letter.
inline constexpr int kMoveCrossLetters = kMoveCrossSlots * 26;

// The move-feature semantics version, bumped whenever encode_move changes what
// a given Move encodes to (v1: exchanges carry their surrendered tiles; v2:
// the post-move cross-check entries). A
// checkpoint is valid only with the version its training rows used, and the
// tensor shapes do not change between versions, so nothing structural would
// catch a mismatch. The version therefore travels in the checkpoint config and
// the exported ONNX metadata, and the engine-side loader rejects a stale model
// rather than feed it off-distribution rows.
inline constexpr int kMoveEncodingVersion = 2;

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

// The cross-check entries `m` changes on `board` (whose move-generation caches
// must be valid; it is played on and restored, so unchanged on return):
//   cells    int32[kMoveCrossSlots]    1 + axis*kMoveCells + square per entry,
//                                       axis 0 for the horizontal-play
//                                       cross-check block and 1 for the
//                                       vertical-play one; 0 in empty slots
//   letters  uint8[kMoveCrossLetters]  slot-major, 26 per slot: 1 iff that
//                                       letter is legal at the entry's square
//                                       after the move (0 in empty slots)
// Empty and "no letter legal" differ -- a move can kill a square outright --
// which is why emptiness lives in `cells`. `undo` is scratch, passed in so a
// batch reuses its allocations. EXCHANGE and PASS have no entries.
void encode_move_cross_checks(Board& board, const Move& m, BoardUndo& undo, int32_t* cells,
                              uint8_t* letters);

// The batch form over one position's candidate set, candidate-major. Works on a
// copy of `board`, building its move-generation caches from `dict`.
void encode_moves_cross_checks(const Board& board, const Dictionary& dict, const Move* moves,
                               int64_t n, int32_t* cells, uint8_t* letters);

// One encoded candidate set, owning the buffers encode_move and
// encode_move_cross_checks fill, so it crosses an API boundary as one object
// (inference services take it). `count` rows of each field's per-move width.
// The training path writes its own tensors through the pointer forms above.
struct MoveFeatureArrays {
  std::vector<int32_t> letters;
  std::vector<uint8_t> blanks;
  std::vector<int32_t> squares;
  std::vector<uint8_t> tile_mask;
  std::vector<float> scalars;
  std::vector<int32_t> cross_cells;
  std::vector<uint8_t> cross_letters;
  int count = 0;

  // Size the buffers for `n` moves and encode them. All candidates start from
  // the same position: one board, one differential.
  void encode(const Board& board, const Dictionary& dict, const Move* moves, int n,
              int pre_move_score_diff);

  // Size every buffer for `n` moves, zero-filled.
  void resize(int n);
};

}  // namespace move_set
}  // namespace scribblez

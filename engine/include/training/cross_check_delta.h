#pragma once

// The cross-check changes a candidate move causes, as a short per-move list.
//
// The position evaluation teacher scores a candidate on its post-move board,
// cross-check planes included (kCrossChecks in encoding/input_encoder.h). The
// move set evaluation student encodes the pre-move board once per position and
// describes each candidate with per-move features, so the post-move cross-checks
// can reach it only as a per-move feature. They matter because they are lexical
// facts (the hooks of the word just formed) that the pre-move planes do not
// determine. Dense post-move planes per candidate would undo the shared board
// encode the student exists for, but a delta stays small: a move changes the
// cross-check only of the empty squares at the two ends of each perpendicular
// run it extends and at the two ends of its own word.
//
// An entry is one (axis, square) whose letter mask differs between the pre- and
// post-move boards. Squares the move fills are never entries: the placed-tile
// features already mark them occupied, and the planes are zero there. Patching
// the pre-move planes with a move's entries, and zeroing its placed squares,
// reproduces the teacher's post-move planes exactly.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

namespace move_set {

// n placed tiles extend n perpendicular runs (two ends each), plus the two ends
// of the word along the play axis.
inline constexpr int kMoveMaxCrossDeltas = 2 * RACK_SIZE + 2;

// The cross-check entries `m` changes on `board`. The move must be legal on
// `board` and share its frame, and the board's move-generation caches must be
// valid. The move is applied and then undone, so `board` is unchanged on
// return. Entries are sorted by (axis, square); EXCHANGE and PASS have none.
//   axes       uint8[kMoveMaxCrossDeltas]   0: the horizontal-play cross-check
//                                           block, 1: the vertical-play one
//                                           (the kCrossChecks plane halves)
//   squares    int32[kMoveMaxCrossDeltas]   r*BOARD_SIZE+c, board frame
//   old_masks  uint32[kMoveMaxCrossDeltas]  the pre-move CrossCheck::mask
//   new_masks  uint32[kMoveMaxCrossDeltas]  the post-move CrossCheck::mask
//   delta_mask uint8[kMoveMaxCrossDeltas]   1 for real entries, else 0 (and the
//                                           other four arrays are 0 there)
// `undo` is scratch, passed in so a batch reuses its allocations.
void encode_cross_check_deltas(Board& board, const Move& m, BoardUndo& undo, uint8_t* axes,
                               int32_t* squares, uint32_t* old_masks, uint32_t* new_masks,
                               uint8_t* delta_mask);

// The batch form over one position's candidate set, candidate-major. Works on a
// copy of `board`, building its move-generation caches from `dict`.
void encode_cross_check_deltas(const Board& board, const Dictionary& dict, const Move* moves,
                               int64_t n, uint8_t* axes, int32_t* squares, uint32_t* old_masks,
                               uint32_t* new_masks, uint8_t* delta_mask);

}  // namespace move_set
}  // namespace scribblez

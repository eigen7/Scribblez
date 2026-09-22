#pragma once

// The cross-check change a candidate move causes, as a sparse per-move list.
//
// The position evaluation teacher scores a candidate on its post-move board,
// cross-check planes included (input_encoder.h kCrossChecks). The move set
// evaluation student encodes the pre-move board once per position, so a
// candidate's post-move cross-checks can only reach it as a per-move feature --
// and they are lexical facts (the hooks of the word just formed) that the
// pre-move planes do not determine. Dense post-move planes per candidate would
// undo the one-board-encode amortization the student exists for; but a move
// changes the cross-check of at most the two empty squares ending each
// perpendicular run it extends and the two ending its own word, so the change
// is a list of at most kMoveMaxCrossDeltas squares.
//
// An entry is one (axis, square) whose letter mask differs between the pre- and
// post-move board. Squares the move fills are not entries: the placed-tile
// features already say they are occupied, and the planes are zero there. The
// pre-move planes patched with a move's entries (and zeroed on its placed
// squares) are exactly the teacher's post-move planes.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

namespace move_set {

// n placed tiles extend n perpendicular runs (two ends each) and one word along
// the play axis (two ends).
inline constexpr int kMoveMaxCrossDeltas = 2 * RACK_SIZE + 2;

// The cross-check entries `m` changes on `board`, which the move must be legal
// on and share a frame with. `board` is played on and restored, so it is
// unchanged on return; its move-generation caches must be valid. Entries are
// ordered by (axis, square); EXCHANGE and PASS have none.
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

// The batch form, candidate-major, over one position's candidate set.
void encode_cross_check_deltas(const Board& board, const Dictionary& dict, const Move* moves,
                               int64_t n, uint8_t* axes, int32_t* squares, uint32_t* old_masks,
                               uint32_t* new_masks, uint8_t* delta_mask);

}  // namespace move_set
}  // namespace scribblez

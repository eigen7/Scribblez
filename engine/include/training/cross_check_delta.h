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
#include "gpu/cross_check_abi.h"

namespace scribblez {

namespace move_set {

inline constexpr int kMoveMaxCrossDeltas = cross_check_abi::kMaxDeltas;

// The cross-check entries `m` changes on `board`. The move must be legal on
// `board` and share its frame, and the board's move-generation caches must be
// valid. The move is applied and then undone, so `board` is unchanged on
// return. EXCHANGE and PASS have none. `undo` is scratch, passed in so a batch
// reuses its allocations.
cross_check_abi::MoveDeltas cross_check_deltas(Board& board, const Move& m, BoardUndo& undo);

}  // namespace move_set
}  // namespace scribblez

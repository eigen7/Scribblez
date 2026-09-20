#pragma once

// "High-value setup" plays: a play that keeps a J, Q, X or Z in the leave and
// lays a tile beside an empty premium square where that kept tile then hooks --
// the K6 AC.TA pattern of docs/plans/sim_labeled_candidates.md (the A's beside
// two triple-letter squares, the Z kept for either). Static equity prices the
// leave and the points and nothing of the square it opens for the mover, so
// these are the plays a HastyBot candidate cut is most likely to bury.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

namespace scribblez {

class Dictionary;

// True iff the play `m` places a blank.
bool places_blank(const Move& m);

// True iff `m`, a play from `rack` on `before`, is a high-value setup:
//   * it places no blank, and the leave keeps a J, Q, X or Z -- call it H;
//   * some empty square S beside one of its newly placed tiles T admits H: every
//     word H would form at S is valid, the word along the S-T line among them;
//   * H did not already hook at S before the play (the play made the spot);
//   * S is critical: a triple-letter or triple-word square, or a double-letter
//     square with an empty double- or triple-word square at most 4 squares
//     from it along the line a play through S would take (perpendicular to
//     S-T), so the doubled H lands in a multiplied word.
bool is_high_value_setup(const Board& before, const Dictionary& dict, const Rack& rack,
                         const Move& m);

}  // namespace scribblez

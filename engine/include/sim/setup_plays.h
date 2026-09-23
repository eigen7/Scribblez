#pragma once

// "High-value setup" plays: a play that keeps a J, Q, X or Z and lays a tile
// beside an empty premium square where the kept tile then hooks. Example: the
// K6 AC.TA pattern of docs/plans/sim_labeled_candidates.md, whose A's sit
// beside two triple-letter squares with the Z kept for either. Static equity
// prices the leave and the points but not the square the play opens for the
// mover, so a HastyBot candidate cut is likely to bury these plays.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

namespace scribblez {

class Dictionary;

// True iff the play `m` places a blank.
bool places_blank(const Move& m);

// True iff `m`, a play from `rack` on `before`, is a high-value setup:
//   * it places no blank, and the leave keeps a J, Q, X or Z -- call it H;
//   * H hooks at some empty square S beside one of m's placed tiles T: every
//     word H would form at S is valid, including the one along the S-T line;
//   * H did not already hook at S before the play;
//   * S is a triple-letter or triple-word square, or a double-letter square
//     with an empty word-premium square within 4 squares along the line a play
//     through S would take (perpendicular to S-T).
bool is_high_value_setup(const Board& before, const Dictionary& dict, const Rack& rack,
                         const Move& m);

}  // namespace scribblez

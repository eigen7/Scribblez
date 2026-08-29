#pragma once

#include "game/board.h"
#include "training/footprint.h"

#include <array>

namespace scribblez {

// Per-class legality over the footprint classes, in the `flip` frame:
// mask[cls] == true iff the masked softmax should keep that class. Illegal
// classes are driven to -inf before the softmax, so their probability (and
// gradient) is zero. Computed from the current board -- no move-gen, no
// dictionary lookup on the main word, no per-letter rack matching.

using FootprintMask = std::array<bool, kFootprintClasses>;

// Legality for an OPPONENT placement head (opp_next / opp_win): the opponent
// moves next on `board`, so the current-board test is exact for them. A class
// (anchor, orientation, k) is legal iff its covered cells (the first k empty
// cells from the anchor) fit the board AND every covered cell admits some letter
// in the play orientation -- its perpendicular cross-check is non-empty, or the
// square is unconstrained (no perpendicular neighbor) -- AND k <= tile_budget.
// A lone tile (k==1) is orientation-free and legal if it admits a letter in
// either orientation.
//
// This is a SOUND OVER-APPROXIMATION, never masking a truly legal move: it omits
// the main-word dictionary check and joint-rack satisfiability, so it can admit
// footprints no real move realizes (the model learns those toward zero). Tile
// AVAILABILITY is deliberately not modelled here -- that a valid word exists at a
// square (a non-empty cross-check) does not mean the opponent holds a letter for
// it; learning that belief is the model's job, not the mask's.
//
// `board` must have movegen caches built (cross_checks bound to a dictionary);
// on a bare board every square reads as unconstrained. kPassClass is always
// legal; kExtraClass is legal iff `win_head` (the win heads' "not-win" outcome),
// an unused masked-off dummy for a plays head.
void opp_footprint_mask(const Board& board, int tile_budget, bool flip, bool win_head,
                        FootprintMask& mask);

}  // namespace scribblez

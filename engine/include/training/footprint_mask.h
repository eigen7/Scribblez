#pragma once

#include "game/board.h"
#include "training/footprint.h"

#include <array>
#include <cstdint>

namespace scribblez {

// Per-class legality over the footprint classes, in the `flip` frame:
// mask[cls] == true iff the masked softmax should keep that class. Illegal
// classes are driven to -inf before the softmax, so their probability (and
// gradient) is zero. Computed from the current board (and, for the opp mask, the
// opponent's available tiles) -- no move-gen, no dictionary lookup on the main
// word, no joint multi-cell tile contention.

using FootprintMask = std::array<bool, kFootprintClasses>;

// The sound tile budget every caller passes these masks: a mover holds at most
// RACK_SIZE tiles, so a full rack is the loosest cap and never masks a real move
// (a smaller, endgame-aware budget would only tighten -- see the TODO in
// training_targets.cpp). Shared so the per-row training masks and the .mset /
// dashboard collapse cannot drift.
inline constexpr int kMaskTileBudget = RACK_SIZE;

// Legality for an OPPONENT placement head (opp_next / opp_win): the opponent
// moves next on `board`, so the current-board test is exact for them. A class
// (anchor, orientation, k) is legal iff its covered cells (the first k empty
// cells from the anchor) fit the board AND every covered cell admits some
// AVAILABLE letter in the play orientation -- a letter both legal at the square
// (its perpendicular cross-check permits it, or the square is unconstrained) and
// in stock -- AND k <= tile_budget. A lone tile (k==1) is
// orientation-free and legal iff at least one available letter is admissible in
// both orientations at once.
//
// `available_counts` is the opponent's availability as a 27-count array (A..Z
// then blank), the unseen pool the opponent draws from or holds -- 100 tiles
// minus the board minus the mover's rack. A cell's cross-check is intersected
// with the tiles in stock; a blank in stock is a wildcard that satisfies any
// board-legal square. Passing `available_counts == nullptr` disables
// availability (every tile treated as in stock), recovering the pure
// board-legality mask.
//
// Availability masking is SOUND -- it never masks a move that is actually legal
// for the opponent: the opponent plays from a rack drawn out of exactly this
// unseen pool, so any tile it truly plays is in stock (with sufficient count, as
// its tiles are unseen to us). The mask still omits the main-word dictionary
// check and joint multi-cell tile contention, so it can admit footprints no real
// move realizes (the model learns those toward zero); see the contention
// follow-up for tightening the per-cell test to a joint one.
//
// `board` must have movegen caches built (cross_checks bound to a dictionary);
// on a bare board every square reads as unconstrained. kPassClass is always
// legal; kExtraClass is legal iff `win_head` (the win heads' "not-win" outcome),
// an unused masked-off dummy for a plays head.
void opp_footprint_mask(const Board& board, const uint8_t* available_counts, int tile_budget,
                        bool flip, bool win_head, FootprintMask& mask);

// Legality for a SELF placement head (self_next / self_win): the mover plays two
// plies out, after the opponent moves, so the board is unknown. This is an
// opp-move-INVARIANT over-approximation from the current board -- it must never
// mask a footprint that could become legal under some opponent move, so it is
// cross-check-oblivious (the opponent's move can change every cross-check).
//
// A class is legal iff its covered cells fit the board, each is self-reachable,
// and k <= self_budget. Self-reachability uses a tiles-to-reach distance
// transform d(Z) = fewest tiles to bridge to empty Z from the current structure
// (multi-source 4-neighbour BFS, occupied = 0): Z is reachable iff
// d(Z) <= opp_budget + self_budget -- the opponent extends toward Z within its
// budget and the mover finishes within its own. Sound because the BFS is a lower
// bound on the true tile cost. A tile-less board treats every cell as reachable
// (game-start guard). kPassClass is always legal; kExtraClass iff `win_head`.
void self_footprint_mask(const Board& board, int self_budget, int opp_budget, bool flip,
                         bool win_head, FootprintMask& mask);

}  // namespace scribblez

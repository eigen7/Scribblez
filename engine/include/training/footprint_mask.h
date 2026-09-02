#pragma once

#include "game/board.h"
#include "training/footprint.h"

#include <array>
#include <cstdint>

namespace scribblez {

// Per-class legality over the footprint classes, in the board's frame:
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

// Per-class legality for an OPPONENT placement head (opp_next / opp_win): the
// opponent moves next, so the current board is exact. A footprint is legal iff it
// fits, connects to the board (a covered cell abuts an occupied tile), every
// covered cell admits an available letter (its cross-check permits one in stock),
// and k <= tile_budget; the test still omits the main-word dictionary check and
// multi-cell tile contention, so it can admit footprints no real move realizes
// (the model learns those toward zero).
//   - available_counts: the opponent's unseen pool as a 27-count array (A..Z then
//     blank), intersected with each cell's cross-check (a blank is a wildcard);
//     nullptr treats every tile as in stock (pure board legality). SOUND -- never
//     masks a tile the opponent could draw and play.
//   - tile_budget: cap on tiles placed in one move (k).
//   - win_head: keep kExtraClass (the win heads' not-win slot); false for a plays
//     head. kPassClass is always legal.
// `board` must have movegen caches built.
void opp_footprint_mask(const Board& board, const uint8_t* available_counts, int tile_budget,
                        bool win_head, FootprintMask& mask);

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
void self_footprint_mask(const Board& board, int self_budget, int opp_budget, bool win_head,
                         FootprintMask& mask);

// Per-cell reachability plane: out[r*kFootprintSide + c] == 1
// iff some legal "who moves next" footprint covers that cell, else 0. It is the
// opp_footprint_mask (with `available_counts` as the stock and `tile_budget` as
// the cap -- nullptr counts being board legality only) reduced to the board
// squares its legal footprints touch, i.e. "which squares this pool can reach on
// the current board." The input-feature view of the mask; `board` must have
// movegen caches built. Writes exactly kFootprintCells floats.
void footprint_reachable_cells(const Board& board, const uint8_t* available_counts, int tile_budget,
                               float* out);

}  // namespace scribblez

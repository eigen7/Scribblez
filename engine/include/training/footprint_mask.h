#pragma once

#include "game/board.h"
#include "training/footprint.h"

#include <array>
#include <bitset>
#include <cstdint>

namespace scribblez {

// Legality masks over the footprint classes (training/footprint.h), in the
// board's frame: mask[cls] is true iff the masked softmax keeps that class.
// Illegal classes are driven to -inf before the softmax, so they get zero
// probability and zero gradient.
//
// Every mask is built from one primitive, footprint_ply(): one ply of play from
// a seed set of squares, keeping the footprints that abut the seed. For a move
// on the current board the seed is its occupied squares S; for a move one ply
// later it is the set of squares the earlier ply could reach:
//
//   opp_this_turn  = footprint_ply(S, cross-checks on, opponent's pool)    opp mask, kOppReach
//   self_this_turn = footprint_ply(S, cross-checks on, mover's pool)       kSelfReach
//   self_next_turn = footprint_ply(opp_this_turn.reach, cross-checks off)  self mask
//
// kOppReach and kSelfReach are the input planes footprint_reachable_cells builds.
//
// The first ply happens on the known board, so cross-checks and tile
// availability apply. The second follows an unknown opponent move, which can
// rewrite any cross-check and draws from a different rack, so it ignores
// cross-checks: it must never mask a footprint that some opponent move makes
// legal. Each ply is a sound over-approximation of the real move set: no move
// generation, no main-word lookup, no joint tile contention across cells.

using FootprintMask = std::array<bool, kFootprintClasses>;

// The tile budget (max k) every caller passes these masks. A player holds at
// most RACK_SIZE tiles, so a full rack never masks a real move. Shared so the
// training-row masks and footprint_collapse.h cannot drift apart.
//
// TODO(sharpen masks): near the endgame fewer tiles remain, and a tighter cap
// would make the masks more precise. Care is needed for soundness. Two plies
// out the mover has redrawn, so today's rack size is not a sound bound for the
// self mask. The bag count is also not in the observer's information set: only
// the unseen total (bag plus opponent rack, i.e. 100 - board - own rack) is
// knowable. An unsound cap that masks a real target falls back on the loss's
// keep-target guard.
inline constexpr int kMaskTileBudget = RACK_SIZE;

// A set of board squares. A ply's seed and its reach are both SquareSets, so
// plies chain.
struct SquareSet {
  std::bitset<kFootprintCells> bits;

  bool contains(int idx) const { return bits.test(idx); }
  bool contains(int r, int c) const { return contains(r * kFootprintSide + c); }
  bool empty() const { return bits.none(); }
};

// S, the seed for a ply on this board.
SquareSet occupied_squares(const Board& board);

struct FootprintPly {
  FootprintMask mask;  // the ply's legal footprint classes
  SquareSet reach;     // the seed plus every square those footprints cover
};

// One ply from `seed`: the footprints of k <= budget tiles on empty squares
// that abut a seed square. An empty seed (the opening move) has nothing to
// abut, so every footprint that fits on the board is kept.
//   - use_cross_checks: each covered square must admit some letter that is
//     legal there and, given available_counts, in stock. Off for a ply on a
//     board an unknown move will rewrite first.
//   - available_counts: the player's pool as 27 counts (A..Z, then blank; a
//     blank is a wildcard). nullptr treats every tile as in stock. Ignored when
//     cross-checks are off.
//   - win_head: keep kExtraClass (the win heads' not-win outcome); false for a
//     plays head. kPassClass is always kept.
// `board` needs move-generation caches only when cross-checks are on.
FootprintPly footprint_ply(const Board& board, const SquareSet& seed, int budget,
                           bool use_cross_checks, const uint8_t* available_counts, bool win_head);

// The mask for an opponent placement head (opp_next / opp_win), whose player
// moves next on `board`: opp_this_turn above.
void opp_footprint_mask(const Board& board, const uint8_t* available_counts, int tile_budget,
                        bool win_head, FootprintMask& mask);

// The mask for a self placement head (self_next / self_win), whose player moves
// after the opponent: self_next_turn above. `opp_available_counts` is the pool
// the opponent's ply draws from.
void self_footprint_mask(const Board& board, int self_budget, int opp_budget,
                         const uint8_t* opp_available_counts, bool win_head, FootprintMask& mask);

// An input plane: out[r*kFootprintSide + c] is 1 iff some footprint of the
// this-turn ply under `available_counts` covers that cell, else 0 (occupied
// cells are never covered). `board` must have move-generation caches built.
// Writes kFootprintCells floats.
void footprint_reachable_cells(const Board& board, const uint8_t* available_counts, int tile_budget,
                               float* out);

}  // namespace scribblez

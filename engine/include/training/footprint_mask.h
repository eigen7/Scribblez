#pragma once

#include "game/board.h"
#include "training/footprint.h"

#include <array>
#include <bitset>
#include <cstdint>

namespace scribblez {

// Per-class legality over the footprint classes, in the board's frame:
// mask[cls] == true iff the masked softmax should keep that class. Illegal
// classes are driven to -inf before the softmax, so their probability (and
// gradient) is zero.
//
// Every mask is built from one primitive, footprint_ply(): one ply of play from
// a seed set of squares, kept as the footprints that abut the seed. The seed is
// what a ply starts from -- the occupied squares S for a move on this board, or
// the squares a prior ply reached for a move on the board that ply will have
// extended. So:
//
//   opp_this_turn  = footprint_ply(S, cross-checks on, the opponent's pool)
//   self_this_turn = footprint_ply(S, cross-checks on, the mover's pool)   [input plane]
//   self_next_turn = footprint_ply(opp_this_turn.reach, cross-checks off)
//
// The first ply happens on the known board, so its cross-checks and tile
// availability apply. The second happens after an unknown opponent move, which
// can rewrite every cross-check and draws from a separate rack, so it is
// cross-check-free: it must never mask a footprint some opponent move makes
// legal. No move-gen, no main-word dictionary lookup, no joint multi-cell tile
// contention -- each ply is a sound over-approximation of its move.

using FootprintMask = std::array<bool, kFootprintClasses>;

// The sound tile budget every caller passes these masks: a mover holds at most
// RACK_SIZE tiles, so a full rack is the loosest cap and never masks a real move
// (a smaller, endgame-aware budget would only tighten -- see the TODO in
// training_targets.cpp). Shared so the per-row training masks and the .mset /
// dashboard collapse cannot drift.
inline constexpr int kMaskTileBudget = RACK_SIZE;

// A set of board squares, one bit per cell. A ply's seed and the squares it
// reaches are both SquareSets, so plies chain.
struct SquareSet {
  std::bitset<kFootprintCells> bits;

  bool contains(int idx) const { return bits.test(idx); }
  bool contains(int r, int c) const { return contains(r * kFootprintSide + c); }
  bool empty() const { return bits.none(); }
};

// S: the occupied squares.
SquareSet occupied_squares(const Board& board);

struct FootprintPly {
  FootprintMask mask;  // the ply's legal footprint classes
  SquareSet reach;     // the seed plus every square those footprints cover
};

// One ply from `seed`: the footprints of k <= budget tiles on empty squares
// that abut a seed square. With no seed at all (the opener's empty board)
// there is nothing to abut, so every fitting footprint is kept.
//   - use_cross_checks: gate each covered square on its cross-check -- some
//     letter must be legal there and, given available_counts, in stock. Off
//     for a ply on a board an unknown move will first rewrite.
//   - available_counts: the mover's pool as a 27-count array (A..Z then blank),
//     a blank being a wildcard; nullptr treats every tile as in stock. Ignored
//     when cross-checks are off. Sound -- never masks a tile the mover could
//     draw and play.
//   - win_head: keep kExtraClass (the win heads' not-win slot); false for a
//     plays head. kPassClass is always legal.
// `board` needs movegen caches only when cross-checks are on.
FootprintPly footprint_ply(const Board& board, const SquareSet& seed, int budget,
                           bool use_cross_checks, const uint8_t* available_counts, bool win_head);

// Legality for an OPPONENT placement head (opp_next / opp_win): the opponent
// moves next on `board`. footprint_ply(S, cross-checks on, available_counts).
void opp_footprint_mask(const Board& board, const uint8_t* available_counts, int tile_budget,
                        bool win_head, FootprintMask& mask);

// Legality for a SELF placement head (self_next / self_win): the mover plays
// after the opponent. The opponent's this-turn ply (opp_budget,
// opp_available_counts -- the pool it draws from) seeds the mover's
// cross-check-free one (self_budget).
void self_footprint_mask(const Board& board, int self_budget, int opp_budget,
                         const uint8_t* opp_available_counts, bool win_head, FootprintMask& mask);

// Per-cell reachability plane: out[r*kFootprintSide + c] == 1 iff some legal
// this-turn footprint under `available_counts` covers that cell, else 0
// (occupied squares are never covered). The input-feature view of
// footprint_ply(S, cross-checks on, available_counts); `board` must have
// movegen caches built. Writes exactly kFootprintCells floats.
void footprint_reachable_cells(const Board& board, const uint8_t* available_counts, int tile_budget,
                               float* out);

}  // namespace scribblez

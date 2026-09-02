#pragma once

#include "game/board.h"
#include "training/footprint.h"

#include <array>
#include <cstdint>

namespace scribblez {

// Per-class legality over the footprint classes, in the board's frame:
// mask[cls] == true iff the masked softmax should keep that class. Illegal
// classes are driven to -inf before the softmax, so their probability (and
// gradient) is zero.
//
// Every mask is built from one primitive, expand(): one ply of play from a seed
// set of cells, kept as the footprints that abut the seed. The seed is what a
// stage starts from -- the occupied squares S for a move on this board, or the
// squares a prior stage reached for a move on the board that stage will have
// extended. So:
//
//   opp_this_turn  = expand(S, cross-checks on, the opponent's pool)
//   self_this_turn = expand(S, cross-checks on, the mover's pool)   [input plane]
//   self_next_turn = expand(opp_this_turn.reach, cross-checks off)
//
// The first ply happens on the known board, so its cross-checks and tile
// availability apply. The second happens after an unknown opponent move, which
// can rewrite every cross-check and draws from a separate rack, so it is
// cross-check-free: it must never mask a footprint some opponent move makes
// legal. No move-gen, no main-word dictionary lookup, no joint multi-cell tile
// contention -- each stage is a sound over-approximation of its ply.

using FootprintMask = std::array<bool, kFootprintClasses>;

// The sound tile budget every caller passes these masks: a mover holds at most
// RACK_SIZE tiles, so a full rack is the loosest cap and never masks a real move
// (a smaller, endgame-aware budget would only tighten -- see the TODO in
// training_targets.cpp). Shared so the per-row training masks and the .mset /
// dashboard collapse cannot drift.
inline constexpr int kMaskTileBudget = RACK_SIZE;

// Per-cell tiles-to-reach: 0 on a seed cell, k on a cell some legal k-tile
// footprint of the stage covers (the fewest of any that do), kUnreachable
// otherwise. One stage's output is the next stage's seed, and the occupied
// squares are a seed too (occupied_reach), so every stage speaks this type.
struct Reach {
  static constexpr uint8_t kUnreachable = 0xFF;
  std::array<uint8_t, kFootprintCells> tiles;

  bool reachable(int idx) const { return tiles[idx] != kUnreachable; }
  bool reachable(int r, int c) const { return reachable(r * kFootprintSide + c); }
  bool empty() const;  // no reachable cell at all
};

// S: the occupied squares at depth 0, everything else unreachable.
Reach occupied_reach(const Board& board);

struct Expansion {
  FootprintMask mask;  // the stage's legal footprint classes
  Reach reach;         // `seed` plus every cell those footprints cover
};

// One ply from `seed`: the footprints of k <= budget tiles on empty squares
// that abut a seed cell. With no seed at all (the opener's empty board) there is
// nothing to abut, so every fitting footprint is kept.
//   - use_cross_checks: gate each covered cell on its cross-check -- some
//     letter must be legal there and, given available_counts, in stock. Off
//     for a ply on a board an unknown move will first rewrite.
//   - available_counts: the mover's pool as a 27-count array (A..Z then blank),
//     a blank being a wildcard; nullptr treats every tile as in stock. Ignored
//     when cross-checks are off. Sound -- never masks a tile the mover could
//     draw and play.
//   - win_head: keep kExtraClass (the win heads' not-win slot); false for a
//     plays head. kPassClass is always legal.
// `board` needs movegen caches only when cross-checks are on.
Expansion expand(const Board& board, const Reach& seed, int budget, bool use_cross_checks,
                 const uint8_t* available_counts, bool win_head);

// Legality for an OPPONENT placement head (opp_next / opp_win): the opponent
// moves next on `board`. expand(S, cross-checks on, available_counts).
void opp_footprint_mask(const Board& board, const uint8_t* available_counts, int tile_budget,
                        bool win_head, FootprintMask& mask);

// Legality for a SELF placement head (self_next / self_win): the mover plays
// after the opponent. The opponent's this-turn expansion (opp_budget,
// opp_available_counts -- the pool it draws from) becomes the seed of the
// mover's cross-check-free one (self_budget).
void self_footprint_mask(const Board& board, int self_budget, int opp_budget,
                         const uint8_t* opp_available_counts, bool win_head, FootprintMask& mask);

// Per-cell reachability plane: out[r*kFootprintSide + c] == 1 iff some legal
// this-turn footprint under `available_counts` covers that cell, else 0
// (occupied squares are never covered). The input-feature view of
// expand(S, cross-checks on, available_counts); `board` must have movegen
// caches built. Writes exactly kFootprintCells floats.
void footprint_reachable_cells(const Board& board, const uint8_t* available_counts, int tile_budget,
                               float* out);

}  // namespace scribblez

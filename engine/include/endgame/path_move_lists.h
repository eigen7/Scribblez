#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

// Per-tile-type counts packed into 4-bit fields (letters A..P in `lo`, Q..Z and
// then the blank in `hi`), for an O(1) subset test. Rack and per-move counts
// never exceed RACK_SIZE, so every field's high bit is free to act as the
// borrow sentinel in counts_subset's SWAR subtraction.
struct PackedCounts {
  uint64_t lo = 0;
  uint64_t hi = 0;
};

// High bit of every 4-bit field, the borrow sentinel of counts_subset.
inline constexpr uint64_t kCountsHighBits = 0x8888888888888888ull;

PackedCounts pack_rack(const Rack& rack);
PackedCounts pack_move_used(const Move& m);

// True iff `used` is a sub-multiset of `avail`. Setting each field's high bit
// before subtracting makes a field keep that bit exactly when its avail count
// covers its used count.
inline bool counts_subset(const PackedCounts& used, const PackedCounts& avail) {
  return (((avail.lo | kCountsHighBits) - used.lo) & kCountsHighBits) == kCountsHighBits &&
         (((avail.hi | kCountsHighBits) - used.hi) & kCountsHighBits) == kCountsHighBits;
}

// The lanes whose legal-move sets can change when `m`'s tiles land on a board:
// bit r of `rows` marks the horizontal lane of row r, bit c of `cols` the
// vertical lane of column c. A PASS places nothing and touches no lane.
struct LaneTouch {
  uint16_t rows = 0;
  uint16_t cols = 0;
};

// The lanes where playing `m` on `board` (as it stands before `m`) can change
// a play's legality, score, or single-tile dedup, or enable a new play. Every
// other lane's move set carries over verbatim, scores included.
LaneTouch move_lane_influence(const Board& board, const Move& m);

// The legal-play lists of both sides of an endgame search, maintained
// incrementally along the search path instead of regenerated at every node.
// A node's list is derived from the same side's list two plies up, regenerating
// only the lanes the two intervening moves touched. The result is identical to
// a scratch generation, order included.
//
// The derivation chain relies on the caller to seed both sides' root lists,
// call on_make for every move made, and request moves_at(ply) only after every
// shallower ply of the current path has been materialized. A depth-first search
// that generates before descending does this naturally.
class PathMoveLists {
 public:
  // `max_ply` bounds the plies on_make/moves_at will see.
  void reset(const Board* board, const Dictionary* dict, int max_ply);

  // Seed side `side`'s legal plays on the root board; side 0 moves first.
  void set_root_list(int side, const std::vector<Move>& plays);

  // `board` must still be in its pre-move state.
  void on_make(int ply, const Move& m);

  // The side-to-move's legal plays at the current position, `ply` deep down
  // the path; `rack` is that side's current rack.
  const std::vector<Move>& moves_at(int ply, const Rack& rack);

 private:
  // One materialized list, partitioned by lane in MoveGenerator::generate's
  // order (the 15 rows, then the 15 columns): lane L spans
  // [lane_begin[L], lane_begin[L + 1]). `used` holds each move's packed
  // used-tile counts for the rack-subset filter.
  struct Slot {
    std::vector<Move> moves;
    std::vector<PackedCounts> used;
    std::array<uint32_t, 2 * BOARD_SIZE + 1> lane_begin{};
  };

  static int lane_of(const Move& m) { return m.horizontal() ? m.start() : BOARD_SIZE + m.start(); }

  // Rebuild `s` from a lane-ordered scratch generation.
  static void fill_slot(Slot& s, const std::vector<Move>& plays);

  // Derive `out` from `parent`: regenerate `touched` lanes against `rack` on
  // the current board, carry the rest over (subset-filtered iff `filter`).
  void rebuild(Slot& out, const Slot& parent, const LaneTouch& touched, bool filter,
               const Rack& rack);

  const Board* board_ = nullptr;
  const Dictionary* dict_ = nullptr;
  Slot roots_[2];
  std::vector<Slot> slots_;       // indexed by ply; siblings overwrite freely
  std::vector<LaneTouch> masks_;  // influence of the move made at each ply
  std::vector<char> played_;      // the move at each ply was a PLAY (rack changed)
};

}  // namespace scribblez

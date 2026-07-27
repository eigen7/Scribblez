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

// True iff `used` is a sub-multiset of `avail`: per 4-bit field, seeding the
// high bit makes the subtraction borrow-free exactly when the field's avail
// count covers its used count (counts <= RACK_SIZE < 8 keep the bit clean).
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

// The lane-touch set of playing `m` on `board` (as it stands before `m`) -- the
// out-play halo's dual, coarsened to lanes: only in these lanes can a move
// change a play's legality, score, or single-tile dedup, or enable a new play.
//
// Derivation: a placed cell p affects exactly (a) plays placing a tile on p and
// (b) plays placing a tile on the first empty cell reached from p in each of
// the four directions by walking through pre-existing tiles -- in-line, the
// cell a word extends through p's run from; perpendicular, the cell whose
// cross-word p's run rewrites. Marking the rows and columns of p and of those
// (at most four) cells therefore covers every affected play, and every newly
// enabled play too, since it must place on such a cell or play through p.
// Cross-check, anchor, and dedup state in unmarked lanes is untouched, so their
// move sets carry over verbatim, scores included.
LaneTouch move_lane_influence(const Board& board, const Move& m);

// The legal-play lists of both sides of an endgame search, maintained
// incrementally along the search path instead of regenerated from scratch at
// every node. Lists are stored partitioned by lane (the 15 horizontal rows,
// then the 15 vertical columns -- MoveGenerator::generate's emission order),
// and a node's list is derived from the same side's list two plies up:
//
//   - lanes touched by neither intervening move (move_lane_influence) carry
//     over verbatim, filtered by a PackedCounts subset test against the
//     side's current rack when its own intervening move consumed tiles;
//   - touched lanes are dropped wholesale and regenerated on the current
//     board (MoveGenerator::generate_lane), which also produces every newly
//     enabled play -- the two sources are disjoint by construction, so no
//     deduplication is needed.
//
// The result is byte-identical to a scratch generation, in the same order, so
// a consumer cannot tell the difference. The caller owns the discipline the
// derivation chain relies on: seed both sides' root lists, call on_make for
// every move made, and request moves_at(ply) only once every shallower ply of
// the current path has been materialized the same way -- the natural shape of a
// depth-first search, where each node generates before descending. Slots are
// indexed by ply and overwritten freely as sibling subtrees revisit a depth.
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
  // One materialized list: moves partitioned by lane (CSR layout: lane L spans
  // [lane_begin[L], lane_begin[L + 1])), each move's used-tile multiset packed
  // alongside for the rack-subset filter.
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
  std::vector<Slot> slots_;       // indexed by ply
  std::vector<LaneTouch> masks_;  // influence of the move made at each ply
  std::vector<char> played_;      // the move at each ply was a PLAY (rack changed)
};

}  // namespace scribblez

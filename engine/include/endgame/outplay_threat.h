#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile_counts.h"

#include <array>
#include <cstdint>
#include <vector>

namespace scribblez {

// The blocking footprint of a candidate out-play `o` (a play that would empty a
// rack): the board cells whose occupation by another move could prevent `o` from
// being played later. Stored two ways -- cell-exact (to test whether a specific
// move touches it) and summarized as the rows and columns those cells occupy
// (for the pair-disjointness test below).
//
// The cell set is o's full main-word span (its placed and played-through cells),
// the orthogonal neighbors of its placed cells, and the one in-line cell just
// beyond each end of the span. This over-approximates the true blocking set, so
// every classification errs toward "blockable": a false "unblockable" -- the
// only direction that would make the pruning unsound -- cannot arise.
struct OutplayHalo {
  uint16_t rows = 0;                         // bit r set iff a halo cell lies in row r
  uint16_t cols = 0;                         // bit c set iff a halo cell lies in column c
  std::array<uint16_t, BOARD_SIZE> cells{};  // cells[r] bit c set iff (r, c) is a halo cell

  bool contains(int r, int c) const { return (cells[r] >> c) & 1u; }
};

// Build the halo for a PLAY `o` on `board` as it stands before `o` is played.
OutplayHalo build_outplay_halo(const Board& board, const Move& o);

// True iff no single move can touch both halos: they share no row and no column,
// and a move's placed tiles all lie in one row or one column, so no reply can
// disturb both out-plays at once.
inline bool halos_unblockable(const OutplayHalo& a, const OutplayHalo& b) {
  return (a.rows & b.rows) == 0 && (a.cols & b.cols) == 0;
}

// True iff any tile `m` places lies inside `halo` (so playing m would disturb
// the out-play the halo describes).
bool move_touches_halo(const OutplayHalo& halo, const Move& m);

// The canonical 7-bit position mask of the multiset `used` over `rack`'s sorted
// tiles: for each tile type, the lowest-indexed rack slots holding it, one per
// used copy. A function of the multiset alone, so a move's used tiles and a
// leave obtained by subtracting them from the rack map consistently.
uint8_t canonical_used_mask(const Rack& rack, const TileCounts& used);

// Computes, for the side to move at one search node, the out-play threat to hand
// each child node. The threat exists when the leave a move leaves behind holds
// two out-plays that no single opponent reply can both block: the opponent then
// cannot stop the mover from going out next turn, which bounds the opponent's
// value and lets the child prune dominated replies.
//
// One instance is built per node from the mover's legal plays; it buckets them
// by the leave they would empty and builds halos lazily on first use.
class OutplayThreats {
 public:
  // No threat: the value handed to a child whose parent move leaves no
  // unblockable out-play pair behind.
  static constexpr int32_t kNoThreat = -1;

  // `plays` are the mover's legal plays on `board`; `rack` is its full rack.
  OutplayThreats(const Board& board, const Rack& rack, const std::vector<Move>& plays);

  // The threat to hand the child reached by playing `m`: the largest guaranteed
  // smaller-score over the mover's unblockable out-play pairs for the leave `m`
  // leaves behind, or kNoThreat when `m` is a pass, empties the rack, or leaves
  // no such pair that survives `m`.
  int32_t threat_after(const Move& m);

 private:
  const OutplayHalo& halo(int play_index);

  const Board& board_;
  const Rack& rack_;
  const std::vector<Move>& plays_;
  TileCounts rack_counts_;
  std::vector<uint8_t> play_mask_;  // canonical used-mask of each play
  std::vector<OutplayHalo> halos_;  // built lazily; valid iff halo_ready_[i]
  std::vector<char> halo_ready_;
  std::vector<int> survivors_;  // scratch: candidates surviving the query move
};

}  // namespace scribblez

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
// rack): the board cells whose occupation by another move could prevent `o`
// from being played later. Stored both cell-exact and summarized as the rows
// and columns those cells occupy.
//
// The cell set is o's full main-word span, the ends of the existing
// perpendicular runs through its placed cells, and the one in-line cell just
// beyond each end of the span. This over-approximates the true blocking set, so
// every classification errs toward "blocked" -- a false "still playable", the
// only direction that would make pruning unsound, cannot arise.
struct OutplayHalo {
  uint16_t rows = 0;                         // bit r set iff a halo cell lies in row r
  uint16_t cols = 0;                         // bit c set iff a halo cell lies in column c
  std::array<uint16_t, BOARD_SIZE> cells{};  // cells[r] bit c set iff (r, c) is a halo cell

  bool contains(int r, int c) const { return (cells[r] >> c) & 1u; }
};

// `board` must be as it stands before `o` is played.
OutplayHalo build_outplay_halo(const Board& board, const Move& o);

// Whether playing `m` would disturb the out-play the halo describes.
bool move_touches_halo(const OutplayHalo& halo, const Move& m);

// The canonical 7-bit position mask of the multiset `used` over `rack`'s sorted
// tiles: per tile type, the lowest-indexed rack slots holding it, one per used
// copy. A function of the multiset alone, so a move's used tiles and the leave
// obtained by subtracting them from the rack map consistently.
uint8_t canonical_used_mask(const Rack& rack, const TileCounts& used);

// Halos for a fixed list of candidate out-plays on one board, each built on
// first use so a caller that queries only a few never pays for the rest.
class LazyHalos {
 public:
  LazyHalos(const Board& board, const std::vector<Move>& plays);

  const OutplayHalo& operator[](int play_index);

 private:
  const Board& board_;
  const std::vector<Move>& plays_;
  std::vector<OutplayHalo> halos_;  // valid iff ready_[i]
  std::vector<char> ready_;
};

// One out-play a side could make, paired with the halo built on the board it
// was collected on. The halo stays a sound kill trigger as the search board
// evolves: any sequence of moves that disturbs the out-play's legality or score
// must first place a tile in the halo, so dropping an entry the moment a move
// touches its halo guarantees every survivor is still legal at the same score.
// The price is false drops, which only weaken the bounds derived from the set.
struct OutplayEntry {
  Move move;
  OutplayHalo halo;
};

// A side's current candidate out-plays down one search path, sorted by
// descending score so a query usually stops at its first (best) survivor.
using OutplaySet = std::vector<OutplayEntry>;

// best_surviving_score's result when the queried move disturbs every out-play.
constexpr int32_t kNoOutplaySurvivor = -1;

// The highest score among out-plays in `set` that survive `m`, or
// kNoOutplaySurvivor when none does. A pass places nothing, so it survives all.
int32_t best_surviving_score(const OutplaySet& set, const Move& m);

// Rebuild `out` as the entries of `parent` that survive `m`, preserving order.
void assign_surviving(const OutplaySet& parent, const Move& m, OutplaySet& out);

// Rebuild `out` as the out-plays among `plays` (which empty a `rack_size`-tile
// rack) on `board`, halos built eagerly.
void collect_rack_outplays(const Board& board, const std::vector<Move>& plays, int rack_size,
                           OutplaySet& out);

// The out-play set to hand each child of one search node: the out-plays of the
// leave a candidate move keeps, as they stand on the node's own board. Any play
// of a rack subset is already in the node's legal-play list, so bucketing that
// list by used-tile multiset finds every leave's out-plays with no extra move
// generation. Built once per node.
class LeaveOutplays {
 public:
  // `plays` are the mover's legal plays on `board`; `rack` is its full rack.
  LeaveOutplays(const Board& board, const Rack& rack, const std::vector<Move>& plays);

  // Rebuild `out` as the out-plays of the leave `m` keeps that survive `m`
  // itself (a pass keeps the whole rack, so its out-plays are the mover's own).
  // Empty when `m` empties the rack.
  void collect_after(const Move& m, OutplaySet& out);

 private:
  const Rack& rack_;
  const std::vector<Move>& plays_;
  TileCounts rack_counts_;
  std::vector<uint8_t> play_mask_;  // canonical used-mask of each play
  LazyHalos halos_;
};

// The out-play sets in effect at one node of a search, one per seat: the
// candidates that node's futility bounds are read from. A mover pushes the pair
// its children read and pops it on the way back up, so the stack follows the
// descent. Only the replier's half of the root pair is ever populated -- the
// root's own moves are bounded against the replier's out-plays, never against
// its own.
class OutplaySetStack {
 public:
  // Start a fresh search, sizing the per-ply slots for paths of at most
  // `max_ply` plies.
  void reset(int max_ply);

  // Make the out-plays of `plays` (the replier's legal plays with a
  // `rack_size`-tile rack) on the root `board` the set the root's moves are
  // bounded against.
  void collect_root_replier(const Board& board, const std::vector<Move>& plays, int rack_size);
  const OutplaySet& root_replier() const { return root_[1]; }

  const OutplaySet& current(int seat) const { return *cur_[seat]; }

  // Install the sets the child that `m` reaches at `child_ply` will read, where
  // `stm` is the seat playing `m`. `leave_outs` buckets the node's own move
  // list, the source of the mover's next set; passing nullptr leaves the
  // current sets in place, for a search that reads none of them.
  void push(const Move& m, LeaveOutplays* leave_outs, int stm, int child_ply);

  // Restore the sets the matching push() displaced.
  void pop(int child_ply);

 private:
  // One ply's sets, plus the pair push() displaced to install them. Siblings
  // share a slot, since only one is ever being searched.
  struct Slot {
    OutplaySet mover, other;
    OutplaySet* saved[2] = {nullptr, nullptr};
  };

  OutplaySet root_[2];
  std::vector<Slot> slots_;  // indexed by the child's ply
  OutplaySet* cur_[2] = {nullptr, nullptr};
};

}  // namespace scribblez

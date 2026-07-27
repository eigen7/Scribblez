#pragma once

#include "game/tile.h"

#include <array>
#include <string>

namespace scribblez {

// A multiset of tiles as a per-type histogram -- the fast representation for
// counting, adding, and removing. (A Tile indexes it via its implicit
// conversion.)
class TileCounts {
 public:
  void add(Tile t) { ++counts_[t]; }
  bool remove(Tile t);
  int count(Tile t) const { return counts_[t]; }
  int blanks() const { return counts_[BLANK]; }

  int size() const;
  bool empty() const { return size() == 0; }

  // Blanks count as 0.
  int point_value() const;

  // Letters in alphabetical order followed by '?' for each blank.
  std::string to_string() const;

 private:
  std::array<int, 27> counts_{};  // index 0..25 = A..Z, 26 = blank
};

}  // namespace scribblez

#include "inlines/game/tile_counts.inl"

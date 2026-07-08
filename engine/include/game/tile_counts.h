#pragma once

#include "game/tile.h"

#include <array>
#include <string>

namespace scribblez {

// A multiset of tiles as a per-type histogram: how many of each letter A..Z and
// how many blanks. This is the fast representation for counting / adding /
// removing tiles -- e.g. movegen's available-tile scratch, or a Rack's contents
// viewed as counts. (A Tile indexes the histogram via its implicit conversion.)
class TileCounts {
 public:
  void add(Tile t) { ++counts_[t]; }
  bool remove(Tile t);
  int count(Tile t) const { return counts_[t]; }
  int blanks() const { return counts_[BLANK]; }

  int size() const;
  bool empty() const { return size() == 0; }

  // Sum of tile values (blanks count as 0).
  int point_value() const;

  // Letters in alphabetical order followed by '?' for each blank.
  std::string to_string() const;

 private:
  std::array<int, 27> counts_{};  // index 0..25 = A..Z, 26 = blank
};

}  // namespace scribblez

#include "inlines/game/tile_counts.inl"

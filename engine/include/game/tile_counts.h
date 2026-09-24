#pragma once

#include "game/tile.h"

#include <array>
#include <string>
#include <string_view>

namespace scribblez {

// Rack's counterpart: tiles as a per-type histogram, trading Rack's
// compactness for O(1) count().
class TileCounts {
 public:
  // Every tile of the game: TILE_COUNTS as a histogram.
  static TileCounts full_distribution();
  // Tiles from their letters: A..Z in either case, '?' for a blank. Throws
  // util::Exception on any other character.
  static TileCounts from_string(std::string_view letters);

  void add(Tile t) { ++counts_[t]; }
  void add(Tile t, int n) { counts_[t] += n; }
  bool remove(Tile t);
  // Removes each of `other`'s tiles. False if some tile ran short, whose count
  // then stops at 0.
  bool remove(const TileCounts& other);
  int count(Tile t) const { return counts_[t]; }
  int blanks() const { return counts_[BLANK]; }

  int size() const;
  bool empty() const { return size() == 0; }

  // Blanks count as 0.
  int point_value() const;

  // Letters in alphabetical order followed by '?' for each blank.
  std::string to_string() const;

 private:
  std::array<int, TILE_KINDS> counts_{};  // index 0..25 = A..Z, 26 = blank
};

}  // namespace scribblez

#include "inlines/game/tile_counts.inl"

#pragma once

#include "scribblez/tile.h"
#include "scribblez/tile_counts.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

namespace scribblez {

// A Rack holds up to RACK_SIZE tiles. It stores them as a sorted, fixed-size
// array (A..Z then blanks, unused slots empty) -- compact and trivially
// serializable. For histogram-style bookkeeping, ask for counts().
class Rack {
 public:
  void add(Tile t);     // insert, keeping the array sorted
  bool remove(Tile t);  // remove one occurrence; false if absent
  int count(Tile t) const;
  int blanks() const { return count(BLANK); }

  int size() const { return size_; }
  bool empty() const { return size_ == 0; }

  // Letters in alphabetical order followed by '?' for each blank.
  std::string to_string() const;

  // Sum of tile values currently on the rack (blanks count as 0).
  int point_value() const;

  // Histogram view (e.g. for movegen's scratch).
  TileCounts counts() const;

  const std::array<Tile, RACK_SIZE>& tiles() const { return tiles_; }

  // DIAGNOSTIC: bits-based comparators on C++20
  uint64_t bits() const {
    uint64_t b;
    std::memcpy(&b, this, sizeof(b));
    return b;
  }
  bool operator<(const Rack& o) const { return bits() < o.bits(); }
  bool operator==(const Rack& o) const { return bits() == o.bits(); }

 private:
  std::array<Tile, RACK_SIZE> tiles_{};  // sorted ascending; unused slots empty
  int8_t size_ = 0;
};

static_assert(sizeof(Rack) == 8, "Rack should pack into 8 bytes");

}  // namespace scribblez

template <>
struct std::hash<scribblez::Rack> {
  size_t operator()(const scribblez::Rack& r) const noexcept {
    return std::hash<uint64_t>()(r.bits());
  }
};

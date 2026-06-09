#pragma once

#include "scribblez/tile.h"
#include "scribblez/tile_counts.h"

#include <array>
#include <cstdint>
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

  // Comparisons and hashing operate on the 8-byte object representation (see
  // bits()). Because the tile array is kept sorted with trailing empty slots,
  // two racks with the same multiset have identical bytes -- so byte equality
  // is exactly multiset equality, and the byte ordering is an arbitrary but
  // consistent total order (all we need for sorting / run-grouping).
  bool operator==(const Rack& o) const { return bits() == o.bits(); }
  bool operator<(const Rack& o) const { return bits() < o.bits(); }

  // The 8-byte object representation, copied out via memcpy: defined behavior
  // (Rack is trivially copyable) that the optimizer lowers to a single load.
  uint64_t bits() const;

 private:
  std::array<Tile, RACK_SIZE> tiles_{};  // sorted ascending; unused slots empty
  int8_t size_ = 0;
};

static_assert(sizeof(Rack) == 8, "Rack should pack into 8 bytes");

}  // namespace scribblez

template <>
struct std::hash<scribblez::Rack> {
  // A Rack's bits() is its 8-byte sorted representation. std::hash<uint64_t>
  // is the identity on the standard libraries, so returning bits() directly is
  // equivalent and avoids the wrapper.
  size_t operator()(const scribblez::Rack& r) const noexcept {
    return static_cast<size_t>(r.bits());
  }
};

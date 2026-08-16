#pragma once

#include "game/tile.h"
#include "game/tile_counts.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>

namespace scribblez {

// Up to RACK_SIZE tiles, held sorted in a fixed-size array so the type stays
// compact and trivially serializable. TileCounts is the histogram counterpart.
class Rack {
 public:
  void add(Tile t);
  bool remove(Tile t);  // false if absent
  int count(Tile t) const;
  int blanks() const { return count(BLANK); }

  int size() const { return size_; }
  bool empty() const { return size_ == 0; }

  // Letters in alphabetical order followed by '?' for each blank.
  std::string to_string() const;

  // Blanks count as 0.
  int point_value() const;

  TileCounts counts() const;

  const std::array<Tile, RACK_SIZE>& tiles() const { return tiles_; }

  // Comparison and hashing run on the object representation. The array is kept
  // sorted with trailing empty slots, so equal multisets have identical bytes:
  // byte equality is exactly multiset equality, and the byte ordering is an
  // arbitrary but consistent total order.
  bool operator==(const Rack& o) const { return bits() == o.bits(); }
  bool operator<(const Rack& o) const { return bits() < o.bits(); }

  // Copied out via memcpy: defined behavior (Rack is trivially copyable) that
  // the optimizer lowers to a single load.
  uint64_t bits() const;

 private:
  std::array<Tile, RACK_SIZE> tiles_{};  // sorted ascending; unused slots empty
  int8_t size_ = 0;
};

static_assert(sizeof(Rack) == 8, "Rack should pack into 8 bytes");

}  // namespace scribblez

template <>
struct std::hash<scribblez::Rack> {
  // std::hash<uint64_t> is the identity on the standard libraries, so bits()
  // is equivalent and avoids the wrapper.
  size_t operator()(const scribblez::Rack& r) const noexcept { return size_t(r.bits()); }
};

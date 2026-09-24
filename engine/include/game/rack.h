#pragma once

#include "game/tile.h"
#include "game/tile_counts.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace scribblez {

// Up to RACK_SIZE tiles, kept sorted in a fixed 8-byte layout that doubles as
// a canonical multiset key. TileCounts is the histogram counterpart.
class Rack {
 public:
  // `counts` as a rack. Throws util::Exception if it holds more than
  // RACK_SIZE tiles.
  static Rack from_counts(const TileCounts& counts);
  // TileCounts::from_string as a rack; throws as it and from_counts do.
  static Rack from_string(std::string_view letters) {
    return from_counts(TileCounts::from_string(letters));
  }
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

  // Comparison and hashing use the raw bytes. Sorted tiles with empty trailing
  // slots make equal multisets byte-identical, so byte equality is multiset
  // equality, and operator< is an arbitrary but consistent total order.
  bool operator==(const Rack& o) const { return bits() == o.bits(); }
  bool operator<(const Rack& o) const { return bits() < o.bits(); }

  // The object representation as one integer (a memcpy, which compiles to a
  // single load).
  uint64_t bits() const;

 private:
  std::array<Tile, RACK_SIZE> tiles_{};  // sorted ascending; unused slots empty
  int8_t size_ = 0;
};

static_assert(sizeof(Rack) == 8, "Rack should pack into 8 bytes");

}  // namespace scribblez

template <>
struct std::hash<scribblez::Rack> {
  size_t operator()(const scribblez::Rack& r) const noexcept { return r.bits(); }
};

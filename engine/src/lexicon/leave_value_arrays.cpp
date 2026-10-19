#include "lexicon/leave_value_arrays.h"

#include "belief/leave_prior.h"
#include "game/tile_counts.h"
#include "util/exception.h"

#include <algorithm>

namespace scribblez {

namespace {

// `leave`'s tiles in ascending letter order.
std::vector<Tile> sorted_tiles(const Rack& leave) {
  std::vector<Tile> tiles;
  for (Tile t : leave.tiles()) {
    if (!t.is_empty()) tiles.push_back(t);
  }
  std::sort(tiles.begin(), tiles.end(), [](Tile a, Tile b) { return a.index() < b.index(); });
  return tiles;
}

}  // namespace

LeaveValueArrays::LeaveValueArrays(const LeaveValues& values)
    : content_hash_(values.content_hash()) {
  values.for_each([&](const Rack& leave, float value) {
    const int k = leave.size();
    if (k < 1 || k > kLeaveArraysMaxLength) return;
    Length& length = lengths_[k - 1];
    length.values.push_back(value);
    const std::vector<Tile> tiles = sorted_tiles(leave);
    for (int slot = 0, copy = 0; slot < k; ++slot) {
      copy = slot > 0 && tiles[slot].index() == tiles[slot - 1].index() ? copy + 1 : 0;
      length.codes[slot].push_back(code(tiles[slot], copy));
    }
  });
  for (int k = 1; k <= kLeaveArraysMaxLength; ++k) {
    // The model normalizes over these rows, so a leave missing from the table
    // would silently change every probability.
    const int64_t drawable =
      belief::count_multisets(TileCounts::full_distribution(), k, /*cap=*/int64_t(1) << 40);
    if (int64_t(rows(k)) != drawable) {
      throw util::Exception("leave values have {} leaves of length {}, but {} are drawable",
                            rows(k), k, drawable);
    }
  }
}

const LeaveValueArrays::Length& LeaveValueArrays::lengths(int k) const {
  if (k < 1 || k > kLeaveArraysMaxLength) throw util::Exception("no leaves of length {}", k);
  return lengths_[k - 1];
}

std::span<const uint8_t> LeaveValueArrays::codes(int k, int slot) const {
  if (slot < 0 || slot >= k) throw util::Exception("no tile slot {} in a length-{} leave", slot, k);
  return lengths(k).codes[slot];
}

}  // namespace scribblez

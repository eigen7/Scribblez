#include "lexicon/leave_value_arrays.h"

#include "belief/leave_prior.h"
#include "game/tile_counts.h"
#include "util/exception.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace scribblez {

namespace {

LeaveValueRow make_row(const Rack& leave, float value) {
  const float centi = std::round(value * LeaveValueRow::kValueScale);
  if (std::abs(centi) > float(std::numeric_limits<int16_t>::max())) {
    throw util::Exception("leave value {} does not fit LeaveValueRow", value);
  }
  LeaveValueRow row{};
  row.tiles.fill(LeaveValueRow::kNoTile);
  int n = 0;
  for (Tile t : leave.tiles()) {
    if (!t.is_empty()) row.tiles[n++] = uint8_t(t.index());
  }
  std::sort(row.tiles.begin(), row.tiles.begin() + n);
  row.centi_value = int16_t(centi);
  return row;
}

bool by_value_then_tiles(const LeaveValueRow& a, const LeaveValueRow& b) {
  return std::tie(a.centi_value, a.tiles) < std::tie(b.centi_value, b.tiles);
}

}  // namespace

LeaveValueArrays::LeaveValueArrays(const LeaveValues& values)
    : content_hash_(values.content_hash()) {
  values.for_each([&](const Rack& leave, float value) {
    const int k = leave.size();
    if (k >= 1 && k <= kLeaveArraysMaxLength) rows_[k - 1].push_back(make_row(leave, value));
  });
  for (int k = 1; k <= kLeaveArraysMaxLength; ++k) {
    // The model normalizes over these rows, so a leave missing from the table
    // would silently change every probability.
    const int64_t drawable =
      belief::count_multisets(TileCounts::full_distribution(), k, /*cap=*/int64_t(1) << 40);
    if (int64_t(rows_[k - 1].size()) != drawable) {
      throw util::Exception("leave values have {} leaves of length {}, but {} are drawable",
                            rows_[k - 1].size(), k, drawable);
    }
    std::sort(rows_[k - 1].begin(), rows_[k - 1].end(), by_value_then_tiles);
  }
}

std::span<const LeaveValueRow> LeaveValueArrays::of_length(int k) const {
  if (k < 1 || k > kLeaveArraysMaxLength) throw util::Exception("no leaves of length {}", k);
  return rows_[k - 1];
}

}  // namespace scribblez

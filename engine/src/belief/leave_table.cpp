#include "belief/leave_table.h"

#include "belief/leave_prior.h"
#include "game/tile_counts.h"
#include "util/exception.h"

namespace scribblez::belief {

namespace {

LeaveTableRow make_row(const Rack& leave, float value) {
  LeaveTableRow row{};
  const TileCounts counts = leave.counts();
  for (int l = 0; l < TILE_KINDS; ++l) {
    row.counts[l] = uint8_t(counts.count(Tile::of(l)));
    if (row.counts[l] > 0) row.mask |= 1u << l;
  }
  row.value = value;
  return row;
}

}  // namespace

LeaveTable::LeaveTable(const HastyEquity& equity) : klv_hash_(equity.klv_hash()) {
  for (int k = 1; k <= kLeaveTableMaxLength; ++k) {
    for (const ScoredLeave& s : enumerate_leaves(TileCounts::full_distribution(), k))
      rows_.push_back(make_row(s.leave, float(equity.leave_value(s.leave))));
    offsets_[k] = rows_.size();
  }
}

std::span<const LeaveTableRow> LeaveTable::rows(int k) const {
  if (k < 1 || k > kLeaveTableMaxLength) throw util::Exception("no leave table for length {}", k);
  return std::span<const LeaveTableRow>(rows_).subspan(offsets_[k - 1],
                                                       offsets_[k] - offsets_[k - 1]);
}

}  // namespace scribblez::belief

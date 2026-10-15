#pragma once

// The leave table: every leave drawable from a full bag, for each length
// 1..6, with its tile counts, presence mask and static leave value -- the rows
// the leave inference model (docs/plans/deep_bot.md, "The leave inference
// model") scores, sums and samples over. A 7-tile leave arises only from a
// pass and is drawn from the prior alone, so it has no rows.
//
// Built in memory, by this one constructor, from HastyEquity's leave values;
// the trainer reaches it through the FFI (scribblez_leave_table_new). Every
// consumer therefore sees the same rows in the same order, which sampling's
// pairing across candidates and the parity tests depend on. Its content is
// fixed by the leave values (klv_hash, which models record) and by this
// enumeration (which kLeaveInferenceFormatVersion covers).

#include "game/tile.h"
#include "lexicon/hasty_equity.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace scribblez::belief {

inline constexpr int kLeaveTableMaxLength = RACK_SIZE - 1;

struct LeaveTableRow {
  std::array<uint8_t, TILE_KINDS> counts;  // indexed by Tile code; the blank is 26
  uint32_t mask;                           // bit l set iff counts[l] > 0
  float value;                             // HastyEquity::leave_value
};

class LeaveTable {
 public:
  // From the initialized `equity`'s leave values; rows of each length in
  // enumerate_leaves order (belief/leave_prior.h).
  explicit LeaveTable(const HastyEquity& equity);

  // The content hash of the .klv2 the values came from.
  const std::string& klv_hash() const { return klv_hash_; }

  // The length-k rows, 1 <= k <= kLeaveTableMaxLength.
  std::span<const LeaveTableRow> rows(int k) const;

 private:
  std::string klv_hash_;
  std::vector<LeaveTableRow> rows_;                         // every length, length 1 first
  std::array<size_t, kLeaveTableMaxLength + 1> offsets_{};  // rows of length k start at [k - 1]
};

}  // namespace scribblez::belief

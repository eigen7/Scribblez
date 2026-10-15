#pragma once

// The same leaves and values as LeaveValues, laid out for a different query.
// LeaveValues answers "what is this leave worth?" by lookup; LeaveValueArrays
// holds, for each length 1..6, every leave in one flat array, for the passes
// that scan all leaves of a length: the leave inference model's likelihood,
// and sampling from it (docs/plans/deep_bot.md, "The leave inference model").
// A 7-tile leave arises only from a pass; passes are rare enough that the
// model does not try to model them and assumes a uniform draw instead, so
// there are no 7-tile rows.
//
// Built in memory from a LeaveValues; the trainer reaches it through the FFI
// (scribblez_leave_value_arrays_new). Every consumer therefore sees the same
// rows in the same order, which sampling's pairing across candidates and the
// parity tests depend on.

#include "game/tile.h"
#include "lexicon/leave_values.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace scribblez {

inline constexpr int kLeaveArraysMaxLength = RACK_SIZE - 1;

// One leave and its value in 8 bytes; there are 914,624 of them. The length
// is implicit, since each length has its own array.
struct LeaveValueRow {
  static constexpr uint8_t kNoTile = 0xFF;
  // Values are stored in hundredths of a point: ample precision for a leave's
  // value, and a range of +-327 points.
  static constexpr float kValueScale = 100.0f;

  std::array<uint8_t, kLeaveArraysMaxLength> tiles;  // Tile codes ascending, then kNoTile
  int16_t centi_value;

  float value() const { return float(centi_value) / kValueScale; }
};
static_assert(sizeof(LeaveValueRow) == 8, "LeaveValueRow must pack into 8 bytes");

class LeaveValueArrays {
 public:
  // Every leave of `values`, by length. Within a length, rows are sorted by
  // value, ties by tiles, so the order is fixed. Throws util::Exception if
  // `values` lacks a leave drawable from a full bag, or on a value outside
  // int16 hundredths.
  explicit LeaveValueArrays(const LeaveValues& values);

  // LeaveValues::content_hash: which leave values these are.
  const std::string& content_hash() const { return content_hash_; }

  // The length-k leaves, 1 <= k <= kLeaveArraysMaxLength. The passes scan them
  // whole; nothing searches them by value.
  std::span<const LeaveValueRow> of_length(int k) const;

 private:
  std::string content_hash_;
  std::array<std::vector<LeaveValueRow>, kLeaveArraysMaxLength> rows_;  // [k - 1]
};

}  // namespace scribblez

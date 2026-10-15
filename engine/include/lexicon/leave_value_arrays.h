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

class LeaveValueArrays {
 public:
  // One leave and its value in 8 bytes; there are 914,624 of them. The length
  // is implicit, since each length has its own array. The value is in
  // hundredths of a point: ample precision, and a range of +-327 points.
  struct Row {
    static constexpr uint8_t kNoTile = 0xFF;

    std::array<uint8_t, kLeaveArraysMaxLength> tiles;  // Tile codes ascending, then kNoTile
    int16_t centi_value;

    float value() const { return float(centi_value) * 0.01f; }
  };
  static_assert(sizeof(Row) == 8, "a Row must pack into 8 bytes");

  // Every leave of `values`, by length, each length in `values`' own order
  // (LeaveValues::for_each), so the order is fixed by the leave values file.
  // Throws util::Exception if `values` lacks a leave drawable from a full
  // bag, or on a value outside int16 hundredths.
  explicit LeaveValueArrays(const LeaveValues& values);

  // LeaveValues::content_hash: which leave values these are, and so which
  // rows in which order.
  const std::string& content_hash() const { return content_hash_; }

  // The length-k leaves, 1 <= k <= kLeaveArraysMaxLength. The passes scan them
  // whole; nothing searches them by value.
  std::span<const Row> of_length(int k) const;

 private:
  std::string content_hash_;
  std::array<std::vector<Row>, kLeaveArraysMaxLength> rows_;  // [k - 1]
};

}  // namespace scribblez

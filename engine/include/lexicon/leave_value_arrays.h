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
// Each length is a structure of arrays: the values as floats, and one byte
// array per tile slot holding that tile's code (below), so a pass streams
// contiguous arrays. A row is 4 + k bytes.
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
  // A tile's code within a sorted leave: its letter times kCopiesPerTile, plus
  // how many copies of that letter precede it in the leave. A leave is the set
  // of its tiles' codes, so the codes carry its multiplicities (which the pool
  // prior weighs) without per-letter counts, and code letter * kCopiesPerTile
  // marks the letter's presence.
  static constexpr int kCopiesPerTile = kLeaveArraysMaxLength;
  static constexpr int kCodes = TILE_KINDS * kCopiesPerTile;  // 162
  static_assert(kCodes <= 256, "a code must fit a byte");

  static constexpr uint8_t code(Tile letter, int copy) {
    return uint8_t(letter.index() * kCopiesPerTile + copy);
  }

  // Every leave of `values`, by length, each length in `values`' own order
  // (LeaveValues::for_each), so the order is fixed by the leave values file.
  // Throws util::Exception if `values` lacks a leave drawable from a full
  // bag.
  explicit LeaveValueArrays(const LeaveValues& values);

  // LeaveValues::content_hash: which leave values these are, and so which
  // rows in which order.
  const std::string& content_hash() const { return content_hash_; }

  // The number of length-k leaves, 1 <= k <= kLeaveArraysMaxLength.
  size_t rows(int k) const { return lengths(k).values.size(); }

  // The length-k leaves' values, in points.
  std::span<const float> values(int k) const { return lengths(k).values; }

  // The code of tile `slot` (0 <= slot < k) of every length-k leave, the
  // leave's tiles taken in ascending letter order. The passes scan these
  // arrays whole; nothing searches them by value.
  std::span<const uint8_t> codes(int k, int slot) const;

 private:
  // One length's leaves, as parallel arrays.
  struct Length {
    std::vector<float> values;
    std::array<std::vector<uint8_t>, kLeaveArraysMaxLength> codes;  // [slot]; slots 0..k-1
  };

  // Throws util::Exception for k outside 1..kLeaveArraysMaxLength.
  const Length& lengths(int k) const;

  std::string content_hash_;
  std::array<Length, kLeaveArraysMaxLength> lengths_;  // [k - 1]
};

}  // namespace scribblez

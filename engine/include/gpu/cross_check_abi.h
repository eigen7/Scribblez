#pragma once

// A candidate move's cross-check changes (training/cross_check_delta.h) as
// both the CPU reference and the CUDA kernel (gpu/cross_check_deltas_gpu.h)
// write them: plain structs with no engine dependencies, so one header
// compiles under g++ (C++26) and nvcc alike, and the two sides' outputs
// compare field by field.

#include "gpu/rollout_abi.h"

#include <cstdint>

namespace scribblez::cross_check_abi {

// n placed tiles extend n perpendicular runs (two ends each), plus the two ends
// of the word along the play axis.
constexpr int kMaxDeltas = 2 * rollout_abi::kRackSize + 2;

// One (axis, square) whose letter mask the move changes.
struct Delta {
  uint32_t old_mask = 0;  // the pre-move CrossCheck::mask
  uint32_t new_mask = 0;  // the post-move one
  uint16_t square = 0;    // r * kBoardSize + c, in the board's frame
  // 0: the horizontal-play cross-check block (Board::cross_checks(false)),
  // 1: the vertical-play one (the kCrossChecks plane halves).
  uint8_t axis = 0;
  uint8_t pad = 0;

  bool operator==(const Delta&) const = default;
};

struct MoveDeltas {
  Delta deltas[kMaxDeltas];  // [0, count) sorted by (axis, square); the rest zero
  int32_t count = 0;

  bool operator==(const MoveDeltas&) const = default;
};

// Adds `x` to d's entries, keeping them in (axis, square) order: how both
// sides order them.
SCRIBBLEZ_HD inline void insert_sorted(MoveDeltas& d, const Delta& x) {
  int i = d.count++;
  for (; i > 0; --i) {
    const Delta& prev = d.deltas[i - 1];
    if (prev.axis < x.axis || (prev.axis == x.axis && prev.square < x.square)) break;
    d.deltas[i] = prev;
  }
  d.deltas[i] = x;
}

}  // namespace scribblez::cross_check_abi

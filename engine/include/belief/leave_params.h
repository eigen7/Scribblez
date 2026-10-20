#pragma once

// The leave inference model's output (docs/plans/deep_bot.md, "The leave
// inference model"): a tilt of the hypergeometric prior over the leaves of a
// known length drawable from the leave pool,
//
//   log p(l) = base(l; pool) + Σ_t theta_t [t in l] + a v(l) - b v(l)^2 - log Z
//
// where base is the pool's log prior and v(l) the leave's value in points.
// The exported graph's `leave_params` row is this struct's 29 floats in order
// (py/scribblez/leave_inference/onnx_export.py), a layout
// kLeaveInferenceFormatVersion covers.

#include "game/tile.h"

#include <array>

namespace scribblez::belief {

struct LeaveParams {
  static constexpr int kFloats = TILE_KINDS + 2;

  std::array<float, TILE_KINDS> theta;  // per tile kind's presence, by Tile code
  float a;                              // per point of leave value
  float b;                              // per squared point, >= 0
};

// A row of the exported output copies straight in.
static_assert(sizeof(LeaveParams) == sizeof(float) * LeaveParams::kFloats);

}  // namespace scribblez::belief

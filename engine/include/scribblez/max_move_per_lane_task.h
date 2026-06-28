#pragma once

#include "scribblez/encode_context.h"
#include "scribblez/lane_targets.h"
#include "scribblez/max_move_per_lane_input_encoder.h"

namespace scribblez {

// The "highest-scoring move per lane" task: the lean board + rack input
// (MaxMovePerLaneInputEncoder) plus the per-lane occupancy / score / mask labels.
// The labels require enumerating legal moves at the position, so this task reads
// the lexicon from the context's `dict`.
struct MaxMovePerLaneTask {
  static constexpr int kInputFloats = MaxMovePerLaneInputEncoder::kInputFloats;
  static constexpr int kLabelFloats = kLaneLabelFloats;
  static constexpr int kRowFloats = kInputFloats + kLabelFloats;

  static void encode_row(const EncodeContext& ctx, float* out_row);
};

}  // namespace scribblez

#pragma once

#include "encoding/encode_context.h"
#include "training/lane_targets.h"
#include "training/max_move_per_lane_input_encoder.h"

namespace scribblez {

// The max-move-per-lane training task: MaxMovePerLaneInputEncoder's input plus
// the lane_targets.h labels. The labels come from generating every legal move at
// the position, using the lexicon in ctx.spec.dict.
struct MaxMovePerLaneTask {
  static constexpr int kInputFloats = MaxMovePerLaneInputEncoder::kInputFloats;
  static constexpr int kLabelFloats = kLaneLabelFloats;
  static constexpr int kRowFloats = kInputFloats + kLabelFloats;

  static void encode_row(const EncodeContext& ctx, float* out_row);
};

}  // namespace scribblez

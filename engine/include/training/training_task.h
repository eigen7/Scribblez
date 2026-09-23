#pragma once

#include "encoding/encode_context.h"
#include "encoding/input_encoder.h"
#include "training/training_targets.h"  // kLabelFloats, AllTargets

namespace scribblez {

// A training task turns an EncodeContext into one training row: an input
// encoding followed by labels. PositionEncoder replays a game to a sampled
// position and hands the context to Task::encode_row, so one replay pipeline
// serves several training problems; docs/architecture.md describes the replay.
// MaxMovePerLaneTask is the other task.

// The position evaluation task: the input_encoder.h input, then AllTargets.
struct PositionEvalTask {
  static int row_floats(const InputEncodingSpec& spec);

  static void encode_row(const EncodeContext& ctx, float* out_row);
};

}  // namespace scribblez

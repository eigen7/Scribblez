#pragma once

#include "encoding/encode_context.h"
#include "encoding/input_encoder.h"
#include "training/training_targets.h"  // kLabelFloats, AllTargets

namespace scribblez {

// A TrainingTask bundles an input encoding and a set of labels into one training
// row. PositionEncoder replays a game to a sampled position, fills an
// EncodeContext, and hands it to the task's encode_row. Parameterizing the
// encoder on the task lets one replay/streaming pipeline serve several training
// problems.

// The win-probability task: the post-move board/leave/score input plus the WLD,
// score-diff, and placement labels.
struct PositionEvalTask {
  static int row_floats(const InputEncodingSpec& spec);

  static void encode_row(const EncodeContext& ctx, float* out_row);
};

}  // namespace scribblez

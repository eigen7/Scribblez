#pragma once

#include "encoding/encode_context.h"
#include "encoding/input_encoder.h"
#include "training/training_targets.h"  // kLabelFloats, AllTargets

namespace scribblez {

// A TrainingTask bundles an input encoding and a set of labels into one training
// row. PositionEncoder replays a game to a sampled position, fills an
// EncodeContext, and hands it to the task's encode_row, which writes the input
// floats followed by the label floats. Parameterizing the encoder on the task
// lets one replay/streaming pipeline serve several training problems.

// The win-probability task: the post-move board/leave/score input (via the
// stateful GameStateEncoder the context points at) plus the WLD, score-diff, and
// opponent-next-placement labels. The input width is the spec's
// input_floats(); the row is that plus kLabelFloats.
struct PositionEvalTask {
  static int row_floats(const InputEncodingSpec& spec);

  static void encode_row(const EncodeContext& ctx, float* out_row);
};

}  // namespace scribblez

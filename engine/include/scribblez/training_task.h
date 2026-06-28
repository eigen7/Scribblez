#pragma once

#include "scribblez/encode_context.h"
#include "scribblez/input_encoder.h"     // kInputFloats
#include "scribblez/training_targets.h"  // kLabelFloats, AllTargets

namespace scribblez {

// A TrainingTask bundles an input encoding and a set of labels into one training
// row. PositionEncoder replays a game to a sampled position, fills an
// EncodeContext, and hands it to the task's encode_row, which writes kInputFloats
// input floats followed by kLabelFloats label floats. Parameterizing the encoder
// on the task lets one replay/streaming pipeline serve several training problems.

// The win-probability task: the post-move board/leave/score input (via the
// stateful GameStateEncoder the context points at) plus the WLD, score-diff, and
// opponent-next-placement labels.
struct PostMoveValueTask {
  static constexpr int kInputFloats = scribblez::kInputFloats;
  static constexpr int kLabelFloats = scribblez::kLabelFloats;
  static constexpr int kRowFloats = kInputFloats + kLabelFloats;

  static void encode_row(const EncodeContext& ctx, float* out_row);
};

}  // namespace scribblez

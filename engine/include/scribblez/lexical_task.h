#pragma once

#include "scribblez/encode_context.h"
#include "scribblez/lane_targets.h"
#include "scribblez/lexical_input_encoder.h"

namespace scribblez {

// The lexical "highest-scoring move per lane" task: the lean board + rack input
// (LexicalInputEncoder) plus the per-lane occupancy / score / mask labels. The
// labels require enumerating legal moves at the position, so this task reads the
// lexicon from the context's `dict`.
struct LexicalTask {
  static constexpr int kInputFloats = LexicalInputEncoder::kInputFloats;
  static constexpr int kLabelFloats = kLaneLabelFloats;
  static constexpr int kRowFloats = kInputFloats + kLabelFloats;

  static void encode_row(const EncodeContext& ctx, float* out_row);
};

}  // namespace scribblez

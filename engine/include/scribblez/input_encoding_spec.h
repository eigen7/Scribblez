#pragma once

namespace scribblez {

class Dictionary;

// Per-run input-encoding configuration: the lexicon the lexical features
// derive from, whether the contingent-draw potential blocks are computed and
// included, and whether the opponent's rack is included ("open-rack" -- an
// experiment-only information condition in which racks are public; see
// docs/sim_residual_feedback.md). Chosen once per process (baked into the FFI
// session) and carried by every encoder; a model's contingent arm is recorded
// in its ONNX metadata_props ("contingent_features") so serving consumers can
// recover it (open-rack models are research instruments and are not exported
// for serving). The row layout a spec selects is owned by input_encoder.h's
// block registry.
struct InputEncodingSpec {
  const Dictionary* dict;
  bool contingent_features;
  bool opp_rack_input = false;
};

}  // namespace scribblez

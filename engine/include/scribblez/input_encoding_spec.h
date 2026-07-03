#pragma once

namespace scribblez {

class Dictionary;

// Per-run input-encoding configuration: the lexicon the lexical features
// derive from, and whether the contingent-draw potential blocks are computed
// and included. Chosen once per process (baked into the FFI session) and
// carried by every encoder; a model's arm is recorded in its ONNX
// metadata_props ("contingent_features") so serving consumers can recover it.
// The row layout a spec selects is owned by input_encoder.h's block registry.
struct InputEncodingSpec {
  const Dictionary* dict;
  bool contingent_features;
};

}  // namespace scribblez

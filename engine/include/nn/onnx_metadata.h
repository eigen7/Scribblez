#pragma once

#include <string>
#include <vector>

// The metadata_props entries our ONNX exporters stamp into every model the
// serving side consumes, and the one place that reads them back. They say what
// a model file cannot say structurally: which optional input blocks it takes,
// which graph family it is, and which move-feature encoding it was trained on.
// An entry the exporter did not write reads as its documented "absent" value,
// so a consumer that knows of an entry the exporter never heard of still reads
// the model correctly.
//
// The writers are py/scribblez/{position_eval,move_set_eval}/onnx_export.py,
// through the shared py/scribblez/onnx_export_util.py.

namespace scribblez {
namespace nn {

// The `graph` entry's values: which model family, and so which runtime, a file
// belongs to.
inline constexpr const char* kGraphPositionEval = "position_eval";
inline constexpr const char* kGraphMoveSetEval = "move_set_eval";

struct OnnxMetadata {
  // The input-encoding arm: which optional blocks the board row carries.
  bool contingent_features = false;
  bool opp_leave_input = false;

  // Keys the engine-plan cache, so every checkpoint of one architecture shares
  // a plan. Required -- a model without it throws.
  std::string architecture_signature;

  // kGraph* above. Empty for an export predating the entry, which only the
  // position runtime accepts (it is the graph that existed at the time).
  std::string graph;

  // move_set_encoder.h's kMoveEncodingVersion at export time, -1 when absent.
  // Only the move-set exporter stamps it.
  int move_encoding_version = -1;
};

// Throws on unparseable bytes, or on a model with no architecture signature.
OnnxMetadata parse_onnx_metadata(const std::vector<char>& onnx_bytes);

}  // namespace nn
}  // namespace scribblez

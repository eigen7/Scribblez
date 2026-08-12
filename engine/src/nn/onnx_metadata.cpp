#include "nn/onnx_metadata.h"

#include <onnx/onnx_pb.h>

#include <stdexcept>

namespace scribblez {
namespace nn {

OnnxMetadata parse_onnx_metadata(const std::vector<char>& onnx_bytes) {
  onnx::ModelProto model;
  if (!model.ParseFromArray(onnx_bytes.data(), static_cast<int>(onnx_bytes.size()))) {
    throw std::runtime_error("Failed to parse ONNX model bytes");
  }
  OnnxMetadata meta;
  for (int i = 0; i < model.metadata_props_size(); ++i) {
    const auto& kv = model.metadata_props(i);
    if (kv.key() == "contingent_features") {
      meta.contingent_features = kv.value() == "true";
    } else if (kv.key() == "opp_leave_input") {
      meta.opp_leave_input = kv.value() == "true";
    } else if (kv.key() == "model-architecture-signature") {
      meta.architecture_signature = kv.value();
    } else if (kv.key() == "graph") {
      meta.graph = kv.value();
    } else if (kv.key() == "move_encoding_version") {
      meta.move_encoding_version = std::stoi(kv.value());
    }
  }
  if (meta.architecture_signature.empty()) {
    throw std::runtime_error("ONNX model missing the model-architecture-signature metadata entry");
  }
  return meta;
}

}  // namespace nn
}  // namespace scribblez

#include "nn/onnx_metadata.h"

#include "util/exception.h"
#include "util/hash.h"
#include "util/io.h"

#include <onnx/onnx_pb.h>

namespace scribblez {
namespace nn {

int OnnxMetadata::int_entry(const std::string& key, int absent_value) const {
  auto it = entries.find(key);
  if (it == entries.end()) return absent_value;
  return std::stoi(it->second);
}

OnnxMetadata parse_onnx_metadata(const std::vector<char>& onnx_bytes) {
  onnx::ModelProto model;
  if (!model.ParseFromArray(onnx_bytes.data(), int(onnx_bytes.size()))) {
    throw util::CleanException("Failed to parse ONNX model bytes");
  }
  OnnxMetadata meta;
  for (int i = 0; i < model.metadata_props_size(); ++i) {
    const auto& kv = model.metadata_props(i);
    if (kv.key() == "opp_leave_input") {
      meta.opp_leave_input = kv.value() == "true";
    } else if (kv.key() == "model-architecture-signature") {
      meta.architecture_signature = kv.value();
    } else if (kv.key() == "graph") {
      meta.graph = kv.value();
    } else {
      meta.entries[kv.key()] = kv.value();
    }
  }
  if (meta.architecture_signature.empty()) {
    throw util::CleanException(
      "ONNX model missing the model-architecture-signature metadata entry");
  }
  return meta;
}

void check_leave_values(const OnnxMetadata& meta, const std::string& onnx_path,
                        const std::string& expected) {
  const auto it = meta.entries.find("leave_values");
  const std::string declared = it == meta.entries.end() ? "(none)" : it->second;
  if (declared != expected) {
    throw util::CleanException(
      "{} was trained against leave values {}, but these are {}; use a model trained against "
      "them",
      onnx_path, declared, expected);
  }
}

ModelIdentity read_model_identity(const std::string& onnx_path, const std::string& format_key) {
  const OnnxMetadata meta = parse_onnx_metadata(util::read_file_bytes(onnx_path));
  const auto it = meta.entries.find("tag");
  return {.hash = util::file_content_hash(onnx_path),
          .tag = it == meta.entries.end() ? std::string() : it->second,
          .generation = meta.int_entry("generation", -1),
          .format_version = meta.int_entry(format_key, 0)};
}

}  // namespace nn
}  // namespace scribblez

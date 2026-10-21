#include "belief/served_leave_model.h"

#include "nn/onnx_metadata.h"
#include "util/hash.h"
#include "util/io.h"

namespace scribblez::belief {

namespace {

LeaveInferenceNet::Params net_params(const std::string& onnx_path) {
  LeaveInferenceNet::Params p;
  p.onnx_path = onnx_path;
  return p;
}

ServedLeaveModel::Identity read_identity(const std::string& onnx_path) {
  const nn::OnnxMetadata meta = nn::parse_onnx_metadata(util::read_file_bytes(onnx_path));
  const auto it = meta.entries.find("tag");
  return {.hash = util::file_content_hash(onnx_path),
          .tag = it == meta.entries.end() ? std::string() : it->second,
          .generation = meta.int_entry("generation", -1),
          .format_version = meta.int_entry("leave_inference_format_version", 0)};
}

}  // namespace

ServedLeaveModel::ServedLeaveModel(const std::string& onnx_path,
                                   const std::string& leave_values_hash)
    : net_(net_params(onnx_path), leave_values_hash), identity_(read_identity(onnx_path)) {}

LeaveParamsFn ServedLeaveModel::predict_fn() {
  return [this](const float* rows, int count, LeaveParams* out) {
    const std::lock_guard lock(mutex_);
    net_.predict(rows, count, out);
  };
}

}  // namespace scribblez::belief

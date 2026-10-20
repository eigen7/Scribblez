#include "belief/leave_inference_net.h"

#include "encoding/leave_inference_input_encoder.h"
#include "nn/onnx_metadata.h"
#include "util/exception.h"
#include "util/io.h"

#include <algorithm>
#include <cstring>

namespace scribblez::belief {

namespace {

using Encoder = LeaveInferenceInputEncoder;

nn::NeuralNetParams<nn::LeaveInferenceSpec> net_params(const LeaveInferenceNet::Params& p) {
  nn::NeuralNetParams<nn::LeaveInferenceSpec> np;
  np.onnx_path = p.onnx_path;
  np.cuda_device_id = p.cuda_device_id;
  np.max_rows = p.max_rows;
  np.precision = p.precision;
  np.fast_build = p.fast_build;
  np.mount_root = p.mount_root;
  return np;
}

void check_leave_values(const std::string& onnx_path, const std::string& expected) {
  const nn::OnnxMetadata meta = nn::parse_onnx_metadata(util::read_file_bytes(onnx_path));
  const auto it = meta.entries.find("leave_values");
  const std::string declared = it == meta.entries.end() ? "(none)" : it->second;
  if (declared != expected) {
    throw util::CleanException(
      "{} was trained against leave values {}, but these are {}; use a model trained against "
      "them",
      onnx_path, declared, expected);
  }
}

}  // namespace

LeaveInferenceNet::LeaveInferenceNet(const Params& params, const std::string& leave_values_hash)
    : net_(net_params(params)) {
  check_leave_values(params.onnx_path, leave_values_hash);
  net_.load();
  // The format version gate makes these hold for any model it admits; a
  // mismatch would otherwise misread every row.
  if (net_.spatial_planes() != Encoder::kSpatialPlanes ||
      net_.scalar_floats() != Encoder::kScalarFloats) {
    throw util::CleanException(
      "{} takes {} planes and {} scalars, but the encoder writes {} and {}", params.onnx_path,
      net_.spatial_planes(), net_.scalar_floats(), Encoder::kSpatialPlanes, Encoder::kScalarFloats);
  }
}

void LeaveInferenceNet::predict(const float* rows, int count, LeaveParams* out) {
  for (int start = 0; start < count; start += max_rows()) {
    const int chunk = std::min(max_rows(), count - start);
    for (int r = 0; r < chunk; ++r) {
      nn::stage_board_row(net_, rows + size_t(start + r) * Encoder::kInputFloats, r);
    }
    net_.predict(chunk);
    std::memcpy(out + start, net_.host<nn::LeaveParamsOutput>(), sizeof(LeaveParams) * chunk);
  }
}

}  // namespace scribblez::belief

#include "belief/served_leave_model.h"

#include "nn/onnx_metadata.h"

namespace scribblez::belief {

namespace {

LeaveInferenceNet::Params net_params(const std::string& onnx_path) {
  LeaveInferenceNet::Params p;
  p.onnx_path = onnx_path;
  return p;
}

}  // namespace

ServedLeaveModel::ServedLeaveModel(const std::string& onnx_path,
                                   const std::string& leave_values_hash)
    : net_(net_params(onnx_path), leave_values_hash),
      identity_(nn::read_model_identity(onnx_path, "leave_inference_format_version")) {}

LeaveParamsFn ServedLeaveModel::predict_fn() {
  return [this](const float* rows, int count, LeaveParams* out) {
    const std::lock_guard lock(mutex_);
    net_.predict(rows, count, out);
  };
}

}  // namespace scribblez::belief

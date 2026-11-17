#include "belief/served_leave_model.h"

#include "encoding/leave_inference_input_encoder.h"
#include "nn/onnx_metadata.h"
#include "nn/shared_registry.h"

#include <algorithm>
#include <utility>

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
      identity_(nn::read_model_identity(onnx_path, "leave_inference_format_version")),
      batcher_([this](Pack pack) { serve(pack); }) {}

LeaveParamsFn ServedLeaveModel::predict_fn() {
  return [this](const float* rows, int count, LeaveParams* out) {
    Request req{rows, count, out};
    batcher_.submit(req);
    if (req.error) std::rethrow_exception(req.error);
  };
}

void ServedLeaveModel::serve(Pack pack) {
  try {
    // A lone request, the common case under light load, needs no copies.
    if (pack.size() == 1) {
      net_.predict(pack.front()->rows, pack.front()->count, pack.front()->out);
      return;
    }
    constexpr size_t kRowFloats = LeaveInferenceInputEncoder::kInputFloats;
    rows_.clear();
    for (const Request* r : pack)
      rows_.insert(rows_.end(), r->rows, r->rows + size_t(r->count) * kRowFloats);
    out_.resize(rows_.size() / kRowFloats);
    net_.predict(rows_.data(), int(out_.size()), out_.data());
    size_t first = 0;
    for (const Request* r : pack) {
      std::copy_n(&out_[first], r->count, r->out);
      first += size_t(r->count);
    }
  } catch (...) {
    const std::exception_ptr error = std::current_exception();
    for (Request* r : pack) r->error = error;
  }
}

std::shared_ptr<ServedLeaveTable> ServedLeaveTable::shared(const std::string& onnx_path,
                                                           const LeaveValues& values) {
  // The leave values' hash is part of the key: the net refuses a model trained
  // against other values (LeaveInferenceNet).
  using Key = std::pair<std::string, std::string>;
  static nn::SharedRegistry<Key, ServedLeaveTable> registry;
  return registry.get_or_create(Key{onnx_path, values.content_hash()}, [&] {
    return std::make_shared<ServedLeaveTable>(onnx_path, values);
  });
}

AgentLeaveModel agent_leave_model(const std::string& onnx_path, const LeaveValues& values) {
  const std::shared_ptr<ServedLeaveTable> served = ServedLeaveTable::shared(onnx_path, values);
  return {
    .predict = [served, predict = served->served.predict_fn()](
                 const float* rows, int count, LeaveParams* out) { predict(rows, count, out); },
    .table = std::shared_ptr<const LeaveValueArrays>(served, &served->table),
    .identity = served->served.identity()};
}

}  // namespace scribblez::belief

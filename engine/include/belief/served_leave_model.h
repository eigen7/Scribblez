#pragma once

// A leave inference model loaded to serve beliefs from several threads: its
// TensorRT engine (LeaveInferenceNet) behind a mutex, and the identity a
// consumer records as provenance.

#include "belief/leave_inference_net.h"
#include "belief/slog_leave_beliefs.h"

#include <mutex>
#include <string>

namespace scribblez::belief {

class ServedLeaveModel {
 public:
  // The model's provenance, from its file and its ONNX metadata
  // (leave_inference/onnx_export.py).
  struct Identity {
    std::string hash;  // util::file_content_hash of the ONNX
    std::string tag;
    int generation = -1;
    int format_version = 0;
  };

  // Loads `onnx_path`, refusing a model trained against other leave values
  // than `leave_values_hash` (LeaveInferenceNet).
  ServedLeaveModel(const std::string& onnx_path, const std::string& leave_values_hash);

  ServedLeaveModel(const ServedLeaveModel&) = delete;
  ServedLeaveModel& operator=(const ServedLeaveModel&) = delete;

  const Identity& identity() const { return identity_; }

  // A thread-safe LeaveParamsFn over the engine, valid while this object
  // lives.
  LeaveParamsFn predict_fn();

 private:
  LeaveInferenceNet net_;
  std::mutex mutex_;
  Identity identity_;
};

}  // namespace scribblez::belief

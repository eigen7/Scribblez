#pragma once

// The leave inference model's TensorRT engine (docs/plans/deep_bot_v0.md,
// "Serving"): rows of encoding/leave_inference_input_encoder.h in, each
// position's belief::LeaveParams out, for belief::LeaveSampler. It drives
// NeuralNet<LeaveInferenceSpec> directly, as agent/move_proposal_nets.h does.
//
// The model's tilt reads leave values: a and b weigh each leave by its value
// in the leave values it was trained against. Loading therefore refuses a
// model trained against other leave values than the caller's.
//
// Threading: as NeuralNet, one call at a time.

#include "belief/leave_params.h"
#include "nn/model_specs.h"
#include "nn/neural_net.h"
#include "nn/trt_util.h"

#include <string>

namespace scribblez::belief {

class LeaveInferenceNet {
 public:
  struct Params {
    std::string onnx_path;
    int cuda_device_id = 0;
    // Rows per engine call; predict() chunks larger batches.
    int max_rows = nn::LeaveInferenceSpec::kDefaultMaxRows;
    // FP32 (TF32 inside TensorRT's convolutions): the 29 floats feed exp()
    // over every leave of a length, so a coarsely rounded a or b would shift
    // every weight. Reduced precision is unmeasured.
    nn::Precision precision = nn::Precision::kFP32;
    bool fast_build = false;
    std::string mount_root = "/workspace/mount";
  };

  // Loads params.onnx_path. Throws util::CleanException unless the model's
  // `leave_values` entry is `leave_values_hash` (LeaveValues::content_hash)
  // and its input widths are this engine's encoder's.
  LeaveInferenceNet(const Params& params, const std::string& leave_values_hash);

  int max_rows() const { return net_.max_rows(); }

  // `count` rows of LeaveInferenceInputEncoder::kInputFloats each, as
  // encode() writes them, to out[0..count).
  void predict(const float* rows, int count, LeaveParams* out);

 private:
  nn::NeuralNet<nn::LeaveInferenceSpec> net_;
};

}  // namespace scribblez::belief

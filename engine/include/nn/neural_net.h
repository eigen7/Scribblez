#pragma once

#include "nn/trt_util.h"

#include <cstdint>
#include <memory>
#include <string>

namespace boost::program_options {
class options_description;
}

// A thin, synchronous wrapper around a TensorRT engine specialized to the
// Scribblez position evaluation model: two inputs ("input_spatial",
// "input_scalar", at widths the model itself declares) and six outputs ("wld",
// "score_diff" = [mean, std], and four (N,15,15) auxiliary placement masks).
// input_encoder.h is the single source of truth for the encoding layout.
//
// One NeuralNet drives one engine from one thread: predict() blocks until the
// outputs are back, with no cross-thread batching and no async pipeline.

namespace scribblez {
namespace nn {

struct NeuralNetParams {
  std::string onnx_path;  // exported ONNX model load() builds from
  int cuda_device_id = 0;
  int max_batch_size = 256;
  Precision precision = Precision::kFP16;
  uint64_t workspace_bytes = uint64_t{1} << 30;  // 1 GiB TensorRT scratch
  std::string mount_root = "/workspace/mount";   // root of the engine-plan cache

  // Build at TensorRT optimization level 0: take the first working kernel per
  // layer instead of timing tactics. Cuts a cold build from tens of seconds to
  // a few, at the cost of much slower inference -- for tests and quick checks,
  // not production agents. Cached separately from full-optimization plans.
  bool fast_build = false;

  // Register the command-line-facing subset, bound to this struct's fields.
  // Call before parsing argv.
  void add_options(boost::program_options::options_description& desc);
};

class NeuralNet {
 public:
  explicit NeuralNet(const NeuralNetParams& params);
  ~NeuralNet();

  NeuralNet(const NeuralNet&) = delete;
  NeuralNet& operator=(const NeuralNet&) = delete;

  // Build the engine from params.onnx_path, or deserialize a cached plan and
  // refit it with this model's weights -- every checkpoint of one architecture
  // shares a plan, keyed by architecture signature, precision, batch size, GPU
  // compute capability, and TRT version. Exactly once, before predict().
  void load();

  int max_batch_size() const;

  // Valid after load().
  int spatial_planes() const;
  int scalar_floats() const;

  // The model's input-encoding arm, from the entry the exporter stamps into the
  // ONNX metadata_props (see onnx_export.py). Valid after load(); consumers
  // cross-check it against the input widths through input_encoder.h's registry.
  bool contingent_features() const;
  bool opp_leave_input() const;

  // Row-major staging buffers sized for max_batch_size rows. The caller writes
  // the first num_rows rows before calling predict(num_rows).
  float* input_spatial_host();  // num_rows x (spatial_planes() * 225)
  float* input_scalar_host();   // num_rows x scalar_floats()

  // Blocks until the outputs are back. Requires 1 <= num_rows <= max.
  void predict(int num_rows);

  // Valid after predict(), for its first num_rows rows. wld is raw logits. The
  // auxiliary mask outputs are not exposed: they are never copied back.
  const float* wld_host() const;         // num_rows x kWldFloats
  const float* score_diff_host() const;  // num_rows x [mean, std], std positive

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nn
}  // namespace scribblez

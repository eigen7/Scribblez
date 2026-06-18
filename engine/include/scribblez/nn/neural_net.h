#pragma once

#include "scribblez/nn/trt_util.h"

#include <cstdint>
#include <memory>
#include <string>

// A thin, synchronous wrapper around a TensorRT engine specialized to the
// Scribblez post-move value model: two inputs ("input_spatial" (N,33,15,15),
// "input_scalar" (N,936)) and three outputs ("wld" (N,3), "score_diff" (N,801),
// "opp_next_placement" (N,15,15)).
//
// load() builds a TensorRT engine from an ONNX file (or deserializes a cached
// plan keyed by the model's content hash + precision + batch size + GPU
// compute capability + TRT version), then allocates one execution context, one
// CUDA stream, and pinned host + device buffers sized to max_batch_size.
//
// predict() is blocking: it copies the host input rows to the GPU, runs the
// engine, copies the outputs back, and synchronizes the stream before
// returning. There is no batching across threads and no async pipeline -- one
// NeuralNet drives one engine from one thread.

namespace scribblez {
namespace nn {

struct NeuralNetParams {
  int cuda_device_id = 0;
  int max_batch_size = 256;
  Precision precision = Precision::kFP16;
  uint64_t workspace_bytes = uint64_t{1} << 30;  // 1 GiB TensorRT scratch
  std::string mount_root = "/workspace/mount";   // root of the engine-plan cache
};

class NeuralNet {
 public:
  explicit NeuralNet(const NeuralNetParams& params);
  ~NeuralNet();

  NeuralNet(const NeuralNet&) = delete;
  NeuralNet& operator=(const NeuralNet&) = delete;

  // Build (or load a cached) engine from the ONNX file at `onnx_path` and ready
  // the GPU resources. Must be called exactly once before predict().
  void load(const std::string& onnx_path);

  int max_batch_size() const;

  // Host input staging buffers, row-major and sized for max_batch_size rows.
  // The caller writes the first num_rows rows before calling predict(num_rows).
  float* input_spatial_host();  // num_rows x kSpatialFloats
  float* input_scalar_host();   // num_rows x kScalarFloats

  // Run inference on the first num_rows rows of the host input buffers. Blocks
  // until the outputs have been copied back. Requires 1 <= num_rows <= max.
  void predict(int num_rows);

  // Host output buffers, valid after predict() returns. Row-major; only the
  // first num_rows rows are meaningful. wld / score_diff are raw logits.
  const float* wld_host() const;                 // num_rows x kWldFloats
  const float* score_diff_host() const;          // num_rows x kScoreDiffFloats
  const float* opp_next_placement_host() const;  // num_rows x kOppNextPlacementFloats

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nn
}  // namespace scribblez

#pragma once

#include "scribblez/nn/trt_util.h"

#include <cstdint>
#include <memory>
#include <string>

// A thin, synchronous wrapper around a TensorRT engine specialized to the
// Scribblez post-move value model: two inputs ("input_spatial"
// (N,spatial_planes,15,15), "input_scalar" (N,scalar_floats) -- widths declared
// by the model itself) and three
// outputs ("wld" (N,3), "score_diff" (N,2 = [mean, std]), "opp_next_placement"
// (N,15,15)). The input shapes mirror the constants in input_encoder.h, which
// is the single source of truth for the encoding layout.
//
// load() builds a TensorRT engine from the ONNX file at params.onnx_path (or
// deserializes a cached plan keyed by the model's content hash + precision +
// batch size + GPU compute capability + TRT version), then allocates one
// execution context, one CUDA stream, and pinned host + device buffers sized to
// max_batch_size.
//
// predict() is blocking: it copies the host input rows to the GPU, runs the
// engine, copies the wld and score_diff outputs back, and synchronizes the
// stream before returning. The third engine output, opp_next_placement, is
// produced on the GPU (its device buffer must stay bound) but has no inference
// consumer, so it is not copied back. There is no batching across threads and
// no async pipeline -- one NeuralNet drives one engine from one thread.

namespace scribblez {
namespace nn {

struct NeuralNetParams {
  std::string onnx_path;  // exported ONNX model load() builds from
  int cuda_device_id = 0;
  int max_batch_size = 256;
  Precision precision = Precision::kFP16;
  uint64_t workspace_bytes = uint64_t{1} << 30;  // 1 GiB TensorRT scratch
  std::string mount_root = "/workspace/mount";   // root of the engine-plan cache

  // Build the engine at TensorRT builder optimization level 0: skip the tactic-
  // timing search and take the first working kernel for each layer. This cuts a
  // cold engine build by several-fold (tens of seconds down to a few seconds for
  // this model) at the cost of much slower inference, so it is meant for tests
  // and quick checks, not for production agents. Fast-built plans are cached
  // separately from full-optimization plans (see engine_plan_cache_path).
  bool fast_build = false;
};

class NeuralNet {
 public:
  explicit NeuralNet(const NeuralNetParams& params);
  ~NeuralNet();

  NeuralNet(const NeuralNet&) = delete;
  NeuralNet& operator=(const NeuralNet&) = delete;

  // Build (or load a cached) engine from the ONNX file at params.onnx_path and
  // ready the GPU resources. Must be called exactly once before predict().
  void load();

  int max_batch_size() const;

  // The loaded model's declared input widths (read off the engine's tensor
  // shapes): the spatial plane count and the scalar float count of one row.
  // Valid after load(). Together they say which input layout (full or base;
  // see input_encoder.h's registry) the model consumes.
  int spatial_planes() const;
  int scalar_floats() const;

  // Host input staging buffers, row-major and sized for max_batch_size rows.
  // The caller writes the first num_rows rows before calling predict(num_rows).
  float* input_spatial_host();  // num_rows x (spatial_planes() * 225)
  float* input_scalar_host();   // num_rows x scalar_floats()

  // Run inference on the first num_rows rows of the host input buffers. Blocks
  // until the outputs have been copied back. Requires 1 <= num_rows <= max.
  void predict(int num_rows);

  // Host output buffers, valid after predict() returns. Row-major; only the
  // first num_rows rows are meaningful. wld is raw logits; score_diff is the
  // [mean, std] of the final-differential Gaussian (std already positive). The
  // opp_next_placement output is not exposed here -- it is not copied back.
  const float* wld_host() const;         // num_rows x kWldFloats
  const float* score_diff_host() const;  // num_rows x kScoreDiffOutputFloats ([mean, std])

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nn
}  // namespace scribblez

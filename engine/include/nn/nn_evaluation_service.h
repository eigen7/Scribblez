#pragma once

#include "nn/eval_service.h"
#include "nn/neural_net.h"

#include <vector>

// Synchronous front-end over a NeuralNet that turns raw encoder rows into
// win-probability evaluations.
//
// Callers hand it a contiguous block of `count` rows, each laid out exactly as
// GameStateEncoder::encode_input() writes them (the model's spatial floats
// followed by its scalar floats; widths per the loaded model). The service
// de-interleaves each row into the engine's separate spatial / scalar input
// buffers, runs the model (splitting into max_batch_size chunks as needed),
// applies a softmax to the WLD logits, and returns one Eval per row.
//
// This is the simplest possible design: every call blocks until the GPU
// finishes, there is no cross-thread batching, no caching, and no yield
// integration. It exists to answer one question -- does a trained model beat
// HastyBot -- not to maximize throughput.

namespace scribblez {
namespace nn {

// The Eval struct and the EvalService interface live in eval_service.h.

class NNEvaluationService : public EvalService {
 public:
  explicit NNEvaluationService(const NeuralNetParams& params);

  // Build/load the engine from the params' onnx_path. Call once before
  // evaluate().
  void load();

  // The loaded model's declared arm and input widths. Valid after load().
  bool contingent_features() const override { return net_.contingent_features(); }
  int spatial_planes() const override { return net_.spatial_planes(); }
  int scalar_floats() const override { return net_.scalar_floats(); }

  // Evaluate `count` rows (`inputs` is count * the model's row float count)
  // and write `count` results into `out`. Blocks until inference completes.
  void evaluate(const float* inputs, int count, Eval* out) override;

  // Convenience overload returning a freshly allocated vector.
  std::vector<Eval> evaluate(const float* inputs, int count);

 private:
  // Run one chunk of at most max_batch_size rows starting at `inputs`,
  // writing `chunk` results into `out`.
  void evaluate_chunk(const float* inputs, int chunk, Eval* out);

  NeuralNet net_;
};

}  // namespace nn
}  // namespace scribblez

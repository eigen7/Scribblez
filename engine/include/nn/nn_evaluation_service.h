#pragma once

#include "nn/eval_service.h"
#include "nn/neural_net.h"

#include <vector>

// Synchronous front-end over a NeuralNet: de-interleave each encoder row into
// the engine's separate spatial / scalar input buffers, run the model in
// max_batch_size chunks, and softmax the WLD logits into one Eval per row.
//
// The simplest possible design -- every call blocks on the GPU, with no
// cross-thread batching, no caching, and no yield integration. It exists to
// answer one question, does a trained model beat HastyBot, not to maximize
// throughput.

namespace scribblez {
namespace nn {

class NNEvaluationService : public EvalService {
 public:
  explicit NNEvaluationService(const NeuralNetParams& params);

  // Call once before evaluate().
  void load();

  // Valid after load().
  bool contingent_features() const override { return net_.contingent_features(); }
  bool opp_leave_input() const override { return net_.opp_leave_input(); }
  int spatial_planes() const override { return net_.spatial_planes(); }
  int scalar_floats() const override { return net_.scalar_floats(); }

  // Blocks until inference completes.
  void evaluate(const float* inputs, int count, Eval* out) override;

  std::vector<Eval> evaluate(const float* inputs, int count);

 private:
  // Run one chunk of at most max_batch_size rows.
  void evaluate_chunk(const float* inputs, int chunk, Eval* out);

  NeuralNet net_;
};

}  // namespace nn
}  // namespace scribblez

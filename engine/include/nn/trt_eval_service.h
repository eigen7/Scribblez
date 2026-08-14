#pragma once

#include "nn/eval_service.h"
#include "nn/neural_net.h"

#include <memory>
#include <vector>

// The TensorRT-backed EvalService<Spec>: the production implementation of the
// interface, kept out of eval_service.h so an agent or a stub-driven test
// depends on the interface alone. One class template serves both model
// families; what differs -- how a Batch's rows reach the engine's staging
// buffers -- is dispatched on the spec's Batch type in the .cpp.
//
// The simplest possible design -- every call blocks on the GPU, with no
// cross-thread batching, no caching, and no yield integration. It exists to
// answer one question, does a trained model beat HastyBot, not to maximize
// throughput.

namespace scribblez {
namespace nn {

template <typename Spec>
class TrtEvalService : public EvalService<Spec> {
 public:
  using SpecBatch = Spec::Batch;
  using Output = Spec::Output;
  using Outputs = Spec::Outputs;
  using AuxOutputs = Spec::AuxOutputs;

  explicit TrtEvalService(const NeuralNetParams<Spec>& params) : net_(params) {}

  // Call once before evaluate().
  void load() { net_.load(); }

  // Valid after load().
  bool contingent_features() const override { return net_.contingent_features(); }
  bool opp_leave_input() const override { return net_.opp_leave_input(); }
  int spatial_planes() const override { return net_.spatial_planes(); }
  int scalar_floats() const override { return net_.scalar_floats(); }

  // Blocks until inference completes. A batch larger than the engine's
  // max_rows is split into chunks, which changes no result: rows are scored
  // independently given their staged context.
  void evaluate(const SpecBatch& batch, Output* out) override;

  std::vector<Output> evaluate(const SpecBatch& batch);

  // Additionally receives each row's aux outputs, batch_rows x AuxOutputs
  // floats in head order, each head decoded per its declared AuxDecode.
  // Requires the service was constructed with params.copy_aux; aux_out may be
  // null.
  void evaluate(const SpecBatch& batch, Output* out, float* aux_out)
    requires(AuxOutputs::size > 0);

 private:
  // The shared driver: per-call staging, then chunked stage/predict/decode;
  // aux_out may be null.
  void evaluate_batch(const SpecBatch& batch, Output* out, float* aux_out);

  NeuralNet<Spec> net_;
};

// Construct and load() the service for `params` -- the one call an agent
// factory needs to stand up production inference.
template <typename Spec>
std::unique_ptr<EvalService<Spec>> make_loaded_service(const NeuralNetParams<Spec>& params);

extern template class TrtEvalService<PositionEvaluationSpec>;
extern template class TrtEvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez

#pragma once

#include "nn/eval_service.h"
#include "nn/neural_net.h"

#include <memory>
#include <span>
#include <string>

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
  using Outputs = Spec::Outputs;
  using AuxOutputs = Spec::AuxOutputs;

  explicit TrtEvalService(const NeuralNetParams<Spec>& params) : net_(params) {}

  // Call once before evaluate().
  void load() { net_.load(); }

  // Valid after load().
  bool opp_leave_input() const override { return net_.opp_leave_input(); }
  int spatial_planes() const override { return net_.spatial_planes(); }
  int scalar_floats() const override { return net_.scalar_floats(); }

  // The base's serialized entry point; unhidden here because the aux
  // overload below would otherwise shadow it for concrete-typed callers.
  using EvalService<Spec>::evaluate;

  // Additionally receives each row's aux outputs, batch_rows x AuxOutputs
  // floats in head order, each head decoded per its declared RowDecode.
  // Requires the service was constructed with params.copy_aux; aux_out may be
  // null. Serialized under the same base-class mutex as evaluate().
  void evaluate(const SpecBatch& batch, std::span<float* const> head_out, float* aux_out)
    requires(AuxOutputs::size > 0);

 protected:
  // Blocks until inference completes. A batch larger than the engine's
  // max_rows is split into chunks, which changes no result: rows are scored
  // independently given their staged context.
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override;

 private:
  // The shared driver: per-call staging, then chunked stage/predict/decode;
  // aux_out may be null.
  void evaluate_batch(const SpecBatch& batch, std::span<float* const> head_out, float* aux_out);

  NeuralNet<Spec> net_;
};

// Construct and load() the service for `params` -- the one call an agent
// factory needs to stand up production inference.
template <typename Spec>
std::unique_ptr<EvalService<Spec>> make_loaded_service(const NeuralNetParams<Spec>& params);

// The position-evaluation leaf service for value truncation, from an .onnx
// path, or null for an empty path. The one place the sim agents and the
// offline generators stand up their leaf model, so they cannot drift on how
// it is served (BF16, whose FP32-range exponent serves the trunk's activation
// magnitudes without FP16 overflow). Shared through PositionEvalService::
// create(): the agents of every game thread of a run resolve the same leaf
// path to one loaded engine, instead of one apiece.
std::shared_ptr<PositionEvalService> load_leaf_position_service(const std::string& onnx_path,
                                                                int cuda_device_id = 0);

extern template class TrtEvalService<PositionEvaluationSpec>;
extern template class TrtEvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez

#pragma once

#include "nn/eval_service.h"
#include "nn/neural_net.h"

#include <memory>
#include <span>
#include <string>

// The TensorRT-backed EvalService<Spec>, the production implementation of the
// interface. It lives apart from eval_service.h so that agents and stub-driven
// tests depend on the interface alone, without CUDA or TensorRT.
//
// One class template serves both the position and the move-set families. The
// only family-specific part, how a Batch's rows reach the engine's staging
// buffers, is overloaded on the spec's Batch type in the .cpp.
//
// Every call is synchronous: it stages, runs, and decodes on the calling thread
// and blocks until the GPU is done. Coalescing concurrent callers into larger
// batches is BatchingPositionEvalService's job, layered on top.

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

  // Unhides the base's evaluate(), which the aux overload below would otherwise
  // shadow for callers holding a TrtEvalService directly.
  using EvalService<Spec>::evaluate;

  // evaluate() plus the spec's aux outputs: aux_out receives, per row, every
  // aux head's decoded floats in list order. Requires params.copy_aux at
  // construction; aux_out may be null. Shares evaluate()'s serialization.
  void evaluate(const SpecBatch& batch, std::span<float* const> head_out, float* aux_out)
    requires(AuxOutputs::size > 0);

 protected:
  // A batch larger than the engine's max_rows is run in chunks. That changes no
  // result, since each row is scored independently of the others.
  void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) override;

 private:
  // The driver behind both entry points; aux_out may be null.
  void evaluate_batch(const SpecBatch& batch, std::span<float* const> head_out, float* aux_out);

  NeuralNet<Spec> net_;
};

// A new, unshared service for `params`, already loaded.
template <typename Spec>
std::unique_ptr<EvalService<Spec>> make_loaded_service(const NeuralNetParams<Spec>& params);

// The position-evaluation service that scores the leaves of value-truncated
// rollouts, or null for an empty path. The sim agents and the offline
// generators all stand up their leaf model through here, so they cannot drift
// on how it is served. The service comes from PositionEvalService::create(), so
// every game thread of a run that names the same leaf model shares one engine.
std::shared_ptr<PositionEvalService> load_leaf_position_service(const std::string& onnx_path,
                                                                int cuda_device_id = 0);

extern template class TrtEvalService<PositionEvaluationSpec>;
extern template class TrtEvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez

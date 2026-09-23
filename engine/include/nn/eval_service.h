#pragma once

#include "nn/model_specs.h"

#include <memory>
#include <mutex>
#include <span>

namespace scribblez {
namespace nn {

// Forward-declared so this header stays free of TensorRT (nn/neural_net.h).
template <typename Spec>
struct NeuralNetParams;

// What a served model declares about the board rows it consumes: its
// input-encoding arm and its input widths. Agents build their InputEncodingSpec
// from these (derive_input_spec() in agent/candidate_evaluator.h), whichever
// model family they serve, so every evaluation service exposes this interface.
class ServedModelInputs {
 public:
  virtual ~ServedModelInputs() = default;

  virtual bool opp_leave_input() const = 0;
  virtual int spatial_planes() const = 0;
  virtual int scalar_floats() const = 0;
};

// The abstract evaluator for one model family, over the Batch its spec
// declares (model_specs.h): a set of positions for the position model, one
// position's candidate moves for the move-set model.
//
// Carries no CUDA or TensorRT dependency, so agents and their unit tests
// depend on this template and inject either TrtEvalService<Spec> or a
// scripted stub.
//
// Thread-safe: evaluate() serializes concurrent callers under a base-class
// mutex, so one loaded service can be shared by, say, SimRunner's rollout
// workers or the per-position runners of a parallel generator. Implementations
// override do_evaluate() and need no locking of their own. That suffices for
// TensorRT, whose contract is one call at a time from any thread
// (neural_net.h).
//
// evaluate() is virtual so a decorator can replace this one-caller-at-a-time
// policy with its own synchronization, as BatchingPositionEvalService does to
// coalesce concurrent callers into larger GPU batches.
template <typename Spec>
class EvalService : public ServedModelInputs {
 public:
  using SpecBatch = Spec::Batch;
  using Outputs = Spec::Outputs;

  // A loaded service for `params`, shared: while an instance for equal params
  // is alive, every call returns it. The game threads of a run thus share one
  // engine and one execution context, whose activation memory would otherwise
  // be paid once per thread. Only the position family defines this (below).
  static std::shared_ptr<EvalService> create(const NeuralNetParams<Spec>& params);

  // Score `batch`. head_out holds one destination per Outputs entry, in list
  // order; head_out[i] receives rows x kRowElems floats, decoded per the
  // head's RowDecode.
  virtual void evaluate(const SpecBatch& batch, std::span<float* const> head_out) {
    std::lock_guard<std::mutex> lock(mutex_);
    do_evaluate(batch, head_out);
  }

 protected:
  virtual void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) = 0;

  // For an implementation's extra entry points (such as TrtEvalService's aux
  // overload), which must serialize with evaluate().
  std::mutex& eval_mutex() { return mutex_; }

 private:
  std::mutex mutex_;
};

using PositionEvalService = EvalService<PositionEvaluationSpec>;
using MoveSetEvalService = EvalService<MoveSetEvaluationSpec>;

// Defined in trt_eval_service.cpp, where it also wraps the shared engine in
// BatchingPositionEvalService. Declared here so callers see the explicit
// specialization rather than implicitly instantiating the primary template.
template <>
std::shared_ptr<PositionEvalService> PositionEvalService::create(
  const NeuralNetParams<PositionEvaluationSpec>& params);

}  // namespace nn
}  // namespace scribblez

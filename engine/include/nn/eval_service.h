#pragma once

#include "nn/model_specs.h"

#include <mutex>
#include <span>

namespace scribblez {
namespace nn {

// What a served model says about the board rows it consumes: its
// input-encoding arm, and the input widths that arm implies. Agents build their
// InputEncodingSpec from the arm and validate the widths against it through the
// layout registry -- the same work whichever model is being served, so every
// evaluation service exposes it through this one interface.
class ServedModelInputs {
 public:
  virtual ~ServedModelInputs() = default;

  virtual bool contingent_features() const = 0;
  virtual bool opp_leave_input() const = 0;
  virtual int spatial_planes() const = 0;
  virtual int scalar_floats() const = 0;
};

// The abstract evaluator for one model family, over the Batch shape that
// family's spec declares (model_specs.h): rows of positions for the position
// model, one position's candidate set for the move set model.
//
// Carries no CUDA/TensorRT dependency: agents and their unit tests depend on
// this template and inject either TrtEvalService<Spec> or a scripted stub.
//
// evaluate() serializes concurrent callers under a base-class mutex, so one
// loaded service is freely shareable -- SimRunner's rollout workers, or many
// single-threaded runners in a position-parallel generator, all call the
// same instance. Implementations override do_evaluate() and need no locking
// of their own; the serialization is sound for the TensorRT service because
// the underlying contract is one call at a time, not thread affinity
// (neural_net.h).
template <typename Spec>
class EvalService : public ServedModelInputs {
 public:
  using SpecBatch = Spec::Batch;
  using Outputs = Spec::Outputs;

  // One destination per Outputs entry, in list order: head_out[i] receives
  // batch-rows x that head's kRowElems floats, decoded per the head's
  // RowDecode.
  void evaluate(const SpecBatch& batch, std::span<float* const> head_out) {
    std::lock_guard<std::mutex> lock(mutex_);
    do_evaluate(batch, head_out);
  }

 protected:
  virtual void do_evaluate(const SpecBatch& batch, std::span<float* const> head_out) = 0;

  // For an implementation's own extra entry points (e.g. the TensorRT
  // service's aux-output overload), which must share the same serialization.
  std::mutex& eval_mutex() { return mutex_; }

 private:
  std::mutex mutex_;
};

using PositionEvalService = EvalService<PositionEvaluationSpec>;
using MoveSetEvalService = EvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez

#pragma once

#include "nn/model_specs.h"

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
template <typename Spec>
class EvalService : public ServedModelInputs {
 public:
  using SpecBatch = Spec::Batch;
  using Outputs = Spec::Outputs;

  // One destination per Outputs entry, in list order: head_out[i] receives
  // batch-rows x that head's kRowElems floats, decoded per the head's
  // RowDecode.
  virtual void evaluate(const SpecBatch& batch, std::span<float* const> head_out) = 0;
};

using PositionEvalService = EvalService<PositionEvaluationSpec>;
using MoveSetEvalService = EvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez

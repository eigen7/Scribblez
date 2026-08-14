#pragma once

#include "nn/model_specs.h"

namespace scribblez {
namespace nn {

struct Eval {
  // P(win) + 0.5 * P(draw), the expected game points under the WLD head.
  float win_prob = 0.0f;
  float p_win = 0.0f;
  float p_draw = 0.0f;
  float p_loss = 0.0f;

  // The ScoreDiff head's Gaussian over the final differential (mover's score
  // minus opponent's). The mean is the points-oriented selection objective,
  // closest in spirit to HastyBot's static equity.
  float score_diff_mean = 0.0f;
  float score_diff_std = 0.0f;
};

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
// model, one position's candidate set for the move set model. The Eval is
// deliberately the same struct whichever family produced it, so an agent's
// EvalObjective ranks alternatives identically -- the families differ in how a
// value is obtained, not in what it means.
//
// Carries no CUDA/TensorRT dependency: agents and their unit tests depend on
// this template and inject either TrtEvalService<Spec> or a scripted stub.
template <typename Spec>
class EvalService : public ServedModelInputs {
 public:
  // Writes one Eval per batch row to `out`.
  virtual void evaluate(const typename Spec::Batch& batch, Eval* out) = 0;
};

using PositionEvalService = EvalService<PositionEvaluationSpec>;
using MoveSetEvalService = EvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez

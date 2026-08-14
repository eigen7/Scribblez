#pragma once

#include "nn/model_specs.h"

#include <span>

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

// Assemble `n` consumer Evals from the decoded rows of the two scoring heads:
// WLD probabilities (the serving side has already applied WldOutput's
// RowDecode) and the score-diff pair. win_prob folds a draw in as half a win.
// Consumer-side vocabulary: the serving layer transports decoded head rows
// and knows nothing of this struct. Implemented in eval_decode.cpp.
void make_evals(const float* wld_probs, const float* score_diff, int n, Eval* out);

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

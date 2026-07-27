#pragma once

namespace scribblez {
namespace nn {

// One model readout for a single encoded position.
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

// Abstract evaluator over encoder rows, each laid out as
// GameStateEncoder::encode_input() writes them. It carries no CUDA/TensorRT
// dependency, so agents and their unit tests depend on this interface and
// inject either the real NNEvaluationService or a scripted stub.
class EvalService {
 public:
  virtual ~EvalService() = default;

  // The served model's input-encoding arm and the input widths it consumes.
  // Agents build their InputEncodingSpec from the arm and validate the widths
  // against it through the layout registry.
  virtual bool contingent_features() const = 0;
  virtual int spatial_planes() const = 0;
  virtual int scalar_floats() const = 0;

  // `inputs` is count * the model's row float count.
  virtual void evaluate(const float* inputs, int count, Eval* out) = 0;
};

}  // namespace nn
}  // namespace scribblez

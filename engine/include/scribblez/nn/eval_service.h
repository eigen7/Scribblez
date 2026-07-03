#pragma once

namespace scribblez {
namespace nn {

// One model readout for a single encoded position.
struct Eval {
  // Win probability: P(win) + 0.5 * P(draw), the expected game points (1 for a
  // win, 0.5 for a draw) under the WLD head.
  float win_prob = 0.0f;
  float p_win = 0.0f;
  float p_draw = 0.0f;
  float p_loss = 0.0f;

  // Final score-differential Gaussian (mover's score minus opponent's) predicted
  // by the ScoreDiff head: its mean and standard deviation. The mean is the
  // points-oriented selection objective, closest in spirit to HastyBot's static
  // equity; the std is the model's predicted uncertainty about that mean.
  float score_diff_mean = 0.0f;
  float score_diff_std = 0.0f;
};

// Abstract evaluator over encoder rows: maps a contiguous block of `count` rows
// (each laid out as GameStateEncoder::encode_input() writes them) to one Eval
// per row. Carries no CUDA/TensorRT dependency, so consumers (the agents) and
// their unit tests can depend on this interface and inject either the real
// NNEvaluationService or a scripted stub.
class EvalService {
 public:
  virtual ~EvalService() = default;

  // The input widths the served model consumes -- the spatial plane count and
  // the scalar float count of one row. A row is their combined float count,
  // and agents derive their InputEncodingSpec (full vs base layout) from them.
  virtual int spatial_planes() const = 0;
  virtual int scalar_floats() const = 0;

  // Evaluate `count` rows (`inputs` is count * the model's row float count)
  // and write `count` results into `out`.
  virtual void evaluate(const float* inputs, int count, Eval* out) = 0;
};

}  // namespace nn
}  // namespace scribblez

#include "nn/eval_decode.h"

#include <Eigen/Core>

namespace scribblez {
namespace nn {

Eval decode_eval(const float* wld_logits, const float* score_diff) {
  Eval e;

  // Map the raw float pointer to an Eigen array (zero-copy), then softmax it
  // numerically stably: subtract the max before exponentiating.
  Eigen::Map<const Eigen::Array3f> v(wld_logits);
  Eigen::Array3f probs = (v - v.maxCoeff()).exp();
  probs /= probs.sum();

  e.p_win = probs[0];
  e.p_draw = probs[1];
  e.p_loss = probs[2];
  e.win_prob = e.p_win + 0.5f * e.p_draw;

  e.score_diff_mean = score_diff[0];
  e.score_diff_std = score_diff[1];
  return e;
}

}  // namespace nn
}  // namespace scribblez

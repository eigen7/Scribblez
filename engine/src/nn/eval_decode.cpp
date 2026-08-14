#include "nn/eval_service.h"

namespace scribblez {
namespace nn {

void make_evals(const float* wld_probs, const float* score_diff, int n, Eval* out) {
  for (int i = 0; i < n; ++i) {
    const float* p = wld_probs + static_cast<size_t>(i) * WldOutput::kRowElems;
    const float* sd = score_diff + static_cast<size_t>(i) * ScoreDiffOutput::kRowElems;
    Eval& e = out[i];
    e.p_win = p[0];
    e.p_draw = p[1];
    e.p_loss = p[2];
    e.win_prob = e.p_win + 0.5f * e.p_draw;
    e.score_diff_mean = sd[0];
    e.score_diff_std = sd[1];
  }
}

}  // namespace nn
}  // namespace scribblez

#include "scribblez/nn/nn_evaluation_service.h"

#include "scribblez/input_encoder.h"
#include "scribblez/training_targets.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstring>

namespace scribblez {
namespace nn {

namespace {

// Softmax the 3 WLD logits [win, draw, loss] into the WLD fields of `e`. The
// win-probability scalar treats a draw as half a win (expected game points).
void fill_wld(const float* logits, Eval& e) {
  // Map the raw float pointer to an Eigen array (zero-copy).
  Eigen::Map<const Eigen::Array3f> v(logits);

  // Numerically stable softmax: subtract the max before exponentiating.
  Eigen::Array3f probs = (v - v.maxCoeff()).exp();
  probs /= probs.sum();

  e.p_win = probs[0];
  e.p_draw = probs[1];
  e.p_loss = probs[2];
  e.win_prob = e.p_win + 0.5f * e.p_draw;
}

}  // namespace

NNEvaluationService::NNEvaluationService(const NeuralNetParams& params) : net_(params) {}

void NNEvaluationService::load() { net_.load(); }

void NNEvaluationService::evaluate(const float* inputs, int count, Eval* out) {
  const int batch = net_.max_batch_size();
  const size_t row_floats =
    static_cast<size_t>(net_.spatial_planes()) * kBoardCells + net_.scalar_floats();
  for (int start = 0; start < count; start += batch) {
    int chunk = std::min(batch, count - start);
    evaluate_chunk(inputs + static_cast<size_t>(start) * row_floats, chunk, out + start);
  }
}

void NNEvaluationService::evaluate_chunk(const float* inputs, int chunk, Eval* out) {
  float* spatial = net_.input_spatial_host();
  float* scalar = net_.input_scalar_host();

  // De-interleave each row's [spatial | scalar] block into the engine's two
  // separate, densely packed input buffers, at the model's own widths.
  const size_t spatial_floats = static_cast<size_t>(net_.spatial_planes()) * kBoardCells;
  const size_t scalar_floats = net_.scalar_floats();
  for (int r = 0; r < chunk; ++r) {
    const float* row = inputs + static_cast<size_t>(r) * (spatial_floats + scalar_floats);
    std::memcpy(spatial + static_cast<size_t>(r) * spatial_floats, row,
                sizeof(float) * spatial_floats);
    std::memcpy(scalar + static_cast<size_t>(r) * scalar_floats, row + spatial_floats,
                sizeof(float) * scalar_floats);
  }

  net_.predict(chunk);

  const float* wld = net_.wld_host();
  const float* sd = net_.score_diff_host();
  for (int r = 0; r < chunk; ++r) {
    Eval e;
    fill_wld(wld + static_cast<size_t>(r) * kWldFloats, e);
    // The score-diff head emits a Gaussian: [mean, std] of the final
    // differential. std is already positive (softplus is applied in-graph).
    const float* row = sd + static_cast<size_t>(r) * kScoreDiffOutputFloats;
    e.score_diff_mean = row[0];
    e.score_diff_std = row[1];
    out[r] = e;
  }
}

std::vector<Eval> NNEvaluationService::evaluate(const float* inputs, int count) {
  std::vector<Eval> out(count);
  if (count > 0) evaluate(inputs, count, out.data());
  return out;
}

}  // namespace nn
}  // namespace scribblez

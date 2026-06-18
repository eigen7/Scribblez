#include "scribblez/nn/nn_evaluation_service.h"

#include "scribblez/input_encoder.h"
#include "scribblez/training_targets.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace scribblez {
namespace nn {

using binlog::kInputFloats;
using binlog::kScalarFloats;
using binlog::kScoreDiffClip;
using binlog::kScoreDiffFloats;
using binlog::kSpatialFloats;
using binlog::kWldFloats;

namespace {

// Softmax the 3 WLD logits [win, draw, loss] into the WLD fields of `e`. The
// win-probability scalar treats a draw as half a win (expected game points).
void fill_wld(const float* logits, Eval& e) {
  float m = std::max({logits[0], logits[1], logits[2]});
  float ew = std::exp(logits[0] - m);
  float ed = std::exp(logits[1] - m);
  float el = std::exp(logits[2] - m);
  float z = ew + ed + el;
  e.p_win = ew / z;
  e.p_draw = ed / z;
  e.p_loss = el / z;
  e.win_prob = e.p_win + 0.5f * e.p_draw;
}

// Expected value of the ScoreDiff head: softmax the kScoreDiffFloats logits into
// a distribution over differentials (bin i == differential i - kScoreDiffClip)
// and return its mean. Uses the standard max-shift for numerical stability.
float score_diff_mean(const float* logits) {
  float m = logits[0];
  for (int i = 1; i < kScoreDiffFloats; ++i) m = std::max(m, logits[i]);
  double z = 0.0;
  double weighted = 0.0;
  for (int i = 0; i < kScoreDiffFloats; ++i) {
    double p = std::exp(static_cast<double>(logits[i] - m));
    z += p;
    weighted += p * (i - kScoreDiffClip);
  }
  return static_cast<float>(weighted / z);
}

}  // namespace

NNEvaluationService::NNEvaluationService(const NeuralNetParams& params) : net_(params) {}

void NNEvaluationService::load(const std::string& onnx_path) { net_.load(onnx_path); }

void NNEvaluationService::evaluate(const float* inputs, int count, Eval* out) {
  const int batch = net_.max_batch_size();
  for (int start = 0; start < count; start += batch) {
    int chunk = std::min(batch, count - start);
    evaluate_chunk(inputs + static_cast<size_t>(start) * kInputFloats, chunk, out + start);
  }
}

void NNEvaluationService::evaluate_chunk(const float* inputs, int chunk, Eval* out) {
  float* spatial = net_.input_spatial_host();
  float* scalar = net_.input_scalar_host();

  // De-interleave each row's [spatial | scalar] block into the engine's two
  // separate, densely packed input buffers.
  for (int r = 0; r < chunk; ++r) {
    const float* row = inputs + static_cast<size_t>(r) * kInputFloats;
    std::memcpy(spatial + static_cast<size_t>(r) * kSpatialFloats, row,
                sizeof(float) * kSpatialFloats);
    std::memcpy(scalar + static_cast<size_t>(r) * kScalarFloats, row + kSpatialFloats,
                sizeof(float) * kScalarFloats);
  }

  net_.predict(chunk);

  const float* wld = net_.wld_host();
  const float* sd = net_.score_diff_host();
  for (int r = 0; r < chunk; ++r) {
    Eval e;
    fill_wld(wld + static_cast<size_t>(r) * kWldFloats, e);
    e.score_diff_mean = score_diff_mean(sd + static_cast<size_t>(r) * kScoreDiffFloats);
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

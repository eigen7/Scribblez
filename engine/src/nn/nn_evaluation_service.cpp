#include "nn/nn_evaluation_service.h"

#include "encoding/input_encoder.h"
#include "nn/eval_decode.h"
#include "training/training_targets.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstring>

namespace scribblez {
namespace nn {

namespace {

constexpr int kMaskRowFloats = kNumMaskHeads * kBoardCells;

}  // namespace

NNEvaluationService::NNEvaluationService(const NeuralNetParams& params) : net_(params) {}

void NNEvaluationService::load() { net_.load(); }

void NNEvaluationService::evaluate(const float* inputs, int count, Eval* out) {
  evaluate(inputs, count, out, nullptr);
}

void NNEvaluationService::evaluate(const float* inputs, int count, Eval* out, float* masks_out) {
  const int batch = net_.max_batch_size();
  const size_t row_floats =
    static_cast<size_t>(net_.spatial_planes()) * kBoardCells + net_.scalar_floats();
  for (int start = 0; start < count; start += batch) {
    int chunk = std::min(batch, count - start);
    evaluate_chunk(inputs + start * row_floats, chunk, out + start,
                   masks_out ? masks_out + start * kMaskRowFloats : nullptr);
  }
}

void NNEvaluationService::evaluate_chunk(const float* inputs, int chunk, Eval* out,
                                         float* masks_out) {
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
    out[r] = decode_eval(wld + static_cast<size_t>(r) * kWldFloats,
                         sd + static_cast<size_t>(r) * kScoreDiffOutputFloats);
  }
  // The mask heads emit logits; consumers get probabilities, mirroring the WLD
  // softmax above.
  if (masks_out) {
    for (int h = 0; h < kNumMaskHeads; ++h) {
      const float* logits = net_.mask_host(h);
      for (int r = 0; r < chunk; ++r) {
        Eigen::Map<const Eigen::ArrayXf> in(logits + r * kBoardCells, kBoardCells);
        Eigen::Map<Eigen::ArrayXf> row(masks_out + r * kMaskRowFloats + h * kBoardCells,
                                       kBoardCells);
        row = 1.0f / (1.0f + (-in).exp());
      }
    }
  }
}

std::vector<Eval> NNEvaluationService::evaluate(const float* inputs, int count) {
  std::vector<Eval> out(count);
  if (count > 0) evaluate(inputs, count, out.data());
  return out;
}

std::unique_ptr<EvalService> make_loaded_service(const NeuralNetParams& params) {
  auto svc = std::make_unique<NNEvaluationService>(params);
  svc->load();
  return svc;
}

}  // namespace nn
}  // namespace scribblez

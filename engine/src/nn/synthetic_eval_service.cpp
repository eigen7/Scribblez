#include "nn/synthetic_eval_service.h"

#include "encoding/input_encoder.h"
#include "util/exception.h"
#include "util/hash.h"
#include "util/math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace scribblez {
namespace nn {

namespace {

constexpr std::string_view kSyntheticPrefix = "synthetic:";

// Each row is reduced to this many projections, from which every head value
// is derived (write_heads()).
constexpr int kProjections = 4;
using Projections = std::array<float, kProjections>;

// The head values' ranges: score-difference means within +-kMeanRange points,
// stds within [kStdFloor, kStdFloor + kStdRange].
constexpr float kMeanRange = 30.0f;
constexpr float kStdFloor = 5.0f;
constexpr float kStdRange = 10.0f;

// A weight in [-1, 1) for input element `i` of projection `k`.
float hashed_weight(int k, uint64_t i) {
  const uint64_t h = util::splitmix64(util::splitmix64(uint64_t(k)) ^ i);
  return float(h >> 40) * (2.0f / float(uint64_t{1} << 24)) - 1.0f;
}

// One row's running projections and squared norm, as its inputs are added.
struct Accumulator {
  Projections dot{};
  float norm_sq = 1.0f;  // never zero, so an all-zero row projects to 0
};

// Adds the inputs `x`, which occupy elements [offset, offset + x.size()) of
// the row the weight table spans (`stride` elements per projection).
template <typename T>
void accumulate(std::span<const T> x, const std::vector<float>& weights, size_t offset,
                Accumulator* acc) {
  const size_t stride = weights.size() / kProjections;
  for (size_t i = 0; i < x.size(); ++i) {
    const float v = float(x[i]);
    acc->norm_sq += v * v;
    for (int k = 0; k < kProjections; ++k) acc->dot[k] += v * weights[k * stride + offset + i];
  }
}

// The projections divided by the row's norm, so that rows of any scale and
// width land in the same small range.
Projections normalized(const Accumulator& acc) {
  const float inv_norm = 1.0f / std::sqrt(acc.norm_sq);
  Projections z = acc.dot;
  for (float& v : z) v *= inv_norm;
  return z;
}

// Adds candidate m's per-move inputs, the arrays (and widths) TrtEvalService
// stages, which follow the board row's `board_floats` in the weight table.
template <TensorDescriptor... Ts>
void accumulate_move(const move_set::MoveFeatureArrays& moves, int m,
                     const std::vector<float>& weights, size_t board_floats, Accumulator* acc,
                     TensorList<Ts...>) {
  size_t offset = board_floats;
  ((accumulate(std::span((moves.*Ts::kBatchSource).data() + size_t(m) * Ts::kRowElems,
                         size_t(Ts::kRowElems)),
               weights, offset, acc),
    offset += Ts::kRowElems),
   ...);
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Row `row`'s decoded head values from its projections: wld as probabilities
// (what TrtEvalService's softmax decode yields), score_diff as [mean, std > 0].
void write_heads(const Projections& z, bool non_finite, int row, std::span<float* const> head_out) {
  float* wld = head_out[0] + size_t(row) * WldOutput::kRowElems;
  float* sd = head_out[1] + size_t(row) * ScoreDiffOutput::kRowElems;
  if (non_finite) {
    std::fill_n(wld, WldOutput::kRowElems, std::numeric_limits<float>::quiet_NaN());
    std::fill_n(sd, ScoreDiffOutput::kRowElems, std::numeric_limits<float>::quiet_NaN());
    return;
  }
  // Draws are rare in real labels, so the draw logit sits lower.
  const std::array<float, 3> logits = {2.0f * z[0], 2.0f * z[1] - 2.0f, 2.0f * z[2]};
  const float top = std::max({logits[0], logits[1], logits[2]});
  float total = 0.0f;
  for (int i = 0; i < 3; ++i) total += wld[i] = std::exp(logits[i] - top);
  for (int i = 0; i < 3; ++i) wld[i] /= total;
  sd[0] = kMeanRange * std::tanh(z[3]);
  sd[1] = kStdFloor + kStdRange * sigmoid(z[0] - z[2]);
}

void evaluate_rows(const PositionEvaluationSpec::Batch& batch, const std::vector<float>& weights,
                   bool non_finite, std::span<float* const> head_out) {
  const size_t row_floats = weights.size() / kProjections;
  for (int r = 0; r < batch.count; ++r) {
    Accumulator acc;
    accumulate(std::span(batch.rows + r * row_floats, row_floats), weights, 0, &acc);
    write_heads(normalized(acc), non_finite, r, head_out);
  }
}

void evaluate_rows(const MoveSetEvaluationSpec::Batch& batch, const std::vector<float>& weights,
                   bool non_finite, std::span<float* const> head_out) {
  using MoveInputs = MoveSetEvaluationSpec::MoveInputs;
  const size_t board_floats = weights.size() / kProjections - MoveInputs::total_row_elems;
  Accumulator board;
  accumulate(std::span(batch.board_row, board_floats), weights, 0, &board);
  for (int m = 0; m < batch.moves->count; ++m) {
    Accumulator acc = board;
    accumulate_move(*batch.moves, m, weights, board_floats, &acc, MoveInputs{});
    write_heads(normalized(acc), non_finite, m, head_out);
  }
}

}  // namespace

std::optional<SyntheticModel> parse_synthetic_model(const std::string& model) {
  if (!model.starts_with(kSyntheticPrefix)) return std::nullopt;
  const std::string_view variant = std::string_view(model).substr(kSyntheticPrefix.size());
  if (variant == "hidden") return SyntheticModel{};
  if (variant == "open-leaves") return SyntheticModel{.opp_leave_input = true};
  if (variant == "non-finite") return SyntheticModel{.non_finite = true};
  throw util::CleanException(
    "unknown synthetic model '{}': expected synthetic:hidden, synthetic:open-leaves or "
    "synthetic:non-finite",
    model);
}

std::string model_content_hash(const std::string& model) {
  if (parse_synthetic_model(model)) return util::content_hash(model);
  return util::file_content_hash(model);
}

template <typename Spec>
SyntheticEvalService<Spec>::SyntheticEvalService(const SyntheticModel& model) : model_(model) {
  static_assert(std::is_same_v<typename Spec::Outputs, TensorList<WldOutput, ScoreDiffOutput>>,
                "write_heads() fills exactly the wld and score_diff heads");
  const size_t n = size_t(input_floats(InputEncodingSpec{nullptr, model.opp_leave_input}) +
                          Spec::MoveInputs::total_row_elems);
  weights_.resize(kProjections * n);
  for (int k = 0; k < kProjections; ++k) {
    for (size_t i = 0; i < n; ++i) weights_[k * n + i] = hashed_weight(k, i);
  }
}

template <typename Spec>
int SyntheticEvalService<Spec>::spatial_planes() const {
  return scribblez::spatial_planes();
}

template <typename Spec>
int SyntheticEvalService<Spec>::scalar_floats() const {
  return scribblez::scalar_floats(InputEncodingSpec{nullptr, model_.opp_leave_input});
}

template <typename Spec>
void SyntheticEvalService<Spec>::do_evaluate(const SpecBatch& batch,
                                             std::span<float* const> head_out) {
  evaluate_rows(batch, weights_, model_.non_finite, head_out);
}

template class SyntheticEvalService<PositionEvaluationSpec>;
template class SyntheticEvalService<MoveSetEvaluationSpec>;

}  // namespace nn
}  // namespace scribblez

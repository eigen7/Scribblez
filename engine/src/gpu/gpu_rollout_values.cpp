#include "gpu/gpu_rollout_values.h"

#include "encoding/input_encoder.h"
#include "encoding/leaf_eval_input_encoder.h"
#include "lexicon/hasty_equity.h"
#include "nn/onnx_metadata.h"
#include "util/exception.h"
#include "util/io.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace scribblez {

namespace {

// The maps' completion reads the heads in their export layout.
static_assert(nn::OwnershipOutput::kRowElems == rollout_abi::kOwnershipFloats);
static_assert(nn::SquarePointsOutput::kRowElems == rollout_abi::kSquarePointsFloats);
static_assert(nn::GoingOutOutput::kRowElems == rollout_abi::kWentOutClasses);
static_assert(nn::EndPointsOutput::kRowElems == rollout_abi::kEndPointsFloats);
static_assert(rollout_abi::kLeafPointsScale == kScoreDiffInputScale);

using Clock = std::chrono::steady_clock;

// The device buffer of the net's output `name`.
const float* device_rows(const nn::NeuralNetBase& net, const char* name) {
  return static_cast<const float*>(net.device_ptr(name));
}

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

nn::NeuralNetParams<nn::LeafEvalSpec> net_params(const GpuRolloutValues::Params& p,
                                                 const HastyEquity& eq) {
  nn::check_leave_values(nn::parse_onnx_metadata(util::read_file_bytes(p.leaf_onnx)), p.leaf_onnx,
                         eq.leave_values().content_hash());
  nn::NeuralNetParams<nn::LeafEvalSpec> params;
  params.onnx_path = p.leaf_onnx;
  params.precision = p.precision;
  params.max_rows = p.leaf_rows;
  return params;
}

}  // namespace

GpuRolloutValues::GpuRolloutValues(const Dictionary& dict, const HastyEquity& eq,
                                   const Params& params)
    : dict_(dict),
      leave_values_(eq.leave_values()),
      rollouts_(dict, eq),
      net_(net_params(params, eq)) {
  net_.load();
}

std::vector<LeafPrediction> GpuRolloutValues::predict(const std::vector<CpuLeaf>& leaves) {
  std::vector<LeafPrediction> out(leaves.size());
  std::vector<float> row(LeafEvalInputEncoder::kInputFloats);
  const int n = int(leaves.size());
  for (int first = 0; first < n; first += net_.max_rows()) {
    const int count = std::min(net_.max_rows(), n - first);
    for (int i = 0; i < count; ++i) {
      const CpuLeaf& leaf = leaves[size_t(first + i)];
      LeafEvalInputEncoder::encode(
        {&leaf.board, leaf.leaves[0], leaf.leaves[1], leaf.scores[0] - leaf.scores[1]}, dict_,
        leave_values_, row.data());
      nn::stage_board_row(net_, row.data(), i);
    }
    net_.predict(count);
    for (int i = 0; i < count; ++i) out[size_t(first + i)] = host_prediction(i);
  }
  return out;
}

LeafPrediction GpuRolloutValues::host_prediction(int row) {
  LeafPrediction p;
  p.wld = wld_from_logits(net_.host<nn::WldOutput>() + 3 * row);
  const auto copy = [row](const float* rows, auto& dst) {
    std::copy_n(rows + size_t(row) * dst.size(), dst.size(), dst.begin());
  };
  copy(net_.host<nn::OwnershipOutput>(), p.ownership);
  copy(net_.host<nn::SquarePointsOutput>(), p.square_points);
  copy(net_.host<nn::GoingOutOutput>(), p.going_out);
  copy(net_.host<nn::EndPointsOutput>(), p.end_points);
  return p;
}

BatchValues GpuRolloutValues::run(const RolloutBatch& batch) {
  auto t0 = Clock::now();
  const std::vector<rollout_abi::RolloutResult> results = rollouts_.run(batch, dict_);
  timings_.rollout_seconds = seconds_since(t0);
  timings_.kernel_seconds = rollouts_.last_timings().kernel_seconds;
  t0 = Clock::now();
  BatchValues out;
  out.outcomes.resize(results.size());
  std::vector<uint32_t> unfinished;
  for (size_t k = 0; k < results.size(); ++k) {
    out.outcomes[k].result = results[k];
    if (results[k].game_over)
      out.outcomes[k].wld = terminal_wld(results[k].diff);
    else
      unfinished.push_back(uint32_t(k));
  }
  const int n = int(unfinished.size());
  for (int first = 0; first < n; first += net_.max_rows()) {
    const int count = std::min(net_.max_rows(), n - first);
    evaluate_leaves(&unfinished[size_t(first)], count, &out.outcomes);
    if (batch.sum_maps) rollouts_.complete_maps(&unfinished[size_t(first)], count, device_heads());
  }
  if (batch.sum_maps) out.maps = rollouts_.maps();
  timings_.leaf_seconds = seconds_since(t0);
  timings_.leaves = int64_t(n);
  totals_.rollout_seconds += timings_.rollout_seconds;
  totals_.kernel_seconds += timings_.kernel_seconds;
  totals_.leaf_seconds += timings_.leaf_seconds;
  totals_.leaves += timings_.leaves;
  return out;
}

rollout_abi::LeafHeadRows GpuRolloutValues::device_heads() const {
  return {
    device_rows(net_, nn::OwnershipOutput::kName), device_rows(net_, nn::SquarePointsOutput::kName),
    device_rows(net_, nn::GoingOutOutput::kName), device_rows(net_, nn::EndPointsOutput::kName)};
}

void GpuRolloutValues::evaluate_leaves(const uint32_t* tasks, int count,
                                       std::vector<RolloutOutcome>* outcomes) {
  rollouts_.encode_leaves(tasks, count,
                          static_cast<float*>(net_.device_ptr(nn::SpatialInput::kName)),
                          static_cast<float*>(net_.device_ptr(nn::ScalarInput::kName)));
  net_.predict_device_inputs(count);
  const float* logits = net_.host<nn::WldOutput>();
  for (int i = 0; i < count; ++i) {
    RolloutOutcome& o = (*outcomes)[tasks[i]];
    o.wld = wld_from_logits(logits + 3 * i);
    if (!std::isfinite(o.wld[0]) || !std::isfinite(o.wld[2]))
      throw util::Exception("the leaf network gave a non-finite value (precision too narrow?)");
  }
}

}  // namespace scribblez

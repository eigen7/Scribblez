#include "gpu/gpu_rollout_values.h"

#include "lexicon/hasty_equity.h"
#include "nn/onnx_metadata.h"
#include "util/exception.h"
#include "util/io.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace scribblez {

namespace {

using Clock = std::chrono::steady_clock;

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
    : dict_(dict), rollouts_(dict, eq), net_(net_params(params, eq)) {
  net_.load();
}

std::vector<RolloutOutcome> GpuRolloutValues::run(const RolloutBatch& batch) {
  auto t0 = Clock::now();
  const std::vector<rollout_abi::RolloutResult> results = rollouts_.run(batch, dict_);
  timings_.rollout_seconds = seconds_since(t0);
  timings_.kernel_seconds = rollouts_.last_timings().kernel_seconds;
  t0 = Clock::now();
  std::vector<RolloutOutcome> outcomes(results.size());
  for (size_t k = 0; k < results.size(); ++k) outcomes[k].result = results[k];
  const int n = int(results.size());
  for (int first = 0; first < n; first += net_.max_rows())
    evaluate_leaves(first, std::min(net_.max_rows(), n - first), &outcomes);
  timings_.leaf_seconds = seconds_since(t0);
  totals_.rollout_seconds += timings_.rollout_seconds;
  totals_.kernel_seconds += timings_.kernel_seconds;
  totals_.leaf_seconds += timings_.leaf_seconds;
  return outcomes;
}

void GpuRolloutValues::evaluate_leaves(int first, int count,
                                       std::vector<RolloutOutcome>* outcomes) {
  rollouts_.encode_leaves(first, count,
                          static_cast<float*>(net_.device_ptr(nn::SpatialInput::kName)),
                          static_cast<float*>(net_.device_ptr(nn::ScalarInput::kName)));
  net_.predict_device_inputs(count);
  const float* logits = net_.host<nn::WldOutput>();
  for (int i = 0; i < count; ++i) {
    RolloutOutcome& o = (*outcomes)[size_t(first + i)];
    if (o.result.game_over) {
      o.wld = terminal_wld(o.result.diff);
      continue;
    }
    o.wld = wld_from_logits(logits + 3 * i);
    if (!std::isfinite(o.wld[0]) || !std::isfinite(o.wld[2]))
      throw util::Exception("the leaf network gave a non-finite value (precision too narrow?)");
  }
}

}  // namespace scribblez

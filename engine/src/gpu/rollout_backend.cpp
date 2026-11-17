#include "gpu/rollout_backend.h"

#include "gpu/gpu_rollout_values.h"
#include "lexicon/hasty_equity.h"
#include "nn/shared_registry.h"
#include "nn/trt_util.h"
#include "util/caller_led_batcher.h"
#include "util/exception.h"

#include <cstddef>
#include <exception>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

constexpr const char* kLeafFormatKey = "leaf_eval_format_version";

// What game threads share of one backend. GpuRolloutValues is not
// thread-safe, so its calls are serialized. One DeepBot turn already fills a
// rollout launch with ~10K rollouts, so concurrent games' run() calls queue
// rather than batch together. A leaf harness's turn values a few hundred
// leaves, too few to fill the GPU, so concurrent games' predictions are
// gathered into one call (util::CallerLedBatcher), one gathering per kind.
class SharedRolloutValues : public RolloutValues {
 public:
  explicit SharedRolloutValues(std::unique_ptr<RolloutValues> inner);

  BatchValues run(const RolloutBatch& batch) override;
  std::vector<LeafPrediction> predict_leaves(std::span<const CpuLeaf* const> leaves) override;
  std::vector<Wld> predict_wld(std::span<const CpuLeaf* const> leaves) override;

 private:
  // One caller's leaves, and its outputs or the exception that failed them.
  template <typename Out>
  struct Request {
    std::span<const CpuLeaf* const> leaves;
    std::vector<Out> out{};
    std::exception_ptr error = nullptr;
  };
  template <typename Out>
  using Batcher = util::CallerLedBatcher<Request<Out>>;

  // Submits `leaves` to `batcher`; returns their outputs or rethrows.
  template <typename Out>
  static std::vector<Out> submit(Batcher<Out>& batcher, std::span<const CpuLeaf* const> leaves);

  // Calls `predict` once over every request's leaves, under the backend's
  // lock, and hands each request its slice. Never throws: a failure is
  // recorded on every request of the pack.
  template <typename Out, typename Predict>
  void serve(std::span<Request<Out>* const> pack, Predict&& predict);

  std::unique_ptr<RolloutValues> inner_;
  std::mutex mutex_;  // the backend, one call at a time
  Batcher<LeafPrediction> predictions_;
  Batcher<Wld> values_;
};

SharedRolloutValues::SharedRolloutValues(std::unique_ptr<RolloutValues> inner)
    : inner_(std::move(inner)),
      predictions_([this](std::span<Request<LeafPrediction>* const> pack) {
        serve(pack,
              [this](std::span<const CpuLeaf* const> all) { return inner_->predict_leaves(all); });
      }),
      values_([this](std::span<Request<Wld>* const> pack) {
        serve(pack,
              [this](std::span<const CpuLeaf* const> all) { return inner_->predict_wld(all); });
      }) {}

BatchValues SharedRolloutValues::run(const RolloutBatch& batch) {
  std::lock_guard<std::mutex> lock(mutex_);
  return inner_->run(batch);
}

std::vector<LeafPrediction> SharedRolloutValues::predict_leaves(
  std::span<const CpuLeaf* const> leaves) {
  return submit(predictions_, leaves);
}

std::vector<Wld> SharedRolloutValues::predict_wld(std::span<const CpuLeaf* const> leaves) {
  return submit(values_, leaves);
}

template <typename Out>
std::vector<Out> SharedRolloutValues::submit(Batcher<Out>& batcher,
                                             std::span<const CpuLeaf* const> leaves) {
  Request<Out> req{leaves};
  batcher.submit(req);
  if (req.error) std::rethrow_exception(req.error);
  return std::move(req.out);
}

template <typename Out, typename Predict>
void SharedRolloutValues::serve(std::span<Request<Out>* const> pack, Predict&& predict) {
  std::vector<const CpuLeaf*> all;
  for (const Request<Out>* r : pack) all.insert(all.end(), r->leaves.begin(), r->leaves.end());
  try {
    std::vector<Out> out;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      out = predict(all);
    }
    auto next = out.begin();
    for (Request<Out>* r : pack) {
      r->out.assign(next, next + std::ptrdiff_t(r->leaves.size()));
      next += std::ptrdiff_t(r->leaves.size());
    }
  } catch (...) {
    const std::exception_ptr error = std::current_exception();
    for (Request<Out>* r : pack) r->error = error;
  }
}

}  // namespace

PoolModel pool_model(const nn::ModelIdentity& id) {
  return {id.hash, id.tag, id.generation, id.format_version};
}

void RolloutBackend::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()("leaf-model", po::value<std::string>(&leaf_model),
                     "the leaf network (a deep_bot_leaf_eval ONNX export); 'synthetic' with "
                     "--backend cpu")(
    "backend", po::value<std::string>(&backend)->default_value(backend),
    "gpu, or cpu (the CPU reference with the synthetic leaf, for tests)")(
    "leaf-precision", po::value<std::string>(&leaf_precision)->default_value(leaf_precision),
    "TensorRT precision for the leaf network: fp32, fp16 or bf16")(
    "leaf-rows", po::value<int>(&leaf_rows)->default_value(leaf_rows),
    "rows per leaf network call");
}

void RolloutBackend::validate() const {
  if (backend != "gpu" && backend != "cpu")
    throw util::CleanException("--backend must be gpu or cpu");
  if (leaf_model.empty()) throw util::CleanException("--leaf-model is required");
  if (gpu() && leaf_model == kSyntheticLeaf)
    throw util::CleanException("--leaf-model {} needs --backend cpu", kSyntheticLeaf);
  if (!gpu() && leaf_model != kSyntheticLeaf)
    throw util::CleanException("--backend cpu needs --leaf-model {}", kSyntheticLeaf);
}

PoolModel RolloutBackend::leaf_identity() const {
  if (!gpu()) return {"", kSyntheticLeaf, -1, 0};
  return pool_model(nn::read_model_identity(leaf_model, kLeafFormatKey));
}

std::unique_ptr<RolloutValues> RolloutBackend::make(const Dictionary& dict) const {
  if (!gpu()) return std::make_unique<CpuRolloutValues>(dict, synthetic_leaf());
  return std::make_unique<GpuRolloutValues>(
    dict, HastyEquity::instance(),
    GpuRolloutValues::Params{.leaf_onnx = leaf_model,
                             .precision = nn::parse_precision(leaf_precision),
                             .leaf_rows = leaf_rows});
}

std::shared_ptr<RolloutValues> RolloutBackend::shared(const Dictionary& dict) const {
  static nn::SharedRegistry<RolloutBackend, RolloutValues> registry;
  return registry.get_or_create(*this,
                                [&] { return std::make_shared<SharedRolloutValues>(make(dict)); });
}

}  // namespace scribblez

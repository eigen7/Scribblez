#include "gpu/rollout_backend.h"

#include "gpu/gpu_rollout_values.h"
#include "lexicon/hasty_equity.h"
#include "nn/shared_registry.h"
#include "nn/trt_util.h"
#include "util/exception.h"

#include <mutex>
#include <utility>

namespace scribblez {

namespace {

constexpr const char* kLeafFormatKey = "leaf_eval_format_version";

// Serializes a backend's run(): GpuRolloutValues is not thread-safe, and one
// DeepBot turn already fills a launch with ~10K rollouts, so concurrent games
// queue rather than batch together.
class LockedRolloutValues : public RolloutValues {
 public:
  explicit LockedRolloutValues(std::unique_ptr<RolloutValues> inner) : inner_(std::move(inner)) {}

  std::vector<RolloutOutcome> run(const RolloutBatch& batch) override {
    std::lock_guard<std::mutex> lock(mutex_);
    return inner_->run(batch);
  }

 private:
  std::unique_ptr<RolloutValues> inner_;
  std::mutex mutex_;
};

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
                                [&] { return std::make_shared<LockedRolloutValues>(make(dict)); });
}

}  // namespace scribblez

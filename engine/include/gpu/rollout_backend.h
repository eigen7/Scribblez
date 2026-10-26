#pragma once

// The rollout backend a GPU labeling tool runs on, as its command line names
// it: the CUDA kernel with a leaf network (gpu/gpu_rollout_values.h), or the
// CPU reference with the synthetic leaf, for the test suites, which hide the
// GPU. Shared by policy_set_pool_tool and blind_spot_confirm_tool, so their
// options and the leaf identity their outputs record cannot drift apart.

#include "data/pool_log.h"
#include "nn/model_specs.h"
#include "nn/onnx_metadata.h"
#include "sim/rollout_values.h"

#include <boost/program_options.hpp>

#include <memory>
#include <string>

namespace scribblez {

class Dictionary;

// A model's provenance as a .pool header records it.
PoolModel pool_model(const nn::ModelIdentity& id);

struct RolloutBackend {
  static constexpr const char* kSyntheticLeaf = "synthetic";

  std::string backend = "gpu";
  std::string leaf_model;  // an ONNX path, or kSyntheticLeaf with the cpu backend
  std::string leaf_precision = "fp16";
  int leaf_rows = nn::LeafEvalSpec::kDefaultMaxRows;

  bool operator==(const RolloutBackend&) const = default;

  bool gpu() const { return backend == "gpu"; }

  // --backend, --leaf-model, --leaf-precision and --leaf-rows.
  void add_options(boost::program_options::options_description& desc);
  // Throws util::CleanException for an inconsistent combination.
  void validate() const;
  // The leaf's provenance: the ONNX file's, or the synthetic leaf's.
  PoolModel leaf_identity() const;
  // The process-wide backend for these settings, shared by every caller that
  // asks while one holds it (SharedRegistry): the GPU one carries a leaf
  // engine and the lexicon tables on the device. Its run() is serialized, so
  // game threads may call it concurrently. The cpu one runs its rollouts on
  // `threads` threads; the gpu one ignores `threads`.
  std::shared_ptr<RolloutValues> shared(const Dictionary& dict, int threads) const;

  // An unshared backend. Factory-internal: agents go through shared() so that
  // heavy device state is paid once per process, not once per agent.
  std::unique_ptr<RolloutValues> make(const Dictionary& dict, int threads) const;
};

}  // namespace scribblez

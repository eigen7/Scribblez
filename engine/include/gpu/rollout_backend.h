#pragma once

// The rollout backend DeepBot's leaf network runs on: the CUDA kernel with a
// leaf network (gpu/gpu_rollout_values.h), or the CPU reference with the
// synthetic leaf, for the test suites, which hide the GPU.

#include "nn/model_specs.h"
#include "sim/rollout_values.h"

#include <memory>
#include <string>

namespace scribblez {

class Dictionary;

struct RolloutBackend {
  static constexpr const char* kSyntheticLeaf = "synthetic";

  std::string backend = "gpu";
  std::string leaf_model;  // an ONNX path, or kSyntheticLeaf with the cpu backend
  std::string leaf_precision = "fp16";
  int leaf_rows = nn::LeafEvalSpec::kDefaultMaxRows;

  bool operator==(const RolloutBackend&) const = default;

  bool gpu() const { return backend == "gpu"; }

  // Throws util::CleanException for an inconsistent combination.
  void validate() const;
  // The process-wide backend for these settings, shared by every caller that
  // asks while one holds it (SharedRegistry): the GPU one carries a leaf
  // engine and the lexicon tables on the device. Game threads may call it
  // concurrently: its run() calls are serialized, and its predictions of
  // concurrent callers are gathered into one network call. `dict` is not part
  // of the key, so every caller must pass the same (process-lifetime)
  // dictionary.
  std::shared_ptr<RolloutValues> shared(const Dictionary& dict) const;

  // An unshared backend. Factory-internal, and for a tool that runs one backend
  // per process: agents go through shared() so that heavy device state is paid
  // once per process, not once per agent.
  std::unique_ptr<RolloutValues> make(const Dictionary& dict) const;
};

}  // namespace scribblez

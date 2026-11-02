#pragma once

// The cross-check entries of a candidate set (training/cross_check_delta.h)
// computed on the GPU, one warp per candidate, with the lexicon's KWG resident
// on the device. It matches the CPU reference, cross_check_deltas, entry for
// entry (gpu_kernel_tests checks it), so an agent that reads its candidates'
// cross-check features from it feeds its model what training fed it.

#include "training/move_set_encoder.h"

#include <memory>
#include <mutex>

namespace scribblez {

class Dictionary;

namespace cross_check_gpu {
struct DeviceState;
}

class GpuCrossCheckDeltas : public move_set::CrossCheckDeltaSource {
 public:
  // The process's instance for `dict`, which must outlive it (SharedRegistry):
  // agents share one, never building their own.
  static std::shared_ptr<GpuCrossCheckDeltas> shared(const Dictionary& dict);

  // Uploads dict's KWG. Factory-internal (shared()), and for a tool or test
  // that runs one per process.
  explicit GpuCrossCheckDeltas(const Dictionary& dict);
  ~GpuCrossCheckDeltas() override;

  // Serialized: game threads may call it concurrently. `dict` must be the
  // dictionary this instance uploaded.
  void deltas(const Board& board, const Dictionary& dict, const Move* moves, int64_t n,
              cross_check_abi::MoveDeltas* out) override;

 private:
  const Dictionary& dict_;
  cross_check_gpu::DeviceState* state_;
  std::mutex mutex_;
};

}  // namespace scribblez

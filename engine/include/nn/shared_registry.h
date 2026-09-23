#pragma once

// A process-wide registry of live instances, keyed on the params that determine
// them. It backs the `create()` factories (PositionEvalService::create(),
// MoveProposalNets::create()) so that all the game threads of a run share one
// loaded model rather than each loading its own.
//
// Entries are held weakly: an instance is freed when its last holder drops it,
// and the next request for the same params builds a fresh one.

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace scribblez {
namespace nn {

template <typename Params, typename T>
class SharedRegistry {
 public:
  // The live instance for `params`, or else a new one from `make()`. The build
  // runs under the registry lock, so two callers racing on equal params get the
  // same instance: the second waits for the first's build to finish.
  template <typename Factory>
  std::shared_ptr<T> get_or_create(const Params& params, Factory&& make) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::erase_if(entries_, [](const auto& entry) { return entry.second.expired(); });
    for (const auto& [key, weak] : entries_) {
      if (key == params) {
        if (std::shared_ptr<T> live = weak.lock()) return live;
      }
    }
    std::shared_ptr<T> made = make();
    entries_.emplace_back(params, made);
    return made;
  }

 private:
  std::mutex mutex_;
  std::vector<std::pair<Params, std::weak_ptr<T>>> entries_;
};

}  // namespace nn
}  // namespace scribblez

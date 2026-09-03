#pragma once

// A process-wide registry of live shared instances keyed on the params that
// determine them -- the machinery behind every `create()` factory that lets a
// run's game threads resolve one set of params to ONE loaded model instead of
// one apiece (PositionEvalService::create(), MoveProposalNets::create()).
// Entries are held weakly, so an instance is freed once its last holder drops
// it; a later run rebuilds.

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace scribblez {
namespace nn {

template <typename Params, typename T>
class SharedRegistry {
 public:
  // The live instance registered for `params`, or -- when none is -- the one
  // `make()` builds, registered before it is returned. Serialized: two callers
  // racing on equal params get one instance, the second waiting out the
  // first's build.
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

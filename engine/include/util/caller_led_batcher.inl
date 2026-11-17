#include "util/caller_led_batcher.h"

#include <vector>

namespace scribblez {
namespace util {

template <typename Request>
void CallerLedBatcher<Request>::submit(Request& req) {
  Slot slot{&req};
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.push_back(&slot);
  if (dispatching_) {
    // Another caller is dispatching and will serve this request.
    served_.wait(lock, [&slot] { return slot.done; });
    return;
  }
  dispatch(lock);
}

template <typename Request>
void CallerLedBatcher<Request>::dispatch(std::unique_lock<std::mutex>& lock) {
  // The lock is released while a pack is served; requests arriving meanwhile
  // see dispatching_ set, wait, and are served by the next iteration.
  dispatching_ = true;
  std::vector<Slot*> slots;
  std::vector<Request*> pack;
  while (!queue_.empty()) {
    slots.assign(queue_.begin(), queue_.end());
    queue_.clear();
    pack.clear();
    for (Slot* s : slots) pack.push_back(s->req);
    lock.unlock();
    serve_(pack);
    lock.lock();
    for (Slot* s : slots) s->done = true;
    served_.notify_all();
  }
  dispatching_ = false;
}

}  // namespace util
}  // namespace scribblez

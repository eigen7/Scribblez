#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <span>
#include <utility>

namespace scribblez {
namespace util {

// Coalesces the concurrent requests of many threads into packs that one
// serve function handles a pack at a time: fewer, fuller calls than one per
// thread, for a resource such as a GPU engine that runs best on large batches
// and takes one caller at a time.
//
// The batching is caller-led, with no background thread. A caller enqueues its
// request and, if no dispatcher is running, becomes the dispatcher; otherwise
// it blocks until its request is served. The dispatcher repeatedly takes every
// queued request, serves them as one pack, and wakes their callers, until the
// queue is empty. Requests that arrive while a pack is served go into the next.
//
// Consequences worth knowing:
//   - There is no owner thread to start, stop, or outlive the callers.
//   - Only the dispatcher calls `serve`, so it runs one pack at a time.
//   - `serve` must not throw: an exception escaping it would strand the other
//     waiters. A Request carries its own error for its caller to rethrow.
template <typename Request>
class CallerLedBatcher {
 public:
  using Serve = std::function<void(std::span<Request* const>)>;

  explicit CallerLedBatcher(Serve serve) : serve_(std::move(serve)) {}

  // Returns once `req` has been served, as part of some pack.
  void submit(Request& req);

 private:
  struct Slot {
    Request* req;
    bool done = false;
  };

  // Serve the queue until it is empty, as the dispatcher. Called with `lock`
  // held; returns with it held.
  void dispatch(std::unique_lock<std::mutex>& lock);

  Serve serve_;
  std::mutex mutex_;
  std::condition_variable served_;
  std::deque<Slot*> queue_;
  bool dispatching_ = false;
};

}  // namespace util
}  // namespace scribblez

#include "util/caller_led_batcher.inl"

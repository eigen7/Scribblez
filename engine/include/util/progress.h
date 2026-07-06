#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

namespace util {

// Tracks completed work items across threads and renders a carriage-return
// progress bar (with rate and ETA) to stderr, refreshed by an internal
// monitor thread. The bar is suppressed when stderr is not a TTY, so piped or
// captured output stays clean. finish() stops the monitor, clears the bar,
// and prints a one-line timing summary; it must be called before destruction
// once work is complete (the destructor only stops the monitor).
class ProgressMeter {
 public:
  // `total` items expected; `noun` names them in the bar and the summary
  // (e.g. "positions"). Rendering starts immediately.
  ProgressMeter(uint64_t total, std::string noun);
  ~ProgressMeter();

  ProgressMeter(const ProgressMeter&) = delete;
  ProgressMeter& operator=(const ProgressMeter&) = delete;

  // Marks one item complete. Thread-safe.
  void add_done() { done_.fetch_add(1, std::memory_order_relaxed); }
  uint64_t done() const { return done_.load(std::memory_order_relaxed); }
  double elapsed_s() const;

  // Stops the monitor, clears the bar line, and prints
  // "<prefix>: <done> <noun> in <T>s (<rate> <noun>/s)" to stderr.
  void finish(const std::string& prefix);

 private:
  void render() const;
  void monitor_loop();
  void stop_monitor();

  const uint64_t total_;
  const std::string noun_;
  std::atomic<uint64_t> done_{0};
  const std::chrono::steady_clock::time_point start_;
  std::atomic<bool> stop_{false};
  std::thread monitor_;
};

}  // namespace util

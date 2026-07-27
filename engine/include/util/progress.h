#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

namespace scribblez::util {

// Tracks completed work items across threads and renders a carriage-return
// progress bar (with rate and ETA) to stderr from an internal monitor thread.
// The bar is suppressed when stderr is not a TTY, so piped output stays clean.
class ProgressMeter {
 public:
  // `noun` names the items in the bar and the summary (e.g. "positions").
  // Rendering starts immediately.
  ProgressMeter(uint64_t total, std::string noun);
  ~ProgressMeter();

  ProgressMeter(const ProgressMeter&) = delete;
  ProgressMeter& operator=(const ProgressMeter&) = delete;

  // Thread-safe.
  void add_done() { done_.fetch_add(1, std::memory_order_relaxed); }
  uint64_t done() const { return done_.load(std::memory_order_relaxed); }
  double elapsed_s() const;

  // Stops the monitor, clears the bar, and prints a one-line timing summary.
  // Must be called once work is complete; the destructor alone only stops the
  // monitor.
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

}  // namespace scribblez::util

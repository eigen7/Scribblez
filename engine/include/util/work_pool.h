#pragma once

// The process's CPU workers: one pool, sized to the process's CPU budget, that
// every parallel loop in the engine runs on. A loop runs on the calling thread
// plus whichever pool threads are idle, so the same agent code uses every core
// when it is alone and degrades to running on its own thread when every core
// is busy with other games: no per-agent thread counts to tune, and no
// oversubscription when many agents fork at once.

#include <atomic>
#include <cstddef>
#include <functional>

namespace scribblez::util {

// Sets the CPU budget: how many threads, the caller included, a parallel loop
// may run on. Defaults to default_thread_count(). play_game and the tools set
// it from their --threads at startup.
void set_worker_count(int n);
int worker_count();

// Runs body(i) for every i in [0, n) and returns once all have finished. If a
// body throws, rethrows the first exception after the rest have finished.
void parallel_for(size_t n, const std::function<void(size_t)>& body);

// A lane's handle on the items of a parallel_lanes loop.
class LaneCursor {
 public:
  // This lane's index, in [0, lane_count(n)): it selects per-lane outputs.
  size_t lane() const { return lane_; }

  // Claims the next unclaimed item into `item`; false once none remain.
  bool next(size_t& item);

 private:
  friend void parallel_lanes(size_t, const std::function<void(LaneCursor&)>&);
  LaneCursor(size_t lane, size_t n, std::atomic<size_t>* next) : lane_(lane), n_(n), next_(next) {}

  size_t lane_;
  size_t n_;
  std::atomic<size_t>* next_;  // the loop's shared claim counter
};

// The number of lanes parallel_lanes(n, ...) runs: min(n, worker_count()).
size_t lane_count(size_t n);

// For loops whose workers carry state (scratch buffers, agents, a batcher):
// runs lane(cursor) once per lane, each lane building its state once and then
// claiming items until none remain, so every item in [0, n) is claimed exactly
// once. A lane that starts after the items are exhausted is skipped. Throws as
// parallel_for does.
void parallel_lanes(size_t n, const std::function<void(LaneCursor&)>& lane);

}  // namespace scribblez::util

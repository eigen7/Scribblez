#pragma once

namespace util {

// The number of logical processors available to this process (its CPU affinity
// mask, so taskset/cgroup cpusets are respected). This is the project-wide
// default for worker-thread counts: compute-bound thread pools (self-play
// games, Monte Carlo workers) default to using every available logical
// processor. The Python counterpart is scribblez.hardware.default_thread_count.
int default_thread_count();

}  // namespace util

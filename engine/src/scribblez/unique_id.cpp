#include "scribblez/unique_id.h"

#include <atomic>
#include <chrono>
#include <cstdint>

namespace scribblez {

uint64_t get_unique_id() {
  static std::atomic<uint64_t> last_id{0};
  while (true) {
    uint64_t ts = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    uint64_t expected = last_id.load(std::memory_order_relaxed);
    if (ts <= expected) continue;  // clock hasn't advanced past last returned value; retry
    if (last_id.compare_exchange_weak(expected, ts, std::memory_order_acq_rel,
                                      std::memory_order_relaxed)) {
      return ts;
    }
    // Another thread updated last_id between our load and CAS; retry.
  }
}

}  // namespace scribblez

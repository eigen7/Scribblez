#pragma once

#include <cstdint>

namespace scribblez {

// Returns a unique uint64_t nanosecond Unix timestamp. Thread-safe. If the
// current clock reading equals the last value returned, spins until the clock
// advances so that every returned value is strictly greater than the previous.
uint64_t get_unique_id();

}  // namespace scribblez

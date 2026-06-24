#pragma once

#include <bit>
#include <cstdint>

namespace util {

// Smallest power of two that is >= n (and >= 1, since std::bit_ceil(0) == 1).
inline uint64_t round_up_pow2(uint64_t n) { return std::bit_ceil(n); }

// Round n up to the next multiple of alignment. alignment must be a power of two.
constexpr uint64_t align_up(uint64_t n, uint64_t alignment) {
  return (n + alignment - 1) & ~(alignment - 1);
}

}  // namespace util

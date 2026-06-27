#pragma once

#include <bit>
#include <cstdint>
#include <random>
#include <vector>

namespace util {

// Smallest power of two that is >= n (and >= 1, since std::bit_ceil(0) == 1).
inline uint64_t round_up_pow2(uint64_t n) { return std::bit_ceil(n); }

// Round n up to the next multiple of alignment. alignment must be a power of two.
constexpr uint64_t align_up(uint64_t n, uint64_t alignment) {
  return (n + alignment - 1) & ~(alignment - 1);
}

// Draws an index from a numerically-stable softmax over a vector of scores.
// Shared by every agent that turns a vector of candidate scores (static equity,
// or a value head's output) into an exploratory move choice.
//
// The sampler owns a weight buffer that sample() reuses across calls, so persist
// one instance over the lifetime of the caller (e.g. as a member) to avoid a
// per-call allocation.
class SoftmaxSampler {
 public:
  // Returns an index in [0, k) drawn with probability proportional to
  // exp((values[i] - max) / temperature), shifting by the max for stability.
  // Preconditions: k >= 1 and temperature > 0.
  int sample(const std::vector<double>& values, int k, double temperature, std::mt19937_64& rng);

 private:
  std::vector<double> weights_;
};

}  // namespace util

#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace scribblez::util {

// Smallest power of two >= n, and 1 for n == 0 (std::bit_ceil's convention).
inline uint64_t round_up_pow2(uint64_t n) { return std::bit_ceil(n); }

// splitmix64 finalizer: decorrelates structured integers (file/game/turn
// triples, seeds) into well-mixed 64-bit values.
constexpr uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// `alignment` must be a power of two.
constexpr uint64_t align_up(uint64_t n, uint64_t alignment) {
  return (n + alignment - 1) & ~(alignment - 1);
}

// Draws an index from a numerically-stable softmax over a vector of scores.
// It owns a weight buffer that sample() reuses, so hold one instance for the
// caller's lifetime rather than paying a per-call allocation.
class SoftmaxSampler {
 public:
  // An index in [0, k) drawn with probability proportional to
  // exp((values[i] - max) / temperature). Requires k >= 1 and temperature > 0.
  int sample(const std::vector<double>& values, int k, double temperature, std::mt19937_64& rng);

 private:
  std::vector<double> weights_;
};

// Linear index of cell (r, c) in a row-major side x side grid, reflected
// across the main diagonal when `transpose`.
constexpr int plane_index(int r, int c, int side, bool transpose) {
  return transpose ? (c * side + r) : (r * side + c);
}

// Row/column deltas of the four orthogonal neighbors: up, down, left, right.
constexpr std::array<std::pair<int, int>, 4> kFourNeighborDeltas = {
  {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

}  // namespace scribblez::util

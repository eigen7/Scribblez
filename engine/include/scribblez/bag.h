#pragma once
#include "scribblez/tile.h"

#include <array>
#include <optional>
#include <random>

namespace scribblez {

// A Bag holds remaining tiles and supports uniform-random draws.
class Bag {
 public:
  // Initialize from the standard English tile distribution.
  explicit Bag(uint64_t seed);

  std::optional<Letter> draw();
  void put_back(Letter t);
  int size() const { return remaining_; }
  const std::array<int, 27>& counts() const { return counts_; }

 private:
  std::array<int, 27> counts_{};
  int remaining_ = 0;
  std::mt19937_64 rng_;
};

}  // namespace scribblez

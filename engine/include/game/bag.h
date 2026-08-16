#pragma once

#include "game/tile.h"

#include <array>
#include <optional>
#include <random>

namespace scribblez {

class Bag {
 public:
  // The full bag's tile count (the standard English distribution); the
  // constructor asserts the distribution table still sums to it.
  static constexpr int kTotalTiles = 100;

  // Initialize from the standard English tile distribution.
  explicit Bag(uint64_t seed);

  std::optional<Tile> draw();
  void put_back(Tile t);
  // `t` must be present. Lets a caller carve an unseen-tile pool out of a full
  // bag.
  void remove(Tile t);
  int size() const { return remaining_; }
  const std::array<int, 27>& counts() const { return counts_; }

 private:
  std::array<int, 27> counts_{};
  int remaining_ = 0;
  std::mt19937_64 rng_;
};

}  // namespace scribblez

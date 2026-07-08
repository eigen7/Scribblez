#pragma once

#include "game/tile.h"

#include <array>
#include <optional>
#include <random>

namespace scribblez {

// A Bag holds remaining tiles and supports uniform-random draws.
class Bag {
 public:
  // Initialize from the standard English tile distribution.
  explicit Bag(uint64_t seed);

  std::optional<Tile> draw();
  void put_back(Tile t);
  // Remove one tile `t` from the bag (it must be present). Used to build a bag of
  // the tiles unseen from a player's POV: start full, then remove the tiles on the
  // board and in the player's own rack.
  void remove(Tile t);
  int size() const { return remaining_; }
  const std::array<int, 27>& counts() const { return counts_; }

 private:
  std::array<int, 27> counts_{};
  int remaining_ = 0;
  std::mt19937_64 rng_;
};

}  // namespace scribblez

#pragma once

#include "game/tile.h"
#include "game/tile_counts.h"

#include <random>

namespace scribblez {

class Bag {
 public:
  // Tiles in a full bag under TILE_COUNTS, the standard English distribution
  // a new Bag starts from.
  static constexpr int kTotalTiles = 100;

  explicit Bag(uint64_t seed) : Bag(seed, TileCounts::full_distribution()) {}
  // A bag holding exactly `tiles`, e.g. Board::unseen_tiles() for a pool of
  // the tiles a player cannot see.
  Bag(uint64_t seed, const TileCounts& tiles);

  Tile draw();  // the bag must not be empty
  void put_back(Tile t);
  // `t` must be present. For taking tiles known to be elsewhere out of a pool.
  void remove(Tile t);
  int size() const { return remaining_; }
  const TileCounts& counts() const { return counts_; }

 private:
  TileCounts counts_;
  int remaining_ = 0;  // counts_.size(), kept to make size() O(1)
  std::mt19937_64 rng_;
};

}  // namespace scribblez

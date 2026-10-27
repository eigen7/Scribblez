#pragma once

#include "game/tile.h"
#include "game/tile_counts.h"

#include <random>
#include <vector>

namespace scribblez {

// The bag a Game draws from: shuffled once, then seat 0 draws from the front
// and seat 1 from the back. Each draw is still uniform over the tiles left, as
// no draw looks at what it takes, but a seat's k-th tile no longer depends on
// how many the other seat has drawn. Two matches on one seed then deal each
// seat largely the same tiles however their play differs, which makes paired
// comparisons between agents far less noisy than independent draws.
class TwoEndedBag {
 public:
  explicit TwoEndedBag(uint64_t seed) : TwoEndedBag(seed, TileCounts::full_distribution()) {}
  TwoEndedBag(uint64_t seed, const TileCounts& tiles);

  Tile draw(int seat);  // the bag must not be empty
  // Inserts `t` at a uniformly random position, which keeps every order of the
  // bag equally likely and so keeps later draws uniform.
  void put_back(Tile t);
  int size() const { return int(tiles_.size()); }

 private:
  std::vector<Tile> tiles_;  // seat 0 draws from the front, seat 1 from the back
  std::mt19937_64 rng_;
};

}  // namespace scribblez

#include "game/two_ended_bag.h"

#include "util/assert.h"

#include <algorithm>

namespace scribblez {

TwoEndedBag::TwoEndedBag(uint64_t seed, const TileCounts& tiles) : rng_(seed) {
  tiles_.reserve(tiles.size());
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) tiles_.insert(tiles_.end(), tiles.count(t), t);
  std::shuffle(tiles_.begin(), tiles_.end(), rng_);
}

Tile TwoEndedBag::draw(int seat) {
  DEBUG_ASSERT(!tiles_.empty());
  if (seat == 1) {
    const Tile t = tiles_.back();
    tiles_.pop_back();
    return t;
  }
  const Tile t = tiles_.front();
  tiles_.erase(tiles_.begin());
  return t;
}

void TwoEndedBag::put_back(Tile t) {
  std::uniform_int_distribution<size_t> slot(0, tiles_.size());
  tiles_.insert(tiles_.begin() + slot(rng_), t);
}

}  // namespace scribblez

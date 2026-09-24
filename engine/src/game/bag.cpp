#include "game/bag.h"

#include "util/assert.h"

#include <cstdlib>

namespace scribblez {

Bag::Bag(uint64_t seed, const TileCounts& tiles)
    : counts_(tiles), remaining_(tiles.size()), rng_(seed) {}

Tile Bag::draw() {
  DEBUG_ASSERT(remaining_ > 0);
  std::uniform_int_distribution<int> dist(0, remaining_ - 1);
  int k = dist(rng_);
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) {
    if (k < counts_.count(t)) {
      counts_.remove(t);
      --remaining_;
      return t;
    }
    k -= counts_.count(t);
  }
  std::abort();  // unreachable: remaining_ is the sum of counts_
}

void Bag::put_back(Tile t) {
  counts_.add(t);
  ++remaining_;
}

void Bag::remove(Tile t) {
  const bool present = counts_.remove(t);
  DEBUG_ASSERT(present, "Bag::remove: tile not present");
  --remaining_;
}

}  // namespace scribblez

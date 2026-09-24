#include "game/bag.h"

#include "util/assert.h"

#include <cstdlib>

namespace scribblez {

Bag::Bag(uint64_t seed) : rng_(seed) {
  counts_ = TILE_COUNTS;
  for (int c : counts_) remaining_ += c;
  DEBUG_ASSERT(remaining_ == kTotalTiles);
}

Tile Bag::draw() {
  DEBUG_ASSERT(remaining_ > 0);
  std::uniform_int_distribution<int> dist(0, remaining_ - 1);
  int k = dist(rng_);
  for (Tile l = Tile::of(0); l < counts_.size(); ++l) {
    if (k < counts_[l]) {
      --counts_[l];
      --remaining_;
      return l;
    }
    k -= counts_[l];
  }
  std::abort();  // unreachable: remaining_ is the sum of counts_
}

void Bag::put_back(Tile t) {
  ++counts_[t];
  ++remaining_;
}

void Bag::remove(Tile t) {
  DEBUG_ASSERT(counts_[t] > 0, "Bag::remove: tile not present");
  --counts_[t];
  --remaining_;
}

}  // namespace scribblez

#include "scribblez/bag.h"

namespace scribblez {

Bag::Bag(uint64_t seed) : rng_(seed) {
  counts_ = TILE_COUNTS;
  for (int c : counts_) remaining_ += c;
}

std::optional<Tile> Bag::draw() {
  if (remaining_ == 0) return std::nullopt;
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
  return std::nullopt;  // unreachable
}

void Bag::put_back(Tile t) {
  ++counts_[t];
  ++remaining_;
}

}  // namespace scribblez

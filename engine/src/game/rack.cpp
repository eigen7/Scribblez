#include "game/rack.h"

#include "util/exception.h"

#include <cstring>

namespace scribblez {

Rack Rack::from_counts(const TileCounts& counts) {
  if (counts.size() > RACK_SIZE) {
    throw util::Exception("a rack holds at most {} tiles, not {}", RACK_SIZE, counts.size());
  }
  Rack rack;
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) {
    for (int i = 0; i < counts.count(t); ++i) rack.add(t);
  }
  return rack;
}

uint64_t Rack::bits() const {
  uint64_t b;
  std::memcpy(&b, this, sizeof(b));
  return b;
}

void Rack::add(Tile t) {
  // Insertion sort; the caller never exceeds RACK_SIZE tiles.
  int i = size_;
  while (i > 0 && t < tiles_[i - 1]) {
    tiles_[i] = tiles_[i - 1];
    --i;
  }
  tiles_[i] = t;
  ++size_;
}

bool Rack::remove(Tile t) {
  for (int i = 0; i < size_; ++i) {
    if (tiles_[i] == t) {
      for (int j = i; j + 1 < size_; ++j) tiles_[j] = tiles_[j + 1];
      --size_;
      tiles_[size_] = Tile::empty();
      return true;
    }
  }
  return false;
}

int Rack::count(Tile t) const {
  int n = 0;
  for (int i = 0; i < size_; ++i)
    if (tiles_[i] == t) ++n;
  return n;
}

std::string Rack::to_string() const {
  std::string s;
  for (int i = 0; i < size_; ++i) s.push_back(tiles_[i].to_char());
  return s;
}

int Rack::point_value() const {
  int v = 0;
  for (int i = 0; i < size_; ++i) v += tiles_[i].value();
  return v;
}

TileCounts Rack::counts() const {
  TileCounts c;
  for (int i = 0; i < size_; ++i) c.add(tiles_[i]);
  return c;
}

}  // namespace scribblez

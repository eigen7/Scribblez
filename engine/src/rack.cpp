#include "scribblez/rack.h"

namespace scribblez {

void Rack::add(Tile t) {
  // Insertion sort into the fixed array (RACK_SIZE is tiny). The caller never
  // exceeds RACK_SIZE tiles.
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
  // tiles_ is sorted, so letters come before blanks; a blank renders as '?'.
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

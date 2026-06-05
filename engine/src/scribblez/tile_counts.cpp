#include "scribblez/tile_counts.h"

namespace scribblez {

std::string TileCounts::to_string() const {
  std::string s;
  for (Tile l = Tile::of(0); l < 26; ++l) {
    for (int i = 0; i < counts_[l]; ++i) s.push_back(l.to_char());
  }
  for (int i = 0; i < counts_[BLANK]; ++i) s.push_back('?');
  return s;
}

int TileCounts::point_value() const {
  int v = 0;
  for (Tile l = Tile::of(0); l < 26; ++l) v += counts_[l] * TILE_VALUES[l];
  return v;
}

}  // namespace scribblez

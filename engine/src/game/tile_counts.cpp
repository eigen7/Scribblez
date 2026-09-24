#include "game/tile_counts.h"

#include "util/exception.h"

#include <algorithm>

namespace scribblez {

TileCounts TileCounts::full_distribution() {
  TileCounts all;
  all.counts_ = TILE_COUNTS;
  return all;
}

TileCounts TileCounts::from_string(std::string_view letters) {
  TileCounts tiles;
  for (const char c : letters) {
    const Tile t = c == '?' ? BLANK : Tile::letter_from_char(c);
    if (t.is_empty()) throw util::Exception("invalid tile '{}' (use A-Z, or ? for a blank)", c);
    tiles.add(t);
  }
  return tiles;
}

bool TileCounts::remove(const TileCounts& other) {
  bool ok = true;
  for (int t = 0; t < TILE_KINDS; ++t) {
    ok &= counts_[t] >= other.counts_[t];
    counts_[t] = std::max(0, counts_[t] - other.counts_[t]);
  }
  return ok;
}

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

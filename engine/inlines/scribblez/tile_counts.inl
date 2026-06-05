#pragma once

#include "scribblez/tile_counts.h"

namespace scribblez {

inline bool TileCounts::remove(Tile t) {
  if (counts_[t] == 0) return false;
  --counts_[t];
  return true;
}

inline int TileCounts::size() const {
  int s = 0;
  for (int c : counts_) s += c;
  return s;
}

}  // namespace scribblez

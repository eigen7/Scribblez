#pragma once

#include "scribblez/move.h"

namespace scribblez {

// Number of leading non-empty glyphs (placed/surrendered tiles).
inline int Move::num_glyphs() const {
  int n = 0;
  for (Glyph g : glyphs) {
    if (g.is_empty()) break;
    ++n;
  }
  return n;
}

}  // namespace scribblez

#include "game/move.h"

namespace scribblez {

// All Move accessors are single-line and defined in the header. Multi-line
// reconstruction logic (leave, word_origin, main_word) and MoveFactory live in
// move.cpp.

template <typename F>
void visit_placed_squares(const Move& m, F&& f) {
  if (m.type() != MoveType::PLAY) return;
  const bool horizontal = m.horizontal();
  const int start = m.start();
  uint16_t mask = m.square_mask();
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    f(r, c);
  }
}

}  // namespace scribblez

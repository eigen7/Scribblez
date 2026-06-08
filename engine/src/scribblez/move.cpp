#include "scribblez/move.h"

#include <algorithm>
#include <array>
#include <cassert>

namespace scribblez {

namespace {
struct Step {
  int dr, dc;
};
Step step_of(const Move& m) { return m.horizontal ? Step{0, 1} : Step{1, 0}; }
}  // namespace

Rack Move::leave(const Rack& rack) const {
  // Collect and sort this move's tiles, then two-pointer merge against the
  // (already sorted) rack: survivors are appended in order, so each Rack::add
  // is an O(1) tail insert.
  std::array<Tile, RACK_SIZE> removed;
  int n = 0;
  for (const Glyph& g : glyphs) {
    if (g.is_empty()) break;
    removed[n++] = g.rack_tile();
  }
  std::sort(removed.begin(), removed.begin() + n);

  Rack leave;
  int j = 0;
  for (Tile t : rack.tiles()) {
    if (t.is_empty()) break;
    if (j < n && t == removed[j]) {
      ++j;  // tile consumed by the move
    } else {
      leave.add(t);
    }
  }
  assert(j == n);  // every played tile was present on the rack
  return leave;
}

std::string Move::main_word(const Board& board) const {
  std::string word;
  if (type != MoveType::PLAY) return word;
  Step s = step_of(*this);
  int r = start_row, c = start_col, gi = 0;
  const int n = num_glyphs();
  while (board.in_bounds(r, c)) {
    Glyph cell = board.at(r, c);
    if (!cell.is_empty()) {
      word.push_back(cell.letter().to_char());  // existing tile
    } else if (gi < n) {
      word.push_back(glyphs[gi++].letter().to_char());  // newly placed tile
    } else {
      break;  // empty square, no tiles left to place: word ends here
    }
    r += s.dr;
    c += s.dc;
  }
  return word;
}

}  // namespace scribblez

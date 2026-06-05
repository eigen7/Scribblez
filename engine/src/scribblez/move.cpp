#include "scribblez/move.h"

namespace scribblez {

namespace {
struct Step {
  int dr, dc;
};
Step step_of(const Move& m) { return m.horizontal ? Step{0, 1} : Step{1, 0}; }
}  // namespace

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

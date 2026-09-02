#include "game/move.h"

#include "util/assert.h"

namespace scribblez {

namespace {
struct Step {
  int dr, dc;
};
Step step_of(const Move& m) { return m.horizontal() ? Step{0, 1} : Step{1, 0}; }
}  // namespace

std::pair<int, int> Move::word_origin(const Board& board) const {
  DEBUG_ASSERT(board.transposed() == transposed_);
  // The lowest set bit of the absolute mask is the first newly placed lane
  // cell; the word may extend left of it through pre-existing tiles.
  int along = 0;
  for (uint16_t m = square_mask_; (m & 1u) == 0; m >>= 1) ++along;
  Step s = step_of(*this);
  int r = horizontal_ ? start_ : along;
  int c = horizontal_ ? along : start_;
  while (board.in_bounds(r - s.dr, c - s.dc) && !board.at(r - s.dr, c - s.dc).is_empty()) {
    r -= s.dr;
    c -= s.dc;
  }
  return {r, c};
}

std::string Move::main_word(const Board& board) const {
  std::string word;
  if (type_ != MoveType::PLAY) return word;
  Step s = step_of(*this);
  auto [r, c] = word_origin(board);
  int gi = 0;
  const int n = num_played_;
  while (board.in_bounds(r, c)) {
    Glyph cell = board.at(r, c);
    if (!cell.is_empty()) {
      word.push_back(cell.letter().to_char());  // existing tile
    } else if (gi < n) {
      word.push_back(glyphs_[gi++].letter().to_char());  // newly placed tile
    } else {
      break;  // empty square, no tiles left to place: word ends here
    }
    r += s.dr;
    c += s.dc;
  }
  return word;
}

// ---------------------------------------------------------------------------

namespace {
// Lay a multiset of rack tiles into `glyphs` starting at `j`, sorted ascending
// (A..Z then blanks) -- the same order a Rack keeps. Returns the next index.
int append_sorted(std::array<Glyph, RACK_SIZE>& glyphs, int j, const TileCounts& tiles) {
  for (Tile t = Tile::of(0); t <= BLANK; ++t) {
    for (int k = 0; k < tiles.count(t); ++k) glyphs[j++] = Glyph::exchanging(t);
  }
  return j;
}
}  // namespace

Move Move::play(bool horizontal, int start, uint16_t square_mask, uint16_t score,
                const Glyph* played, int num_played, bool transposed) {
  Move m;
  m.type_ = MoveType::PLAY;
  m.transposed_ = transposed;
  m.horizontal_ = horizontal;
  m.start_ = start;
  m.num_played_ = num_played;
  m.square_mask_ = square_mask;
  m.score_ = score;
  for (int i = 0; i < num_played; ++i) m.glyphs_[i] = played[i];
  return m;
}

Move Move::transpose() const {
  Move m = *this;
  m.transposed_ = !transposed_;
  if (type_ == MoveType::PLAY) m.horizontal_ = !horizontal_;
  return m;
}

Move Move::exchange(const TileCounts& tiles) {
  Move m;
  m.type_ = MoveType::EXCHANGE;
  m.num_played_ = append_sorted(m.glyphs_, 0, tiles);
  return m;
}

}  // namespace scribblez

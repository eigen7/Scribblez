#include "sim/setup_plays.h"

#include "lexicon/dictionary.h"

#include <array>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

constexpr int kMaxWordPremiumDistance = 4;
constexpr std::array<std::pair<int, int>, 4> kNeighbors = {{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}};

// The J/Q/X/Z tiles `rack` still holds after playing `m` (which places no blank).
std::vector<Tile> kept_heavy_tiles(const Rack& rack, const Move& m) {
  std::vector<Tile> kept;
  for (const char c : {'J', 'Q', 'X', 'Z'}) {
    const Tile t = Tile::from_char(c);
    int left = rack.count(t);
    for (int i = 0; i < m.num_glyphs(); ++i) left -= m.glyph(i).letter() == t;
    if (left > 0) kept.push_back(t);
  }
  return kept;
}

bool places_blank(const Move& m) {
  for (int i = 0; i < m.num_glyphs(); ++i)
    if (m.glyph(i).is_blank()) return true;
  return false;
}

// Whether `t` at the empty square (r, c) hooks: it touches a run on at least
// one axis, and every word it forms is valid. `board`'s caches must be built.
bool hooks_at(const Board& board, int r, int c, Tile t) {
  const CrossCheck vertical_run = board.cross_check_at(false, r, c);
  const CrossCheck horizontal_run = board.cross_check_at(true, c, r);
  if (!vertical_run.has_neighbor && !horizontal_run.has_neighbor) return false;
  const uint32_t bit = 1u << t.index();
  return (vertical_run.mask & bit) != 0 && (horizontal_run.mask & bit) != 0;
}

// An empty word-premium square within reach of (r, c) along one axis: where a
// play through the square, perpendicular to the hook, would land.
bool word_premium_near(const Board& board, int r, int c, bool along_rows) {
  for (int d = -kMaxWordPremiumDistance; d <= kMaxWordPremiumDistance; ++d) {
    const int rr = along_rows ? r + d : r;
    const int cc = along_rows ? c : c + d;
    if (d == 0 || !board.in_bounds(rr, cc) || !board.at(rr, cc).is_empty()) continue;
    if (board.premium_at(rr, cc).word_mult() > 1) return true;
  }
  return false;
}

// `hook_is_horizontal`: the placed tile sits in the square's own row, so a play
// through the square runs along the rows of its column.
bool is_critical(const Board& board, int r, int c, bool hook_is_horizontal) {
  const Premium p = board.premium_at(r, c);
  if (p == Premium::TLS || p == Premium::TWS) return true;
  return p == Premium::DLS && word_premium_near(board, r, c, /*along_rows=*/hook_is_horizontal);
}

struct SetupProbe {
  const Board& before;
  const Board& after;
  const std::vector<Tile>& kept;
};

// Whether the empty square (r, c), beside a tile the play just placed, is a
// spot the play made for one of the kept heavy tiles.
bool opens_spot(const SetupProbe& probe, int r, int c, bool hook_is_horizontal) {
  if (!is_critical(probe.after, r, c, hook_is_horizontal)) return false;
  for (const Tile t : probe.kept)
    if (hooks_at(probe.after, r, c, t) && !hooks_at(probe.before, r, c, t)) return true;
  return false;
}

}  // namespace

bool is_high_value_setup(const Board& before, const Dictionary& dict, const Rack& rack,
                         const Move& m) {
  if (m.type() != MoveType::PLAY || places_blank(m)) return false;
  const std::vector<Tile> kept = kept_heavy_tiles(rack, m);
  if (kept.empty()) return false;

  before.ensure_movegen_caches(dict);
  Board after = before;
  after.apply(m);
  after.ensure_movegen_caches(dict);
  const SetupProbe probe{before, after, kept};

  bool found = false;
  visit_placed_squares(m, [&](int pr, int pc) {
    for (const auto& [dr, dc] : kNeighbors) {
      const int r = pr + dr, c = pc + dc;
      if (found || !after.in_bounds(r, c) || !after.at(r, c).is_empty()) continue;
      found = opens_spot(probe, r, c, /*hook_is_horizontal=*/dr == 0);
    }
  });
  return found;
}

}  // namespace scribblez

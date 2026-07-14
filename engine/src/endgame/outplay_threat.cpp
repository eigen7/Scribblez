#include "endgame/outplay_threat.h"

#include "game/glyph.h"
#include "game/tile.h"

#include <algorithm>

namespace scribblez {

namespace {

// Add cell (r, c) to the halo, ignoring out-of-bounds cells.
void halo_add(OutplayHalo& h, int r, int c) {
  if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return;
  h.rows |= static_cast<uint16_t>(1u << r);
  h.cols |= static_cast<uint16_t>(1u << c);
  h.cells[r] |= static_cast<uint16_t>(1u << c);
}

// (row, col) of lane position `p` on the line `line`, in the play's orientation.
// For a horizontal play the line is a row and `p` a column; vertical is the swap.
void lane_cell(bool horizontal, int line, int p, int& r, int& c) {
  r = horizontal ? line : p;
  c = horizontal ? p : line;
}

bool lane_filled(const Board& board, bool horizontal, int line, int p) {
  int r, c;
  lane_cell(horizontal, line, p, r, c);
  return board.at(r, c).has_letter();
}

// Add lane position `p` (a cell of the main word or an in-line neighbor).
void add_lane_cell(OutplayHalo& h, bool horizontal, int line, int p) {
  int r, c;
  lane_cell(horizontal, line, p, r, c);
  halo_add(h, r, c);
}

// Add the cells just beyond each end of the maximal existing perpendicular run
// through lane position `p`. The cross-word o forms at a placed cell is that
// run plus the placed tile, and a tile can only join it at the run's ends -- so
// those end cells are exactly where a reply can rewrite the cross-word's
// legality or score. With no existing run the ends are the immediate
// orthogonal neighbors.
void add_cross_run_ends(OutplayHalo& h, const Board& board, bool horizontal, int line, int p) {
  int r, c;
  lane_cell(horizontal, line, p, r, c);
  const int dr = horizontal ? 1 : 0;
  const int dc = horizontal ? 0 : 1;
  int ra = r - dr, ca = c - dc;
  while (ra >= 0 && ca >= 0 && board.at(ra, ca).has_letter()) {
    ra -= dr;
    ca -= dc;
  }
  halo_add(h, ra, ca);
  int rb = r + dr, cb = c + dc;
  while (rb < BOARD_SIZE && cb < BOARD_SIZE && board.at(rb, cb).has_letter()) {
    rb += dr;
    cb += dc;
  }
  halo_add(h, rb, cb);
}

// The used-tile multiset of a play (each blank counts as a blank tile).
TileCounts move_used_counts(const Move& m) {
  TileCounts used;
  for (int i = 0; i < m.num_glyphs(); ++i) used.add(m.glyph(i).rack_tile());
  return used;
}

}  // namespace

OutplayHalo build_outplay_halo(const Board& board, const Move& o) {
  OutplayHalo h;
  const bool horizontal = o.horizontal();
  const int line = o.start();

  // Placed extent along the lane, plus the end cells of each placed tile's
  // existing perpendicular run (the only cells a reply can rewrite that
  // tile's cross-word from).
  int lo = BOARD_SIZE, hi = -1;
  uint16_t mask = o.square_mask();
  for (int p = 0; mask; ++p, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    lo = std::min(lo, p);
    hi = std::max(hi, p);
    add_cross_run_ends(h, board, horizontal, line, p);
  }

  // Extend the span through contiguous existing tiles: those played-through
  // cells are part of o's main word, so occupying past them can hook the word.
  while (lo - 1 >= 0 && lane_filled(board, horizontal, line, lo - 1)) --lo;
  while (hi + 1 < BOARD_SIZE && lane_filled(board, horizontal, line, hi + 1)) ++hi;
  for (int p = lo; p <= hi; ++p) add_lane_cell(h, horizontal, line, p);

  // The in-line cells just beyond each end can extend the main word.
  add_lane_cell(h, horizontal, line, lo - 1);
  add_lane_cell(h, horizontal, line, hi + 1);
  return h;
}

bool move_touches_halo(const OutplayHalo& halo, const Move& m) {
  bool touched = false;
  visit_placed_squares(m, [&](int r, int c) { touched = touched || halo.contains(r, c); });
  return touched;
}

uint8_t canonical_used_mask(const Rack& rack, const TileCounts& used) {
  TileCounts remaining = used;
  uint8_t mask = 0;
  const std::array<Tile, RACK_SIZE>& tiles = rack.tiles();
  for (int i = 0; i < rack.size(); ++i) {
    const Tile t = tiles[i];
    if (remaining.remove(t)) mask |= static_cast<uint8_t>(1u << i);
  }
  return mask;
}

OutplayThreats::OutplayThreats(const Board& board, const Rack& rack, const std::vector<Move>& plays)
    : board_(board),
      rack_(rack),
      plays_(plays),
      rack_counts_(rack.counts()),
      halos_(plays.size()),
      halo_ready_(plays.size(), 0) {
  play_mask_.reserve(plays.size());
  for (const Move& p : plays) play_mask_.push_back(canonical_used_mask(rack, move_used_counts(p)));
}

const OutplayHalo& OutplayThreats::halo(int play_index) {
  if (!halo_ready_[play_index]) {
    halos_[play_index] = build_outplay_halo(board_, plays_[play_index]);
    halo_ready_[play_index] = 1;
  }
  return halos_[play_index];
}

int32_t OutplayThreats::threat_after(const Move& m) {
  if (m.type() != MoveType::PLAY) return kNoThreat;  // a pass leaves the full rack

  // The leave after m, and its canonical mask: the bucket key of m's out-plays.
  TileCounts leave = rack_counts_;
  for (int i = 0; i < m.num_glyphs(); ++i) leave.remove(m.glyph(i).rack_tile());
  if (leave.empty()) return kNoThreat;  // m empties the rack: it is itself an out-play
  const uint8_t key = canonical_used_mask(rack_, leave);

  // Out-plays of the leave that survive m (m does not touch their halo). m's own
  // placement always touches its own halo, so it drops out here naturally.
  survivors_.clear();
  for (int j = 0; j < static_cast<int>(plays_.size()); ++j) {
    if (play_mask_[j] != key) continue;
    if (move_touches_halo(halo(j), m)) continue;
    survivors_.push_back(j);
  }

  // The unblockable pair with the largest guaranteed smaller score.
  int32_t best = kNoThreat;
  for (size_t a = 0; a < survivors_.size(); ++a) {
    for (size_t b = a + 1; b < survivors_.size(); ++b) {
      if (!halos_unblockable(halo(survivors_[a]), halo(survivors_[b]))) continue;
      const int32_t smaller =
        std::min<int32_t>(plays_[survivors_[a]].score(), plays_[survivors_[b]].score());
      best = std::max(best, smaller);
    }
  }
  return best;
}

}  // namespace scribblez

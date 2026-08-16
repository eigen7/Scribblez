#include "endgame/outplays.h"

#include "game/glyph.h"
#include "game/tile.h"

#include <algorithm>

namespace scribblez {

namespace {

// Add cell (r, c) to the halo, ignoring out-of-bounds cells.
void halo_add(OutplayHalo& h, int r, int c) {
  if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return;
  h.rows |= uint16_t(1u << r);
  h.cols |= uint16_t(1u << c);
  h.cells[r] |= uint16_t(1u << c);
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

bool by_entry_score_desc(const OutplayEntry& a, const OutplayEntry& b) {
  return a.move.score() > b.move.score();
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
    if (remaining.remove(t)) mask |= uint8_t(1u << i);
  }
  return mask;
}

LazyHalos::LazyHalos(const Board& board, const std::vector<Move>& plays)
    : board_(board), plays_(plays), halos_(plays.size()), ready_(plays.size(), 0) {}

const OutplayHalo& LazyHalos::operator[](int play_index) {
  if (!ready_[play_index]) {
    halos_[play_index] = build_outplay_halo(board_, plays_[play_index]);
    ready_[play_index] = 1;
  }
  return halos_[play_index];
}

int32_t best_surviving_score(const OutplaySet& set, const Move& m) {
  for (const OutplayEntry& e : set) {
    if (!move_touches_halo(e.halo, m)) return e.move.score();
  }
  return kNoOutplaySurvivor;
}

void assign_surviving(const OutplaySet& parent, const Move& m, OutplaySet& out) {
  out.clear();
  for (const OutplayEntry& e : parent) {
    if (!move_touches_halo(e.halo, m)) out.push_back(e);
  }
}

void collect_rack_outplays(const Board& board, const std::vector<Move>& plays, int rack_size,
                           OutplaySet& out) {
  out.clear();
  for (const Move& m : plays) {
    if (m.type() == MoveType::PLAY && m.num_glyphs() == rack_size)
      out.push_back({m, build_outplay_halo(board, m)});
  }
  std::sort(out.begin(), out.end(), by_entry_score_desc);
}

LeaveOutplays::LeaveOutplays(const Board& board, const Rack& rack, const std::vector<Move>& plays)
    : rack_(rack), plays_(plays), rack_counts_(rack.counts()), halos_(board, plays) {
  play_mask_.reserve(plays.size());
  for (const Move& p : plays) play_mask_.push_back(canonical_used_mask(rack, move_used_counts(p)));
}

void LeaveOutplays::collect_after(const Move& m, OutplaySet& out) {
  out.clear();

  // The leave after m, and its canonical mask: the bucket key of m's leave's
  // out-plays. A pass subtracts nothing, so its bucket is the full rack's.
  TileCounts leave = rack_counts_;
  for (int i = 0; i < m.num_glyphs(); ++i) leave.remove(m.glyph(i).rack_tile());
  if (leave.empty()) return;  // m empties the rack: the game ends with it
  const uint8_t key = canonical_used_mask(rack_, leave);

  // Out-plays of the leave that survive m (m does not touch their halo). m's own
  // placement always touches its own halo, so it drops out here naturally.
  for (int j = 0; j < int(plays_.size()); ++j) {
    if (play_mask_[j] != key) continue;
    if (move_touches_halo(halos_[j], m)) continue;
    out.push_back({plays_[j], halos_[j]});
  }
  std::sort(out.begin(), out.end(), by_entry_score_desc);
}

void OutplaySetStack::reset(int max_ply) {
  if (int(slots_.size()) < max_ply) slots_.resize(max_ply);
  root_[0].clear();
  root_[1].clear();
  cur_[0] = &root_[0];
  cur_[1] = &root_[1];
}

void OutplaySetStack::collect_root_replier(const Board& board, const std::vector<Move>& plays,
                                           int rack_size) {
  collect_rack_outplays(board, plays, rack_size, root_[1]);
}

void OutplaySetStack::push(const Move& m, LeaveOutplays* leave_outs, int stm, int child_ply) {
  Slot& slot = slots_[child_ply];
  slot.saved[0] = cur_[0];
  slot.saved[1] = cur_[1];
  if (leave_outs == nullptr) return;
  leave_outs->collect_after(m, slot.mover);
  assign_surviving(*cur_[1 - stm], m, slot.other);
  cur_[stm] = &slot.mover;
  cur_[1 - stm] = &slot.other;
}

void OutplaySetStack::pop(int child_ply) {
  const Slot& slot = slots_[child_ply];
  cur_[0] = slot.saved[0];
  cur_[1] = slot.saved[1];
}

}  // namespace scribblez
